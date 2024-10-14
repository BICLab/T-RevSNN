# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import math
import os, sys
import time
import json
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.cuda.amp as amp
from typing import Optional

from torchinfo import summary

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEma

from config import get_config
from models import *
from loss import *
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint,
    load_checkpoint_finetune,
    save_checkpoint,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor,
    adaptive_clip_grad,
)
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based.surrogate import ATan

logger = None

def parse_option():
    parser = argparse.ArgumentParser(
        "Spike RevCol training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size for single GPU"
    )
    parser.add_argument("--data-path", type=str, default="data", help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--finetune", help="finetune from checkpoint")

    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )

    parser.add_argument(
        "--output",
        default="outputs/",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")

    # ema
    parser.add_argument("--model-ema", action="store_true")

    # distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        required=False,
        help="local rank for DistributedDataParallel",
    )

    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    args, _ = parser.parse_known_args()
    # print(args)
    config = get_config(args)

    return args, config


def calc_non_zero_rate(s_dict, nz_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]
        x_shape = torch.tensor(list(v.shape))
        all_neural = torch.prod(x_shape)
        z = torch.nonzero(v)
        if k in nz_dict.keys():
            nz_dict[k] += (z.shape[0] / all_neural).item() / idx
        else:
            nz_dict[k] = (z.shape[0] / all_neural).item() / idx
    return nz_dict


def calc_firing_rate(s_dict, fr_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]
        if k in fr_dict.keys():
            fr_dict[k] += v.mean().item() / idx
        else:
            fr_dict[k] = v.mean().item() / idx
    return fr_dict


firing_dict = {}
def forward_hook_fn(module, input, output):
    global firing_dict
    firing_dict[module.name] = output.detach().data


def main(config, ngpus_per_node):
    config.defrost()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        return

    # linear scale the learning rate according to total batch size, base bs 1024
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * world_size / 1024.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * world_size / 1024.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * world_size / 1024.0
    )

    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    dist.init_process_group(
        backend="nccl",
        init_method=config.dist_url,
        world_size=world_size,
        rank=rank,
    )
    np.random.seed(config.SEED)
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.set_device(rank)
    global logger
    global firing_dict

    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
    )
    logger.info(config.dump())
    writer = None
    if dist.get_rank() == 0:
        writer = SummaryWriter(config.OUTPUT)

    (
        _,
        _,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)
    # data_loader_train_list = list(data_loader_train)
    # data_loader_val_list = list(data_loader_val)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    model = model.cuda()
    try:
        model_info = summary(model, (1, 3, 224, 224), verbose=0)
    except:
        model_info = summary(model, (1, model.num_subnet, 2, 32, 32), verbose=0)
    reset_net(model)
    logger.info(str(model_info))

    model_ema = None
    if config.MODEL_EMA:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        logger.info(f"Using EMA...")
        model_ema = ModelEma(
            model,
            decay=config.MODEL_EMA_DECAY,
        )

    optimizer = build_optimizer(config, model)
    if config.TRAIN.AMP:
        logger.info(
            f"-------------------------------Using Pytorch AMP...--------------------------------"
            # f"-------------------------------Using Apex AMP...--------------------------------"
        )
        # model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=128.0, combine_grad=True, combine_ddp=False)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        find_unused_parameters=False,
        output_device=config.LOCAL_RANK,
    )
    if "reuse" in config.MODEL.NAME:
        model._set_static_graph()
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # scaler = amp.GradScaler()
    scaler = amp.GradScaler(init_scale=32768)

    lr_scheduler = build_scheduler(config)

    # if config.AUG.MIXUP > 0.0:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    if config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # criterion_bce = torch.nn.BCEWithLogitsLoss()
    # criterion_bce = torch.nn.SmoothL1Loss()
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, logger, model_ema
        )

        logger.info(f"Start validation")
        acc1, acc5, _ = validate(
            config, data_loader_val, model, writer, epoch=config.TRAIN.START_EPOCH, firing_rate=False
        )

        logger.info(
            f"Accuracy of the network on the 50000 test images: {acc1:.1f}, {acc5:.1f}%"
        )

        if config.EVAL_MODE:
            return

    if config.MODEL.FINETUNE:
        load_checkpoint_finetune(config, model_without_ddp, logger)
        model_without_ddp.change_act(1.0)
        logger.info(f"Start validation")
        # for key, param in list(model.named_parameters()):
        #     logger.info(f"{key}")
        #     if "dwconv2_reuse.weight" in key:
        #         logger.info(f"{key}, {param.mean().item()}")

        acc1, acc5, _ = validate(
            config, data_loader_val, model, writer, epoch=config.TRAIN.START_EPOCH, firing_rate=False
        )
        # return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        if epoch < 150 and (not config.MODEL.FINETUNE):
            act_learn = 0.5 * (1 - math.cos(math.pi * epoch / 50)) * 1.0
        else:
            act_learn = 1.0
        model.module.change_act(act_learn)
        logger.info(f"Current act learn is: {act_learn:.4f}")

        model_ema = train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            writer,
            scaler,
            model_ema,
        )

        acc1, acc5, _ = validate(config, data_loader_val, model, writer, epoch)
        logger.info(
            f"Accuracy of the network on the 50000 test images: {acc1:.2f}, {acc5:.2f}%"
        )

        if config.MODEL_EMA:
            acc1_ema, acc5_ema, _ = validate_ema(
                config, data_loader_val, model_ema.ema, writer, epoch
            )
            logger.info(
                f"Accuracy of the EMA network on the 50000 test images: {acc1_ema:.1f}, {acc5_ema:.1f}%"
            )
            acc1 = max(acc1, acc1_ema)

        if dist.get_rank() == 0 and epoch % config.SAVE_FREQ == 0:
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                acc1,
                max_accuracy,
                optimizer,
                logger,
                model_ema,
            )

        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion_ce,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    writer,
    scaler,
    model_ema: Optional[ModelEma] = None,
):
    global logger
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    cls_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    data_time = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        data_time.update(time.time() - end)
        lr_scheduler.step_update(optimizer, idx / num_steps + epoch, config)
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        # with amp.autocast(enabled=config.TRAIN.AMP, dtype=torch.bfloat16):
        with amp.autocast(enabled=config.TRAIN.AMP):
            output_label = model(samples)

            if model.module.dvs:
                loss = TET_loss(output_label[-1], targets, criterion_ce, 1.0, 0.001)
            else:
                loss = compound_loss_only_cls(
                    output_label,
                    targets,
                    criterion_ce,
                    epoch,
                )

        if not math.isfinite(loss.item()):
            print(
                "Loss is {} in iteration {}, !".format(
                    loss.item(), idx
                )
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = get_grad_norm(filter(lambda p: p.requires_grad, model.parameters()))
        if config.TRAIN.CLIP_GRAD:
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), config.TRAIN.CLIP_GRAD
            # )
            adaptive_clip_grad(filter(lambda p: p.requires_grad, model.parameters()), config.TRAIN.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        cls_loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if dist.get_rank() == 0 and idx % 10 == 0:
            writer.add_scalar(
                "Train/loss", cls_loss_meter.val, epoch * num_steps + idx
            )
            writer.add_scalar(
                "Train/grad_norm", norm_meter.val, epoch * num_steps + idx
            )

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"datatime {data_time.val:.4f} ({data_time.avg:.4f})\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"cls loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler._scale.item()}\t"
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    torch.cuda.empty_cache()
    
    return model_ema


@torch.no_grad()
def validate(config, data_loader, model, writer, epoch, firing_rate=False):
    global logger
    # global firing_dict
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    # acc1_meter_list = []
    # acc5_meter_list = []
    # for i in range(4):
    #     acc1_meter_list.append(AverageMeter())
    #     acc5_meter_list.append(AverageMeter())

    end = time.time()
    # if firing_rate:
    #     T = getattr(model, "T", 1)
    #     fr_dict, nz_dict = {}, {}
    #     for i in range(T):
    #         fr_dict["t" + str(i)] = {}
    #         nz_dict["t" + str(i)] = {}
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        reset_net(model)
        outputs = model(images)

        # if len(acc1_meter_list) != len(outputs):
        #     acc1_meter_list = acc1_meter_list[: len(outputs)]
        #     acc5_meter_list = acc5_meter_list[: len(outputs)]

        output_last = outputs[-1]
        del outputs

        if model.module.dvs:
            output_last = output_last.mean(dim=0)
        loss = criterion(output_last, target)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), target.size(0))

        # for i, subnet_out in enumerate(outputs):
        #     acc1, acc5 = accuracy(subnet_out, target, topk=(1, 5))

        #     acc1 = reduce_tensor(acc1)
        #     acc5 = reduce_tensor(acc5)

        #     acc1_meter_list[i].update(acc1.item(), target.size(0))
        #     acc5_meter_list[i].update(acc5.item(), target.size(0))
        acc1, acc5 = accuracy(output_last, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # if firing_rate:
        #     for t in range(T):
        #         fr_single_dict = calc_firing_rate(
        #             firing_dict, fr_dict["t" + str(t)], len(data_loader) - 1, t
        #         )
        #         fr_dict["t" + str(t)] = fr_single_dict
        #         nz_single_dict = calc_non_zero_rate(
        #             firing_dict, nz_dict["t" + str(t)], len(data_loader) - 1, t
        #         )
        #         nz_dict["t" + str(t)] = nz_single_dict
        #     firing_dict = {}
        #     torch.cuda.epmty_cache()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                # f"Acc@1 {acc1_meter_list[-1].val:.3f} ({acc1_meter_list[-1].avg:.3f})\t"
                # f"Acc@1 {acc5_meter_list[-1].val:.3f} ({acc5_meter_list[-1].avg:.3f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )

    logger.info(
        # f" * Acc@1 {acc1_meter_list[-1].avg:.3f} Acc@5 {acc5_meter_list[-1].avg:.3f}"
        f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}"
    )
    if dist.get_rank() == 0:
        # for i in range(len(acc1_meter_list)):
        #     writer.add_scalar(f"Val_top1/acc_{i}", acc1_meter_list[i].avg, epoch)
        #     writer.add_scalar(f"Val_top5/acc_{i}", acc5_meter_list[i].avg, epoch)
        writer.add_scalar(f"Val_top1/acc", acc1_meter.avg, epoch)
        writer.add_scalar(f"Val_top5/acc", acc5_meter.avg, epoch)

        # if firing_rate:
        #     non_zero_str = json.dumps(nz_dict, indent=4)
        #     firing_rate_str = json.dumps(fr_dict, indent=4)
        #     logger.info("non-sero rate: ")
        #     logger.info(non_zero_str)
        #     logger.info("\n firing rate: ")
        #     logger.info(firing_rate_str)
    # return acc1_meter_list[-1].avg, acc5_meter_list[-1].avg, loss_meter.avg    
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def validate_ema(config, data_loader, model, writer, epoch):
    global logger
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        reset_net(model)
        outputs = model(images)
        output_last = outputs[-1]
        del outputs

        loss = criterion(output_last, target)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), target.size(0))

        acc1, acc5 = accuracy(output_last, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )

    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == "__main__":
    _, config = parse_option()

    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()

    main(config, ngpus_per_node)
