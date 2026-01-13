"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import mindspore as ms
from mindspore import dataset, nn, ops
from mindspore.dataset import vision, transforms
from mindcv import create_model
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models

import models.cifar as models
import models.imagenet as customized_models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# Models
default_model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

customized_models_names = sorted(
    name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name])
)

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    # choices=model_names,
                    help='model architecture: ' +
                         # ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--best-acc', default='0', type=float,
                    help='best accuracy')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# ms.set_context(device_target='CPU', device_id=0)
# ms.set_context(device_target='CPU')

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
ms.set_seed(args.manualSeed)
# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)
if args.gpu_id == -1:
    ms.set_context(device_target='CPU')
elif args.gpu_id == -2:
    ms.set_context(device_target='CPU')
    ms.set_context(mode=ms.GRAPH_MODE)
elif args.gpu_id == 2:
    ms.set_context(device_target='GPU')
    ms.set_context(mode=ms.GRAPH_MODE)
else:
    ms.set_context(device_target='GPU')

best_acc = args.best_acc  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = vision.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], is_hwc=False)

    temp_dataset = dataset.ImageFolderDataset(traindir).map(
        transforms.Compose([
            vision.Decode(),
            vision.Resize(256),
            vision.RandomCrop(224),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            normalize,
        ])
    )
    train_loader = dataset.GeneratorDataset(
        temp_dataset, shuffle=True, num_parallel_workers=args.workers, column_names=["image", "label"]
    ).batch(args.train_batch)

    temp_dataset = dataset.ImageFolderDataset(valdir).map(
        transforms.Compose([
            vision.Decode(),
            vision.Resize(256),
            # vision.Rescale(rescale=256, shift=0),
            vision.CenterCrop(224),
            vision.ToTensor(),
            normalize,
        ])
    )
    val_loader = dataset.GeneratorDataset(
        temp_dataset, shuffle=False, num_parallel_workers=args.workers, column_names=["image", "label"]
    ).batch(args.test_batch)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
        )
    elif args.arch == "resnet":
        # print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](depth=20)
        model = create_model("resnet18", num_classes=1000)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.trainable_params()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.SGD(
        model.trainable_params(), learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        param_dict = ms.load_checkpoint(args.resume)
        ms.load_param_into_net(model, param_dict)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best ACC'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    def forward_fn(data, label):
        logits = model(data)
        _loss = criterion(logits, label)
        return _loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, grad_fn, optimizer)
        test_loss, test_acc = test(val_loader, model, criterion)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, best_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'acc': test_acc,
        #     'best_acc': best_acc,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, checkpoint=args.checkpoint)
        save_checkpoint(model, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(train_loader, model, grad_fn, optimizer):
    # switch to train mode
    model.set_train(mode=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        (loss, outputs), grads = grad_fn(inputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.numpy().item(), inputs.shape[0])
        top1.update(prec1.numpy().item(), inputs.shape[0])
        top5.update(prec5.numpy().item(), inputs.shape[0])

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer(grads)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | Best ACC{best_acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
            best_acc=best_acc
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(val_loader, model, criterion):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # model.eval()
    model.set_train(mode=False)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.numpy().item(), inputs.shape[0])
        top1.update(prec1.numpy().item(), inputs.shape[0])
        top5.update(prec5.numpy().item(), inputs.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | Best ACC{best_acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
            best_acc=best_acc,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(obj, is_best, checkpoint='checkpoint', filename='checkpoint.ckpt'):
    filepath = os.path.join(checkpoint, filename)
    ms.save_checkpoint(obj, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.ckpt'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = state['lr']
        ops.assign(optimizer.learning_rate, ms.Tensor(state['lr'], ms.float32))


if __name__ == '__main__':
    main()
