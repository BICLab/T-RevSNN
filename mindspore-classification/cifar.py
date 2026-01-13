"""
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import mindspore as ms
from mindspore import nn, ops, dataset
from mindspore.dataset import vision
from mindspore.dataset.transforms import Compose
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import models.cifar as models
from models.cifar.ms_spike_rev_reuse import tiny as trevsnn_tiny

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
#                     choices=model_names,
#                     help='model architecture: ' +
#                          ' | '.join(model_names) +
#                          ' (default: resnet18)')
# only trevsnn_tiny
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument(
    '--block-name', type=str, default='BasicBlock',
    help='the building block for Resnet and Preresnet: '
         'BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)'
)
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--best-acc', default='0', type=float,
                    help='best accuracy')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# use_cuda = torch.cuda.is_available()
# ms.set_context(device_target='GPU', device_id=args.gpu_id)
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


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
ms.set_seed(args.manualSeed)

best_acc = args.best_acc  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = Compose([
        vision.RandomCrop(32, padding=4),
        vision.RandomHorizontalFlip(),
        vision.ToTensor(),
        vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False),
    ])

    transform_test = Compose([
        vision.ToTensor(),
        vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False),
    ])
    if args.dataset == 'cifar10':
        dataloader = dataset.Cifar10Dataset
        num_classes = 10
        col_names = ["image", "label"]
        ds_dir = './data/cifar-10'
    else:
        dataloader = dataset.Cifar100Dataset
        num_classes = 100
        col_names = ["image", "coarse_label", "fine_label"]
        ds_dir = './data/cifar-100'

    # trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainset = dataloader(dataset_dir=ds_dir, usage='train')
    trainset = trainset.map(transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    trainloader = dataset.GeneratorDataset(
        trainset, shuffle=True, num_parallel_workers=args.workers, column_names=col_names
    )
    trainloader = trainloader.batch(args.train_batch)

    # testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testset = dataloader(dataset_dir=ds_dir, usage='test')
    testset = testset.map(transform_test)
    # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader = dataset.GeneratorDataset(
        testset, shuffle=False, num_parallel_workers=args.workers, column_names=col_names
    )
    testloader = testloader.batch(args.test_batch)

    # Model
    # print("==> creating model '{}'".format(args.arch))
    print("==> creating model tiny trevsnn")
    args.arch = "tiny trevsnn"
    # if args.arch.startswith('resnext'):
    #     model = models.__dict__[args.arch](
    #         cardinality=args.cardinality,
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         widen_factor=args.widen_factor,
    #         # dropRate=args.drop,
    #     )
    # elif args.arch.startswith('densenet'):
    #     model = models.__dict__[args.arch](
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         growthRate=args.growthRate,
    #         compressionRate=args.compressionRate,
    #         dropRate=args.drop,
    #     )
    # elif args.arch.startswith('wrn'):
    #     model = models.__dict__[args.arch](
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         widen_factor=args.widen_factor,
    #         dropRate=args.drop,
    #     )
    # elif args.arch.endswith('resnet'):
    #     model = models.__dict__[args.arch](
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         block_name=args.block_name,
    #     )
    # else:
    #     model = models.__dict__[args.arch](num_classes=num_classes)
    model = trevsnn_tiny(num_classes=num_classes)

    # model = torch.nn.DataParallel(model)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.trainable_params()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        # checkpoint = torch.load(args.resume)
        param_dict = ms.load_checkpoint(args.resume)
        # param_not_load, _ = ms.load_param_into_net(model, param_dict)
        ms.load_param_into_net(model, param_dict)
        # best_acc = checkpoint['best_acc']
        # start_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best ACC'])

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = nn.SGD(
        model.trainable_params(), learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    def forward_fn(data, label):
        logits = model(data)
        _loss = criterion(logits, label)
        return _loss, logits
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, grad_fn, optimizer)
        test_loss, test_acc = test(testloader, model, criterion)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, best_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'acc': test_acc,
        #         'best_acc': best_acc,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, checkpoint=args.checkpoint)
        save_checkpoint(model, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, grad_fn, optimizer):
    # switch to train mode
    model.set_train(mode=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, others in enumerate(trainloader):
        if args.dataset == 'cifar10':
            (inputs, targets) = others
        else:
            (inputs, _, targets) = others
        # measure data loading time
        data_time.update(time.time() - end)
        targets = targets.astype(ms.int32)

        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        (loss, outputs), grads = grad_fn(inputs, targets)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        input_size = inputs.shape[0]
        losses.update(loss.numpy().item(), input_size)
        top1.update(prec1.numpy().item(), input_size)
        top5.update(prec5.numpy().item(), input_size)

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer(grads)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | '
                      'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | Best ACC:{best_acc: .4f}').format(
            batch=batch_idx + 1,
            size=len(trainloader),
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
    return losses.avg, top1.avg


def test(testloader, model, criterion):
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
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, others in enumerate(testloader):
        if args.dataset == 'cifar10':
            (inputs, targets) = others
        else:
            (inputs, _, targets) = others
        # measure data loading time
        data_time.update(time.time() - end)
        targets = targets.astype(ms.int32)

        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        inputs_size = inputs.shape[0]
        losses.update(loss.numpy().item(), inputs_size)
        top1.update(prec1.numpy().item(), inputs_size)
        top5.update(prec5.numpy().item(), inputs_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


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
