# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import os
import math
import torch
import numpy as np
import torch.distributed as dist
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import split_to_train_test_set

from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from .dvs_utils import Cutout, SNNAugmentWide


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])
    
    return train_idx, test_idx


def build_loader(config, num_subnet=10):
    config.defrost()
    dataset_train, _ = build_dataset(is_train=True, config=config, num_subnet=num_subnet)
    # dataset_train = list(dataset_train)
    config.freeze()
    print(f"global rank {dist.get_rank()} successfully build train dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        # persistent_workers=True,
    )

    # -----------------------------------Val Dataset-----------------------------------

    dataset_val, _ = build_dataset(is_train=False, config=config, num_subnet=num_subnet)
    # dataset_val = list(dataset_val)
    print(f"global rank {dist.get_rank()} successfully build val dataset")

    # indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        dataset_val, shuffle=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        # persistent_workers=True,
    )

    # setup mixup / cutmix
    mixup_fn = None
    # mixup_active = (
    #     config.AUG.MIXUP > 0
    #     or config.AUG.CUTMIX > 0.0
    #     or config.AUG.CUTMIX_MINMAX is not None
    # )
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP,
    #         cutmix_alpha=config.AUG.CUTMIX,
    #         cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB,
    #         switch_prob=config.AUG.MIXUP_SWITCH_PROB,
    #         mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING,
    #         num_classes=config.MODEL.NUM_CLASSES,
    #     )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config, num_subnet=10):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == "imagenet22K":
        if is_train:
            root = config.DATA.DATA_PATH
        else:
            root = config.DATA.EVAL_DATA_PATH
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 21841
    elif config.DATA.DATASET == "gesture":
        if is_train:
            transform = transforms.Compose(
                transforms.RandomHorizontalFlip(p=0.5),
                SNNAugmentWide(),
            )
            dataset = DVS128Gesture(
                config.DATA.DATA_PATH, 
                train=True, 
                data_type='frame', 
                frames_number=num_subnet, 
                split_by='number', 
                transform=transform,
            )
        else:
            dataset = DVS128Gesture(
                config.DATA.DATA_PATH, 
                train=False, 
                data_type='frame', 
                frames_number=num_subnet, 
                split_by='number', 
                # transform=DVSAug(None, train=False)
            )
        nb_classes = 11
    elif config.DATA.DATASET == "cifar10dvs":
        if is_train:
            random_crop = transforms.RandomCrop(
                64, padding=4
            )
            transform = transforms.Compose(
                [
                    SNNAugmentWide(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    random_crop,
                ]
            )
            dataset = CIFAR10DVS(
                config.DATA.DATA_PATH, 
                # train=True, 
                data_type='frame', 
                frames_number=num_subnet, 
                split_by='number', 
                transform=transform,
            )
            # idx, _ = split_to_train_test_set(0.9, dataset, 10)
            # dataset = torch.utils.data.Subset(dataset, idx)
        else:
            dataset = CIFAR10DVS(
                config.DATA.DATA_PATH, 
                # train=False, 
                data_type='frame', 
                frames_number=num_subnet, 
                split_by='number', 
                # transform=DVSAug(None, train=False)
            )
            _, idx = split_to_train_test_set(0.9, dataset, 10)
            dataset = torch.utils.data.Subset(dataset, idx)
        nb_classes = 10
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0
            else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != "none"
            else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4
            )
        return transform

    t = []
    if resize_im:
        if config.DATA.IMG_SIZE > 224:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            )
            print(f"Warping {config.DATA.IMG_SIZE} size input images...")
        elif config.TEST.CROP:
            size = int((232 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
