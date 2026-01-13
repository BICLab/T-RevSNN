# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import copy
import mindspore
from mindspore import ops

import mindspore.nn as nn
from .ms_spike_modules import Quant, DownSampling, UpSampling, MS_SpikeConvNextBlock, ScaledStdConv2d
from .ms_rev_function import ReverseFunction


class Fusion(nn.Cell):
    def __init__(self, level, channels, first_col):
        super().__init__()
        self.level = level
        self.first_col = first_col

        self.down = (
            DownSampling(channels[level - 1], channels[level], 3, 2, 1)
            if level in [1, 2]
            else nn.Identity()
        )
        if level == 3:
            self.down = DownSampling(channels[level - 1], channels[level], 1, 1, 0)

        if not first_col and level != 3:
            if level in [0, 1]:
                self.up = UpSampling(1, channels[level + 1], channels[level], 1, 1, 0)
            elif level == 2:
                self.up = DownSampling(channels[level + 1], channels[level], 1, 1, 0)
            else:
                self.up = nn.Identity()

    def construct(self, c_down, c_up):
        if self.first_col:
            return self.down(c_down)

        if self.level == 3:
            return self.down(c_down)

        return self.up(c_up) + self.down(c_down)


class MS_SpikeConvNextBlock_Reuse(nn.Cell):
    def __init__(
        self,
        in_channel,
        hidden_dim,
        out_channel,
        kernel_size=3,
        pwconv2_reuse=None,
        dwconv2_reuse=None,
    ):
        super().__init__()

        self.quant = Quant()

        self.dwconv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            padding=kernel_size // 2,
            group=in_channel,
            has_bias=True,
            pad_mode="pad"
        )
        self.pwconv1 = ScaledStdConv2d(in_channel, hidden_dim, 1)

        # 参数共享
        self.pwconv2 = pwconv2_reuse
        self.dwconv2 = dwconv2_reuse

    def _quant(self, x):
        if self.training:
            return self.quant(x)
        return ops.Round()(ops.clip_by_value(x, 0.0, 1.0))

    def construct(self, x):
        shortcut = x
        x = self._quant(x)
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self._quant(x)
        x = self.pwconv2(x)
        x = self.dwconv2(x)
        return shortcut + x


class Level(nn.Cell):
    def __init__(
        self,
        level,
        channels,
        layers,
        kernel_size,
        first_col,
        pwconv2_reuse=None,
        dwconv2_reuse=None,
    ):
        super().__init__()

        self.fusion = Fusion(level, channels, first_col)

        self.blocks = nn.SequentialCell(
            MS_SpikeConvNextBlock(
                channels[level],
                4 * channels[level],
                channels[level],
                kernel_size=kernel_size,
            ),
            MS_SpikeConvNextBlock_Reuse(
                channels[level],
                4 * channels[level],
                channels[level],
                kernel_size=kernel_size,
                pwconv2_reuse=pwconv2_reuse,
                dwconv2_reuse=dwconv2_reuse,
            ),
        )

    def construct(self, c_down, c_up):
        x = self.fusion(c_down, c_up)
        x = ops.Squeeze(0)(x)
        x = self.blocks(x)
        x = ops.ExpandDims()(x, 0)
        return x


class SubNet(nn.Cell):
    def __init__(self, channels, kernel_size, first_col, save_memory,
                 pwconv2_reuse_stage0, dwconv2_reuse_stage0,
                 pwconv2_reuse_stage1, dwconv2_reuse_stage1,
                 pwconv2_reuse_stage2, dwconv2_reuse_stage2,
                 pwconv2_reuse_stage3, dwconv2_reuse_stage3):
        super().__init__()

        self.save_memory = save_memory

        init = 0.5
        self.alpha0 = mindspore.Parameter(init * ops.ones((1, channels[0], 1, 1)))
        self.alpha1 = mindspore.Parameter(init * ops.ones((1, channels[1], 1, 1)))
        self.alpha2 = mindspore.Parameter(init * ops.ones((1, channels[2], 1, 1)))
        self.alpha3 = mindspore.Parameter(init * ops.ones((1, channels[3], 1, 1)))

        self.level0 = Level(0, channels, None, kernel_size, first_col,
                            pwconv2_reuse_stage0, dwconv2_reuse_stage0)
        self.level1 = Level(1, channels, None, kernel_size, first_col,
                            pwconv2_reuse_stage1, dwconv2_reuse_stage1)
        self.level2 = Level(2, channels, None, kernel_size, first_col,
                            pwconv2_reuse_stage2, dwconv2_reuse_stage2)
        self.level3 = Level(3, channels, None, kernel_size, first_col,
                            pwconv2_reuse_stage3, dwconv2_reuse_stage3)
        
        self.rev_func = ReverseFunction(
            self.level0, self.level1, self.level2, self.level3
        )

    # def _clamp_alpha(self):
    #     for a in [self.alpha0, self.alpha1, self.alpha2, self.alpha3]:
    #         sign = ops.Sign()(a)
    #         a.set_data(sign * ops.maximum(ops.Abs()(a), 1e-3))

    def construct(self, x, c0, c1, c2, c3):
        # self._clamp_alpha()

        if self.save_memory:
            _, c0, c1, c2, c3 = self.rev_func(
                x, c0, c1, c2, c3, self.alpha0, self.alpha1, self.alpha2, self.alpha3
            )
            return c0, c1, c2, c3

        raise NotImplementedError


class Classifier(nn.Cell):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = ScaledStdConv2d(in_channels, num_classes, 1)

    def construct(self, x):
        x = ops.Squeeze(0)(x)
        x = ops.ReduceMean(keep_dims=True)(x, (-1, -2))
        x = self.conv(x)
        return x.reshape(x.shape[0], -1)


class SpikeClassifier(nn.Cell):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.quant = Quant()
        self.act = nn.LeakyReLU()
        self.conv1 = ScaledStdConv2d(in_channels, num_classes, 1)
        self.conv2 = ScaledStdConv2d(num_classes, num_classes, 1)

    def construct(self, x):
        x = self.quant(x) if self.training else ops.Round()(ops.clip_by_value(x, 0, 1))
        x = ops.Squeeze(0)(x)
        x = ops.ReduceMean(keep_dims=True)(x, (-1, -2))
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x.reshape(x.shape[0], -1)


class MS_PatchEmbed(nn.Cell):
    def __init__(self, channels, num_subnet):
        super().__init__()
        self.num_subnet = num_subnet
        self.conv1 = ScaledStdConv2d(3, channels, 7, stride=2, padding=3)
        self.conv2 = ScaledStdConv2d(channels, channels, 3, stride=2, padding=1)
        self.act = nn.LeakyReLU()

    def construct(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return None, x


class FullNet(nn.Cell):
    def __init__(
        self,
        channels=[32, 64, 96, 128],
        layers=[2, 3, 6, 3],
        num_subnet=5,
        kernel_size=3,
        num_classes=1000,
        drop_path=0.0,
        save_memory=True,
        inter_supv=True,
        dvs=False,
    ):
        super().__init__()
        self.num_subnet = num_subnet
        self.inter_supv = inter_supv
        self.dvs = dvs
        self.channels = channels

        self.stem = MS_PatchEmbed(channels[0], num_subnet=num_subnet)

        # # ---------- drop path ----------
        # dp_rate = ops.linspace(
        #     mindspore.Tensor(0, mindspore.float32),
        #     mindspore.Tensor(drop_path, mindspore.float32),
        #     sum(layers)
        # ).asnumpy().tolist()

        # ---------- reuse conv ----------
        pwconv2_reuse_stage0 = ScaledStdConv2d(channels[0] * 4, channels[0], 1)
        dwconv2_reuse_stage0 = nn.Conv2d(
            channels[0], channels[0], kernel_size,
            padding=kernel_size // 2, group=channels[0], pad_mode="pad"
        )
        pwconv2_reuse_stage1 = ScaledStdConv2d(channels[1] * 4, channels[1], 1)
        dwconv2_reuse_stage1 = nn.Conv2d(
            channels[1], channels[1], kernel_size,
            padding=kernel_size // 2, group=channels[1], pad_mode="pad"
        )
        pwconv2_reuse_stage2 = ScaledStdConv2d(channels[2] * 4, channels[2], 1)
        dwconv2_reuse_stage2 = nn.Conv2d(
            channels[2], channels[2], kernel_size,
            padding=kernel_size // 2, group=channels[2], pad_mode="pad"
        )
        pwconv2_reuse_stage3 = ScaledStdConv2d(channels[3] * 4, channels[3], 1)
        dwconv2_reuse_stage3 = nn.Conv2d(
            channels[3], channels[3], kernel_size,
            padding=kernel_size // 2, group=channels[3], pad_mode="pad"
        )

        # ---------- SubNets ----------
        self.subnets = nn.CellList([
            SubNet(
                channels,
                kernel_size,
                first_col=(i == 0),
                save_memory=save_memory,
                pwconv2_reuse_stage0=pwconv2_reuse_stage0,
                dwconv2_reuse_stage0=dwconv2_reuse_stage0,
                pwconv2_reuse_stage1=pwconv2_reuse_stage1,
                dwconv2_reuse_stage1=dwconv2_reuse_stage1,
                pwconv2_reuse_stage2=pwconv2_reuse_stage2,
                dwconv2_reuse_stage2=dwconv2_reuse_stage2,
                pwconv2_reuse_stage3=pwconv2_reuse_stage3,
                dwconv2_reuse_stage3=dwconv2_reuse_stage3,
            )
            for i in range(num_subnet)
        ])

        # ---------- classifier ----------
        if not inter_supv:
            self.cls_blocks = SpikeClassifier(channels[-1], num_classes)
        else:
            self.cls_blocks = nn.CellList([
                Classifier(channels[-1], num_classes),
                Classifier(channels[-1], num_classes),
                Classifier(channels[-1], num_classes),
                SpikeClassifier(channels[-1], num_classes),
            ])

        # ops
        self.split = ops.Split(axis=1, output_num=num_subnet)
        self.squeeze1 = ops.Squeeze(1)
        self.unsqueeze0 = ops.ExpandDims()

    def construct(self, x):
        if self.dvs:
            return self._forward_dvs(x)
        else:
            return self._forward(x)

    def _forward_dvs(self, imgs):
        outputs = []
        c0 = c1 = c2 = c3 = None

        imgs = self.split(imgs)  # tuple

        c3_sum = 0.0
        for i in range(self.num_subnet):
            _, x = self.stem[i](self.squeeze1(imgs[i]))
            c0, c1, c2, c3 = self.subnets[i](self.unsqueeze0(x, 0), c0, c1, c2, c3)
            c3_sum = c3_sum + c3

        outputs.append(self.cls_blocks(c3_sum / self.num_subnet))
        return outputs

    def _forward(self, img):
        outputs = []

        _, x = self.stem(img)
        interval = self.num_subnet // 4

        # c0 = c1 = c2 = c3 = 0
        c0 = ops.zeros((1, img.shape[0], self.channels[0], x.shape[2], x.shape[3]))
        c1 = ops.zeros((1, img.shape[0], self.channels[1], x.shape[2] // 2, x.shape[3] // 2))
        c2 = ops.zeros((1, img.shape[0], self.channels[2], x.shape[2] // 4, x.shape[3] // 4))
        c3 = ops.zeros((1, img.shape[0], self.channels[3], x.shape[2] // 4, x.shape[3] // 4))

        for i in range(self.num_subnet):
            c0, c1, c2, c3 = self.subnets[i](self.unsqueeze0(x, 0), c0, c1, c2, c3)
            if (i + 1) % interval == 0:
                outputs.append(self.cls_blocks[i // interval](c3))

        return outputs


##-------------------------------------- Tiny -----------------------------------------


def tiny(
    save_memory, inter_supv=True, drop_path=0.0, num_classes=1000, kernel_size=3
):
    channels = [64, 128, 256, 384]
    layers = [1, 1, 1, 1]
    num_subnet = 4
    return FullNet(
        channels,
        layers,
        num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        kernel_size=kernel_size,
    )
