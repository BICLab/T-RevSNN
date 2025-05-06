# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import copy
import mindspore
import mindspore.nn as nn
from models.ms_spike_modules import Quant, DownSampling, UpSampling, MS_SpikeConvNextBlock, ScaledStdConv2d
from models.ms_rev_function import ReverseFunction
# from timm.models.layers import trunc_normal_


class Fusion(nn.Cell):
    def __init__(self, level, channels, first_col) -> None:
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
            self.up = (
                UpSampling(1, channels[level + 1], channels[level], 1, 1, 0)
                if level in [0, 1]
                else nn.Identity()
            )
            if level == 2:
                self.up = DownSampling(channels[level + 1], channels[level], 1, 1, 0)

            # self.tau = (
            #     nn.Parameter(
            #         shortcut_scale_init_value * mindspore.ones((1, channels[level], 1, 1)),
            #         requires_grad=True,
            #         # requires_grad=False,
            #     )
            #     if shortcut_scale_init_value > 0
            #     else None
            # )

    def construct(self, *args):
        c_down, c_up = args

        if self.first_col:
            x = self.down(c_down)
            return x

        if self.level == 3:
            x = self.down(c_down)
        else:
            # x = self.up(c_up) * self.tau + self.down(c_down)
            x = self.up(c_up) + self.down(c_down)
        return x


class MS_SpikeConvNextBlock_Reuse(nn.Cell):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in Pymindspore

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        in_channel,
        hidden_dim,
        out_channel,
        kernel_size=3,
        layer_scale_init_value=1,
        drop_path=0.00,
        pwconv2_reuse=None,
        dwconv2_reuse=None,
    ):
        super().__init__()
        self.dwconv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            groups=in_channel,
        )
        self.pwconv1 = ScaledStdConv2d(in_channel, hidden_dim, 1)

        self.act_learn = 1

        assert pwconv2_reuse is not None
        assert dwconv2_reuse is not None
        self.pwconv2_reuse = copy.copy(pwconv2_reuse)
        self.dwconv2_reuse = copy.copy(dwconv2_reuse)

        # self.gamma = (
        #     nn.Parameter(
        #         layer_scale_init_value * mindspore.ones((1, out_channel, 1, 1)), requires_grad=True
        #     )
        #     if layer_scale_init_value > 0
        #     else None
        # )

    def construct(self, x):
        shortcut = x
        if self.training:
            x = Quant.apply(x)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round_()
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        if self.training:
            x = Quant.apply(x)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round_()
        x = self.pwconv2_reuse(x)
        x = self.dwconv2_reuse(x)
        # x = shortcut + x * self.gamma
        x = shortcut + x
        return x


class Level(nn.Cell):
    def __init__(
        self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, pwconv2_reuse=None, dwconv2_reuse=None,
    ) -> None:
        super().__init__()
        assert pwconv2_reuse is not None
        assert dwconv2_reuse is not None
        self.fusion = Fusion(level, channels, first_col)

        # for conv
        modules = [
            MS_SpikeConvNextBlock(
                channels[level],
                int(4 * channels[level]),
                channels[level],
                kernel_size=kernel_size,
            ),
            MS_SpikeConvNextBlock_Reuse(
                channels[level],
                int(4 * channels[level]),
                channels[level],
                kernel_size=kernel_size,
                pwconv2_reuse=pwconv2_reuse,
                dwconv2_reuse=dwconv2_reuse,
            ),
        ]

        self.blocks = nn.Sequential(*modules)

    def construct(self, *args):
        x = self.fusion(*args)
        x = x.squeeze(0).contiguous()
        x = self.blocks(x)
        x = x.unsqueeze(0).contiguous()
        return x


class SubNet(nn.Cell):
    def __init__(
        self, channels, layers, kernel_size, first_col, dp_rates, save_memory, 
        pwconv2_reuse_stage0=None,
        dwconv2_reuse_stage0=None,
        pwconv2_reuse_stage1=None,
        dwconv2_reuse_stage1=None,
        pwconv2_reuse_stage2=None,
        dwconv2_reuse_stage2=None,
        pwconv2_reuse_stage3=None,
        dwconv2_reuse_stage3=None,
    ) -> None:
        super().__init__()
        assert pwconv2_reuse_stage0 is not None
        assert dwconv2_reuse_stage0 is not None
        assert pwconv2_reuse_stage1 is not None
        assert dwconv2_reuse_stage1 is not None
        assert pwconv2_reuse_stage2 is not None
        assert dwconv2_reuse_stage2 is not None
        assert pwconv2_reuse_stage3 is not None
        assert dwconv2_reuse_stage3 is not None
        self.save_memory = save_memory
        shortcut_scale_init_value = 0.5
        self.alpha0 = (
            nn.Parameter(
                shortcut_scale_init_value * mindspore.ones((1, channels[0], 1, 1)),
                requires_grad=True,
                # requires_grad=False,
            )
            if shortcut_scale_init_value > 0
            else None
        )
        self.alpha1 = (
            nn.Parameter(
                shortcut_scale_init_value * mindspore.ones((1, channels[1], 1, 1)),
                requires_grad=True,
                # requires_grad=False,
            )
            if shortcut_scale_init_value > 0
            else None
        )
        self.alpha2 = (
            nn.Parameter(
                shortcut_scale_init_value * mindspore.ones((1, channels[2], 1, 1)),
                requires_grad=True,
                # requires_grad=False,
            )
            if shortcut_scale_init_value > 0
            else None
        )
        self.alpha3 = (
            nn.Parameter(
                shortcut_scale_init_value * mindspore.ones((1, channels[3], 1, 1)),
                requires_grad=True,
                # requires_grad=False,
            )
            if shortcut_scale_init_value > 0
            else None
        )

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates, pwconv2_reuse=pwconv2_reuse_stage0, dwconv2_reuse=dwconv2_reuse_stage0)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates, pwconv2_reuse=pwconv2_reuse_stage1, dwconv2_reuse=dwconv2_reuse_stage1)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates, pwconv2_reuse=pwconv2_reuse_stage2, dwconv2_reuse=dwconv2_reuse_stage2)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates, pwconv2_reuse=pwconv2_reuse_stage3, dwconv2_reuse=dwconv2_reuse_stage3)

    def _construct_reverse(self, *args):
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        # _, c0, c1, c2, c3 = SpikeReverseFunction.apply(local_funs, *args)
        _, c0, c1, c2, c3 = ReverseFunction.apply(local_funs, alpha, *args)
        return c0, c1, c2, c3

    def construct(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._construct_reverse(*args)
        else:
            return self._construct_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with mindspore.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign


class Classifier(nn.Cell):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            ScaledStdConv2d(in_channels, num_classes, 1)
        )
    def construct(self, x):
        x = x.squeeze(0)
        x = x.mean(dim=[-1, -2], keepdim=True)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


class SpikeClassifier(nn.Cell):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.act_learn = 1
        self.classifier1 = nn.Sequential(
            ScaledStdConv2d(in_channels, num_classes, 1)
        )
        self.classifier2 = ScaledStdConv2d(num_classes, num_classes, 1)

    def construct(self, x):
        # NOTE: only for T = 1
        if self.training:
            x = Quant.apply(x, 0, 1)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round_()
        x = x.squeeze(0).mean(dim=[-1, -2], keepdim=True)
        x = self.classifier1(x)
        x = mindspore.nn.functional.leaky_relu(x, self.act_learn)
        x = self.classifier2(x).view(x.size(0), -1)
        return x


class MS_PatchEmbed(nn.Cell):
    def __init__(self, channels, num_subnet) -> None:
        super().__init__()
        self.act_learn = 1
        self.num_subnet = num_subnet
        # TODO
        self.down1 = nn.Sequential(
            ScaledStdConv2d(3, channels, kernel_size=7, stride=2, padding=3),
        )
        self.down2 = nn.Sequential(
            ScaledStdConv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        )

    def construct(self, x):
        x = self.down1(x)
        x = mindspore.nn.functional.leaky_relu(x, self.act_learn)
        x = self.down2(x)
        return None, x


class MS_PatchEmbed_DVS(nn.Cell):
    def __init__(self, channels, num_subnet) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.down = nn.Sequential(
            ScaledStdConv2d(2, channels, kernel_size=3, stride=1, padding=1),
        )

    def construct(self, x):
        x = self.down(x)
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
    ) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.inter_supv = inter_supv
        self.channels = channels
        self.layers = layers
        self.dvs = dvs

        if dvs:
            self.stem = nn.CellList(
                [
                    MS_PatchEmbed_DVS(channels[0], num_subnet=num_subnet) 
                    for _ in range(num_subnet)
                ]
            )
        else:
            self.stem = MS_PatchEmbed(channels[0], num_subnet=num_subnet)

        dp_rate = [x.item() for x in mindspore.linspace(0, drop_path, sum(layers))]

        pwconv2_reuse_stage0 = ScaledStdConv2d(channels[0] * 4, channels[0], 1)
        dwconv2_reuse_stage0 = nn.Conv2d(
            channels[0],
            channels[0],
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            groups=channels[0],
        )
        pwconv2_reuse_stage1 = ScaledStdConv2d(channels[1] * 4, channels[1], 1)
        dwconv2_reuse_stage1 = nn.Conv2d(
            channels[1],
            channels[1],
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            groups=channels[1],
        )
        pwconv2_reuse_stage2 = ScaledStdConv2d(channels[2] * 4, channels[2], 1)
        dwconv2_reuse_stage2 = nn.Conv2d(
            channels[2],
            channels[2],
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            groups=channels[2],
        )
        pwconv2_reuse_stage3 = ScaledStdConv2d(channels[3] * 4, channels[3], 1)
        dwconv2_reuse_stage3 = nn.Conv2d(
            channels[3],
            channels[3],
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            groups=channels[3],
        )
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(
                f"subnet{str(i)}",
                SubNet(
                    channels,
                    layers,
                    kernel_size,
                    first_col=first_col,
                    dp_rates=dp_rate,
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
            )

        if not inter_supv:
            self.cls_blocks = SpikeClassifier(in_channels=channels[-1], num_classes=num_classes)
        else:
            self.cls_blocks = nn.CellList(
                [
                    Classifier(in_channels=channels[-1], num_classes=num_classes),
                    Classifier(in_channels=channels[-1], num_classes=num_classes),
                    Classifier(in_channels=channels[-1], num_classes=num_classes),
                    SpikeClassifier(in_channels=channels[-1], num_classes=num_classes),
                ]
            )

        self.apply(self._init_weights)

    def construct(self, x):
        if self.dvs:
            return self._construct_intermediate_supervision_dvs(x)
        else:
            return self._construct_intermediate_supervision(x)

    def _construct_intermediate_supervision_dvs(self, imgs):
        x_cls_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        imgs = imgs.chunk(self.num_subnet, dim=1)
        c3s = 0.
        for i in range(self.num_subnet):
            _, x = self.stem[i](imgs[i].squeeze(1))
            c0, c1, c2, c3 = getattr(self, f"subnet{str(i)}")(x.unsqueeze(0), c0, c1, c2, c3)
            c3s += c3

        x_cls_out.append(self.cls_blocks(c3 / self.num_subnet))
        return x_cls_out

    def _construct_intermediate_supervision(self, img):
        x_cls_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4
        _, x = self.stem(img)
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f"subnet{str(i)}")(x.unsqueeze(0), c0, c1, c2, c3)

            if (i + 1) % interval == 0:
                x_cls_out.append(self.cls_blocks[i // interval](c3))

        return x_cls_out

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def change_act(self, m):
        for module in self.modules():
            if hasattr(module, "act_learn"):
                self.act_learn = m


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


def tiny_dvs(
    save_memory, inter_supv=True, drop_path=0.0, num_classes=1000, kernel_size=3
):
    channels = [32, 64, 128, 256]
    layers = [1, 1, 1, 1]
    num_subnet = 8
    return FullNet(
        channels,
        layers,
        num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        kernel_size=kernel_size,
        dvs=True,
    )


##-------------------------------------- Small -----------------------------------------


def small(
    save_memory, inter_supv=True, drop_path=0.0, num_classes=1000, kernel_size=3
):
    channels = [96, 192, 384, 512]
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
