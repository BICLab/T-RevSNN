from typing import List, Tuple, Union
from itertools import repeat
import collections.abc

import mindspore
from mindspore import ops, nn


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> Union[int, List[int]]:
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class ScaledStdConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        gamma=1.0,
        eps=1e-6,
        gain_init=1.0,
        pad_mode="pad",
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            group=groups,
            has_bias=bias,
            pad_mode=pad_mode,
        )

        self.gain = mindspore.Parameter(
            ops.ones((out_channels, 1, 1, 1)) * gain_init
        )

        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gamma / fan_in ** 0.5
        self.eps = eps

        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.sqrt = ops.Sqrt()

    def construct(self, x):
        w = self.weight
        mean = self.reduce_mean(w, (1, 2, 3))
        var = self.reduce_mean((w - mean) ** 2, (1, 2, 3))
        w_norm = (w - mean) / self.sqrt(var + self.eps)
        w_norm = w_norm * (self.gain * self.scale)

        return ops.Conv2D(
            out_channel=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad=self.padding,
            dilation=self.dilation,
            group=self.group,
            pad_mode=self.pad_mode,
        )(x, w_norm)


class Quant(nn.Cell):
    def __init__(self):
        super().__init__()
        self.floor = ops.Floor()
        self.clip = ops.clip_by_value

    def construct(self, x, min_value=0.0, max_value=1.0):
        x_clip = self.clip(x, min_value, max_value)
        return self.floor(x_clip + 0.5)

    def bprop(self, x, min_value, max_value, out, dout):
        mask = (x >= min_value) & (x <= max_value)
        dx = dout * mask
        return dx, None, None


class MS_SpikeConvNextBlock(nn.Cell):
    def __init__(
        self,
        in_channel,
        hidden_dim,
        out_channel,
        kernel_size=3,
        layer_scale_init_value=1.0,
    ):
        super().__init__()

        self.quant = Quant()
        self.clip = ops.clip_by_value
        self.round = ops.Round()

        self.dwconv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            padding=kernel_size // 2,
            group=in_channel,
            has_bias=True,
            pad_mode="pad",
        )

        self.pwconv1 = ScaledStdConv2d(in_channel, hidden_dim, 1)
        self.pwconv2 = ScaledStdConv2d(hidden_dim, out_channel, 1)

        self.dwconv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size,
            padding=kernel_size // 2,
            group=out_channel,
            has_bias=True,
            pad_mode="pad",
        )

        self.gamma = (
            mindspore.Parameter(layer_scale_init_value * ops.ones((1, out_channel, 1, 1)))
            if layer_scale_init_value > 0
            else None
        )

    def _quant(self, x):
        if self.training:
            return self.quant(x)
        x = self.clip(x, 0.0, 1.0)
        return self.round(x)

    def construct(self, x):
        shortcut = x
        x = self._quant(x)
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self._quant(x)
        x = self.pwconv2(x)
        x = self.dwconv2(x)
        return shortcut + x * self.gamma


class DownSampling(nn.Cell):
    def __init__(
        self,
        in_channels,
        embed_dims,
        kernel_size=3,
        stride=2,
        padding=1,
        reshape=True,
    ):
        super().__init__()

        self.quant = Quant()
        self.reshape = reshape
        self.squeeze = ops.Squeeze(0)
        self.unsqueeze = ops.ExpandDims()

        self.conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size,
            stride=stride,
            padding=padding,
            has_bias=False,
            pad_mode="pad",
        )
        self.bn = nn.BatchNorm2d(embed_dims)

    def construct(self, x):
        x = self.quant(x) if self.training else ops.Round()(ops.clip_by_value(x, 0, 1))
        if self.reshape:
            x = self.squeeze(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.reshape:
            x = self.unsqueeze(x, 0)
        return x


class UpSampling(nn.Cell):
    def __init__(
        self,
        ratio=1,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        self.quant = Quant()

        self.squeeze = ops.Squeeze(0)
        self.unsqueeze = ops.ExpandDims()

        self.conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            has_bias=False,
            pad_mode="pad", 
        )
        self.bn = nn.BatchNorm2d(embed_dims)

        self.scale = 2 ** ratio

        self.clip = ops.clip_by_value
        self.round = ops.Round()

    def _quant(self, x):
        if self.training:
            return self.quant(x)
        x = self.clip(x, 0.0, 1.0)
        return self.round(x)

    def construct(self, x):
        # x: [B, C, H, W]
        x = self._quant(x)

        x = self.squeeze(x)        # [B, C, H, W]
        x = self.conv(x)           # [B, C', H, W]
        x = self.bn(x)

        _, _, h, w = x.shape
        new_h = h * self.scale
        new_w = w * self.scale

        x = ops.ResizeNearestNeighbor(size=(new_h, new_w))(x)
        x = self.unsqueeze(x, 0)            # [B, C', H', W']

        return x
