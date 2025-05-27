from typing import List, Tuple, Union
from itertools import repeat
import collections.abc

import mindspore
import mindspore.nn as nn


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
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(mindspore.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def construct(self, x):
        weight = mindspore.mint.nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            weight=(self.gain * self.scale).view(-1),
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        return mindspore.ops.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Quant(nn.Cell):
    def __init__(self):
        super(Quant, self).__init__()

    def construct(self, i, min_value=0, max_value=1):
        self.min = min_value
        self.max = max_value
        self.save_for_backward(i)
        return mindspore.floor(mindspore.clamp(i, min=min_value, max=max_value) + 0.5)

    def bprop(self, grad_output):
        grad_input = grad_output.clone()
        i, = self.saved_tensors
        grad_input[i < self.min] = 0
        grad_input[i > self.max] = 0
        return grad_input, None, None


class MS_SpikeConvNextBlock(nn.Cell):
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

        self.pwconv2 = ScaledStdConv2d(hidden_dim, out_channel, 1)
        self.dwconv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            groups=in_channel,
        )

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * mindspore.ones((1, out_channel, 1, 1)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

    def construct(self, x):
        shortcut = x
        if self.training:
            x = Quant(x)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round()
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        if self.training:
            x = Quant(x)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round()
        x = self.pwconv2(x)
        x = self.dwconv2(x)
        x = shortcut + x * self.gamma
        return x


class DownSampling(nn.Cell):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=False,
        reshape=True,
    ):
        super().__init__()
        self.reshape = reshape

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer

    def construct(self, x):
        if self.training:
            x = Quant(x)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round()
        if self.reshape:
            x = x.squeeze(0)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        if self.reshape:
            x = x.unsqueeze(0)
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

        out_dim = embed_dims
        self.encode_conv = nn.Conv2d(
            in_channels,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.encode_bn = nn.BatchNorm2d(out_dim)

        self.upsample = nn.Upsample(scale_factor=2**ratio, mode="nearest")

    def construct(self, x):
        if self.training:
            x = Quant(x)
        else:
            x = mindspore.clamp(x, 0, 1)
            x.round()
        x = x.squeeze(0)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        x = self.upsample(x)
        x = x.unsqueeze(0)
        return x
