import torch
import torch.nn as nn
from timm.models.layers import ScaledStdConv2d


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=1):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.floor(torch.clamp(i, min=min_value, max=max_value) + 0.5)

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MS_SpikeConvNextBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

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
                layer_scale_init_value * torch.ones((1, out_channel, 1, 1)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        shortcut = x
        if self.training:
            x = Quant.apply(x)
        else:
            x = torch.clamp(x, 0, 1)
            x.round_()
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        if self.training:
            x = Quant.apply(x)
        else:
            x = torch.clamp(x, 0, 1)
            x.round_()
        x = self.pwconv2(x)
        x = self.dwconv2(x)
        x = shortcut + x * self.gamma
        return x


class DownSampling(nn.Module):
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

    def forward(self, x):
        if self.training:
            x = Quant.apply(x)
        else:
            x = torch.clamp(x, 0, 1)
            x.round_()
        if self.reshape:
            x = x.squeeze(0)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        if self.reshape:
            x = x.unsqueeze(0)
        return x


class UpSampling(nn.Module):
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

    def forward(self, x):
        if self.training:
            x = Quant.apply(x)
        else:
            x = torch.clamp(x, 0, 1)
            x.round_()
        x = x.squeeze(0)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        x = self.upsample(x)
        x = x.unsqueeze(0)
        return x
