# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import mindspore
import mindspore.nn as nn
from typing import Any, List, Tuple


def get_gpu_device(*args):
    fwd_gpu_devices = list(
        set(
            arg.get_device()
            for arg in args
            if isinstance(arg, mindspore.Tensor) and arg.is_cuda
        )
    )
    return fwd_gpu_devices


def set_device_states(fwd_state) -> None:
    mindspore.set_rng_state(fwd_state)


def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[mindspore.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, mindspore.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


def get_cpu_and_gpu_states(gpu_devices):
    return mindspore.get_rng_state()


class ReverseFunction(nn.Cell):
    def __init__(self):
        super(ReverseFunction, self).__init__()

    def construct(self, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        self.run_functions = run_functions
        self.alpha = alpha
        self.preserve_rng_state = True

        # Hard code here
        self.gpu_autocast_kwargs = {
            "enabled": True,
            "dtype": mindspore.float16,
            "cache_enabled": True,
        }
        self.cpu_autocast_kwargs = {
            "enabled": True,
            "dtype": mindspore.float16,
            "cache_enabled": True,
        }

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            self.first_col = True
        else:
            self.first_col = False
        # gpu_devices = get_gpu_device(*args)
        # self.gpu_devices = gpu_devices
        self.cpu_states_0, self.gpu_states_0 = get_cpu_and_gpu_states()
        c0 = l0(x, c1) + c0 * alpha0
        self.cpu_states_1, self.gpu_states_1 = get_cpu_and_gpu_states()
        c1 = l1(c0, c2) + c1 * alpha1
        self.cpu_states_2, self.gpu_states_2 = get_cpu_and_gpu_states()
        c2 = l2(c1, c3) + c2 * alpha2
        self.cpu_states_3, self.gpu_states_3 = get_cpu_and_gpu_states()
        c3 = l3(c2, None) + c3 * alpha3
        self.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    def bprop(self, *grad_outputs):
        x, c0, c1, c2, c3 = self.saved_tensors
        l0, l1, l2, l3 = self.run_functions
        alpha0, alpha1, alpha2, alpha3 = self.alpha
        _, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        g3_up = g3_right
        g3_left = g3_up * alpha3  ##shortcut
        set_device_states(self.cpu_states_3, self.gpu_devices, self.gpu_states_3)
        oup3 = l3(c2, None)
        mindspore.grad(oup3, grad_position=g3_up)
        c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
        g2_up = g2_right + c2.grad
        g2_left = g2_up * alpha2  ##shortcut

        (c3_left,) = detach_and_grad((c3_left,))
        set_device_states(self.cpu_states_2, self.gpu_devices, self.gpu_states_2)
        oup2 = l2(c1, c3_left)
        mindspore.grad(oup2, grad_position=g2_up)
        c3_left.requires_grad = False
        cout3 = c3_left * alpha3  ##alpha3 update
        mindspore.grad(cout3, grad_position=g3_up)

        c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
        g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
        g1_up = g1_right + c1.grad
        g1_left = g1_up * alpha1  ##shortcut

        (c2_left,) = detach_and_grad((c2_left,))
        set_device_states(self.cpu_states_1, self.gpu_devices, self.gpu_states_1)
        oup1 = l1(c0, c2_left)
        mindspore.grad(oup1, grad_position=g1_up)
        c2_left.requires_grad = False
        cout2 = c2_left * alpha2  ##alpha2 update
        mindspore.grad(cout2, grad_position=g2_up)

        c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
        g0_up = g0_right + c0.grad
        g0_left = g0_up * alpha0  ##shortcut
        g2_left = (
            g2_left + c2_left.grad if c2_left.grad is not None else g2_left
        )  ## Fusion

        (c1_left,) = detach_and_grad((c1_left,))
        set_device_states(self.cpu_states_0, self.gpu_devices, self.gpu_states_0)
        oup0 = l0(x, c1_left)
        mindspore.grad(oup0, grad_position=g0_up)
        c1_left.requires_grad = False
        cout1 = c1_left * alpha1  ##alpha1 update
        mindspore.grad(cout1, grad_position=g1_up)

        c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
        gx_up = x.grad  ## Fusion
        g1_left = (
            g1_left + c1_left.grad if c1_left.grad is not None else g1_left
        )  ## Fusion
        c0_left.requires_grad = False
        cout0 = c0_left * alpha0  ##alpha0 update
        mindspore.grad(cout0, grad_position=g0_up)

        if self.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
