# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import mindspore
import mindspore.nn as nn
from mindspore import ops
from typing import Any, List, Tuple


def set_device_states(state):
    mindspore.set_rng_state(state)


def detach_and_grad(inputs):
    if not isinstance(inputs, tuple):
        raise RuntimeError("Only tuple supported")
    return tuple(
        ops.stop_gradient(x) if isinstance(x, mindspore.Tensor) else x
        for x in inputs
    )


def get_cpu_and_gpu_states():
    return mindspore.get_rng_state()


class ReverseFunction(nn.Cell):
    def __init__(self, l0, l1, l2, l3):
        super().__init__()
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, c0, c1, c2, c3, alpha0, alpha1, alpha2, alpha3):
        
        c0_out = self.l0(x, c1) + c0 * alpha0
        c1_out = self.l1(c0_out, c2) + c1 * alpha1
        c2_out = self.l2(c1_out, c3) + c2 * alpha2
        c3_out = self.l3(c2_out, None) + c3 * alpha3

        return x, c0_out, c1_out, c2_out, c3_out

    def bprop(self, x, c0, c1, c2, c3, alpha0, alpha1, alpha2, alpha3, out, dout):
        _, g0, g1, g2, g3 = dout

        # -------- l3 --------
        def l3_fn(c2):
            return self.l3(c2, None)

        grad_c2 = self.grad_op(l3_fn)(c2, g3)
        g2 = g2 + grad_c2
        g3_left = g3 * alpha3

        # -------- l2 --------
        def l2_fn(c1, c3):
            return self.l2(c1, c3)

        grad_c1, grad_c3 = self.grad_op(l2_fn)(c1, c3, ops.Squeeze(0)(g2))
        g1 = g1 + grad_c1
        g3_left = g3_left + grad_c3
        g2_left = g2 * alpha2

        # -------- l1 --------
        def l1_fn(c0, c2):
            return self.l1(c0, c2)

        grad_c0, grad_c2 = self.grad_op(l1_fn)(c0, c2, g1)
        g0 = g0 + grad_c0
        g2_left = g2_left + grad_c2
        g1_left = g1 * alpha1

        # -------- l0 --------
        def l0_fn(x, c1):
            return self.l0(x, c1)

        grad_x, grad_c1 = self.grad_op(l0_fn)(x, c1, g0)
        g1_left = g1_left + grad_c1
        g0_left = g0 * alpha0

        return (
            grad_x,
            g0_left,
            g1_left,
            g2_left,
            g3_left,
            None,
            None,
            None,
            None,
        )