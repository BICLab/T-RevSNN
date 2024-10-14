# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import torch
from torch.functional import Tensor


def compound_loss_only_cls(
    output_label,
    targets,
    criterion_ce,
    epoch,
):
    cls_loss = []
    for i, label in enumerate(output_label):
        ratio_c = (i + 1) / len(output_label)

        l = criterion_ce(label, targets) * ratio_c
        # if dist.get_rank() == 0:
        #     print(f'ihx: {ihx}, ihy: {ihy}')
        cls_loss.append(l)
        # feature_loss.append(torch.dist(output_feature[i], teacher_feature) *  feature_coe)
    final_cls_loss = criterion_ce(output_label[-1], targets)
    cls_loss.append(final_cls_loss)
    # print(feature_loss)
    loss = torch.sum(torch.stack(cls_loss), dim=0)
    del cls_loss
    return loss


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total
