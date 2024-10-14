# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

from models.spike_rev_reuse import spike_revcol_tiny_reuse, spike_revcol_tiny_reuse_dvs, spike_revcol_small_reuse


def build_model(config):
    model_type = config.MODEL.TYPE

    ##-------------------------------------- tiny ----------------------------------------------------------------------------------------------------------------------#

    if model_type == "spike_revcol_tiny_reuse":
        model = spike_revcol_tiny_reuse(
            save_memory=config.REVCOL.SAVEMM,
            inter_supv=config.REVCOL.INTER_SUPV,
            drop_path=config.REVCOL.DROP_PATH,
            num_classes=config.MODEL.NUM_CLASSES,
            kernel_size=config.REVCOL.KERNEL_SIZE,
            kind=config.REVCOL.LEVEL_KIND,
        )

    elif model_type == "spike_revcol_tiny_reuse_dvs":
        model = spike_revcol_tiny_reuse_dvs(
            save_memory=config.REVCOL.SAVEMM,
            inter_supv=False,
            drop_path=config.REVCOL.DROP_PATH,
            num_classes=config.MODEL.NUM_CLASSES,
            kernel_size=config.REVCOL.KERNEL_SIZE,
            kind=config.REVCOL.LEVEL_KIND,
            num_subnet=config.REVCOL.NUM_SUBNET,
        )

    ##-------------------------------------- small ----------------------------------------------------------------------------------------------------------------------#
    
    elif model_type == "spike_revcol_small_reuse":
        model = spike_revcol_small_reuse(
            save_memory=config.REVCOL.SAVEMM,
            inter_supv=config.REVCOL.INTER_SUPV,
            drop_path=config.REVCOL.DROP_PATH,
            num_classes=config.MODEL.NUM_CLASSES,
            kernel_size=config.REVCOL.KERNEL_SIZE,
            kind=config.REVCOL.LEVEL_KIND,
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
