# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""TorchAir bridge for the MiniCPM-o causal-convolution pack operator."""

from __future__ import annotations

import torch

_PACK_REGISTERED = False
_BLOCK_REGISTERED = False


def register_minicpmo_causal_conv_pack_converter() -> None:
    """Expose the packaged ACLNN operator as a TorchAir AscendIR node.

    Registration is lazy so importing vLLM Ascend on CPU-only tooling does not
    import TorchAir. The function is idempotent because model workers can ask
    for the same fixed-shape graph more than once during startup.
    """
    global _PACK_REGISTERED
    if _PACK_REGISTERED:
        return

    from torch_npu.dynamo import torchair
    from torchair.ge import custom_op

    op = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack.default

    @torchair.register_fx_node_ge_converter(op)
    def convert_minicpmo_causal_conv_pack(x, cache, meta_outputs=None):
        del meta_outputs
        return custom_op(
            "MinicpmoCausalConvPack",
            inputs={"x": x, "cache": cache},
            outputs=["packed", "new_cache"],
        )

    _PACK_REGISTERED = True


def register_minicpmo_causal_conv_block_converter() -> None:
    """Expose the FP32 910C two-Conv MIX kernel to TorchAir."""
    global _BLOCK_REGISTERED
    if _BLOCK_REGISTERED:
        return

    from torch_npu.dynamo import torchair
    from torchair.ge import custom_op

    op = torch.ops._C_ascend.npu_minicpmo_causal_conv_block.default

    @torchair.register_fx_node_ge_converter(op)
    def convert_minicpmo_causal_conv_block(
        hidden,
        conv_input,
        cnn_cache,
        gate_conv,
        conv1_weight,
        conv1_bias,
        norm_weight,
        norm_bias,
        conv2_weight,
        conv2_bias,
        meta_outputs=None,
    ):
        del meta_outputs
        return custom_op(
            "MinicpmoCausalConvBlock",
            inputs={
                "hidden": hidden,
                "conv_input": conv_input,
                "cnn_cache": cnn_cache,
                "gate_conv": gate_conv,
                "conv1_weight": conv1_weight,
                "conv1_bias": conv1_bias,
                "norm_weight": norm_weight,
                "norm_bias": norm_bias,
                "conv2_weight": conv2_weight,
                "conv2_bias": conv2_bias,
            },
            outputs=["hidden_out", "new_cache"],
        )

    _BLOCK_REGISTERED = True
