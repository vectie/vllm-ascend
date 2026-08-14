# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""TorchAir bridge for the MiniCPM-o causal-convolution pack operator."""

from __future__ import annotations

import torch

_PACK_REGISTERED = False
_BLOCK_REGISTERED = False
_LINEAR_REGISTERED = False


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


def register_minicpmo_causal_conv_linear_converter() -> None:
    """Expose the 910C causal pack + Cube projection operator to TorchAir."""
    global _LINEAR_REGISTERED
    if _LINEAR_REGISTERED:
        return

    from torch_npu.dynamo import torchair
    from torchair.ge import custom_op

    op = torch.ops._C_ascend.npu_minicpmo_causal_conv_linear.default

    @torchair.register_fx_node_ge_converter(op)
    def convert_minicpmo_causal_conv_linear(x, cache, weight, bias, meta_outputs=None):
        del meta_outputs
        return custom_op(
            "MinicpmoCausalConvLinear",
            inputs={"x": x, "cache": cache, "weight": weight, "bias": bias},
            outputs=["projected", "new_cache"],
        )

    _LINEAR_REGISTERED = True


def register_minicpmo_causal_conv_block_converter() -> None:
    """Lower the fused eager block into a GE-visible Conv graph.

    The ACLNN MIX kernel remains the eager implementation of
    ``npu_minicpmo_causal_conv_block``.  Replaying that kernel as a single GE
    custom node, however, hides its two matrix multiplications, LayerNorm,
    Mish, gated residual, and cache assembly from Graph Engine.  Keep only the
    model-specific causal packing as custom nodes and expose all compute-heavy
    work as native GE primitives so GE can schedule and optimize it together
    with the surrounding MLP.
    """
    global _BLOCK_REGISTERED
    if _BLOCK_REGISTERED:
        return

    from torch_npu.dynamo import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import Const, DataType, custom_op

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

        cache1, cache2 = ge.SplitV(
            cnn_cache,
            Const([512, 512], dtype=DataType.DT_INT64),
            Const(1, dtype=DataType.DT_INT32),
            num_split=2,
        )

        packed1, new_cache1 = custom_op(
            "MinicpmoCausalConvPack",
            inputs={"x": conv_input, "cache": cache1},
            outputs=["packed", "new_cache"],
        )
        convolution = ge.MatMulV2(
            packed1,
            conv1_weight,
            conv1_bias,
            None,
            transpose_x2=True,
        )
        convolution = ge.Reshape(
            convolution,
            Const([2, 50, 512], dtype=DataType.DT_INT64),
        )
        convolution, _, _ = ge.LayerNormV4(
            convolution,
            Const([512], dtype=DataType.DT_INT64),
            gamma=norm_weight,
            beta=norm_bias,
            epsilon=1e-5,
        )
        convolution = ge.Mish(convolution)

        packed2, new_cache2 = custom_op(
            "MinicpmoCausalConvPack",
            inputs={"x": convolution, "cache": cache2},
            outputs=["packed", "new_cache"],
        )
        convolution = ge.MatMulV2(
            packed2,
            conv2_weight,
            conv2_bias,
            None,
            transpose_x2=True,
        )
        convolution = ge.Reshape(
            convolution,
            Const([2, 50, 512], dtype=DataType.DT_INT64),
        )
        hidden_out = ge.Add(hidden, ge.Mul(gate_conv, convolution))
        new_cache = ge.ConcatV2(
            [new_cache1, new_cache2],
            Const(1, dtype=DataType.DT_INT32),
            N=2,
        )
        return hidden_out, new_cache

    _BLOCK_REGISTERED = True
