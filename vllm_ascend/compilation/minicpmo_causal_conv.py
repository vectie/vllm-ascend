# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""TorchAir bridge for the MiniCPM-o causal-convolution pack operator."""

from __future__ import annotations

import torch

_REGISTERED = False


def register_minicpmo_causal_conv_pack_converter() -> None:
    """Expose the packaged ACLNN operator as a TorchAir AscendIR node.

    Registration is lazy so importing vLLM Ascend on CPU-only tooling does not
    import TorchAir. The function is idempotent because model workers can ask
    for the same fixed-shape graph more than once during startup.
    """
    global _REGISTERED
    if _REGISTERED:
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

    _REGISTERED = True
