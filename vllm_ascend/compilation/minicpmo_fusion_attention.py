# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""TorchAir bridge for torch-npu fused-attention v3 inference graphs."""

from __future__ import annotations

import math

import torch

_REGISTERED = False


def register_minicpmo_fusion_attention_v3_converter() -> None:
    """Lower the MiniCPM-o BNSD/no-dropout v3 call to FlashAttentionScore.

    CANN exposes ``FlashAttentionScore`` in GE, but the TorchAir build used by
    the 910C competition image registers a converter only for the older
    seven-output ``npu_fusion_attention`` overload. PyTorch 2.10 SDPA selects
    the six-tensor v3 overload. Register its inference subset lazily so the
    complete DiT block can remain a single GE-visible graph.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from torch_npu.dynamo import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import Const, DataType

    op = torch.ops.npu.npu_fusion_attention_v3.default

    @torchair.register_fx_node_ge_converter(op)
    def convert_minicpmo_fusion_attention_v3(
        query,
        key,
        value,
        head_num,
        input_layout,
        pse=None,
        padding_mask=None,
        atten_mask=None,
        scale=1.0,
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        inner_precise=0,
        prefix=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        sparse_mode=0,
        gen_mask_parallel=True,
        sync=False,
        softmax_layout="",
        sink=None,
        meta_outputs=None,
    ):
        del gen_mask_parallel, sync, softmax_layout, sink, meta_outputs
        if input_layout not in {"BSH", "BNSD"} or not math.isclose(
            keep_prob, 1.0, rel_tol=1e-9
        ):
            raise NotImplementedError(
                "MiniCPM-o fused-attention v3 converter requires BSH/BNSD and keep_prob=1"
            )
        softmax_max, softmax_sum, softmax_out, attention_out = ge.FlashAttentionScore(
            query,
            key,
            value,
            real_shift=pse,
            drop_mask=None,
            padding_mask=padding_mask,
            atten_mask=atten_mask,
            prefix=prefix,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            q_start_idx=Const(0, dtype=DataType.DT_INT64),
            kv_start_idx=Const(0, dtype=DataType.DT_INT64),
            head_num=head_num,
            input_layout=input_layout,
            scale_value=scale,
            keep_prob=keep_prob,
            pre_tockens=pre_tockens,
            next_tockens=next_tockens,
            inner_precise=inner_precise,
            sparse_mode=sparse_mode,
        )
        seed = Const(0, dtype=DataType.DT_INT64)
        offset = Const(0, dtype=DataType.DT_INT64)
        return attention_out, softmax_max, softmax_sum, softmax_out, seed, offset

    _REGISTERED = True
