# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op


def test_minicpmo_final_adaln_fp32() -> None:
    enable_custom_op()
    torch.manual_seed(20260820)
    hidden = torch.randn(2, 50, 512, device="npu", dtype=torch.float32)
    modulation = torch.randn(2, 1, 1024, device="npu", dtype=torch.float32) * 0.1
    weight = torch.randn(80, 512, device="npu", dtype=torch.float32) * 0.01
    bias = torch.randn(80, device="npu", dtype=torch.float32) * 0.01

    actual = torch.ops._C_ascend.npu_minicpmo_final_adaln(
        hidden,
        modulation,
        weight,
        bias,
    )

    shift, scale = modulation.chunk(2, dim=-1)
    normalized = F.layer_norm(hidden, (512,), eps=1e-6)
    expected = F.linear(normalized * (1.0 + scale) + shift, weight, bias)
    error = (actual - expected).abs()
    assert error.max().item() <= 2e-3
    assert error.mean().item() <= 2e-4
