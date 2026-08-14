# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op


def test_minicpmo_causal_conv_linear_fp32() -> None:
    enable_custom_op()
    torch.manual_seed(20260815)
    x = torch.randn(2, 50, 512, device="npu", dtype=torch.float32)
    cache = torch.randn(2, 512, 2, device="npu", dtype=torch.float32)
    weight = torch.randn(512, 1536, device="npu", dtype=torch.float32) * 0.01
    bias = torch.randn(512, device="npu", dtype=torch.float32) * 0.01

    projected, new_cache = torch.ops._C_ascend.npu_minicpmo_causal_conv_linear(
        x,
        cache,
        weight,
        bias,
    )

    history = torch.cat((cache, x.transpose(1, 2)), dim=2)
    packed = torch.stack(
        [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
        dim=1,
    ).reshape(100, 1536)
    expected = F.linear(packed, weight, bias).reshape(2, 50, 512)
    expected_cache = x[:, -2:, :].transpose(1, 2).contiguous()

    torch.testing.assert_close(projected, expected, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(new_cache, expected_cache, rtol=0, atol=0)
