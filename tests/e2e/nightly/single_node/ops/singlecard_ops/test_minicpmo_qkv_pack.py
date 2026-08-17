# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_minicpmo_qkv_pack(dtype: torch.dtype) -> None:
    enable_custom_op()
    torch.manual_seed(20260817)
    inputs = [torch.randn(2, 50, 512, device="npu", dtype=dtype) for _ in range(3)]

    outputs = torch.ops._C_ascend.npu_minicpmo_qkv_pack(*inputs)

    for source, packed in zip(inputs, outputs, strict=True):
        expected = source.reshape(2, 50, 8, 64).transpose(1, 2).contiguous()
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
