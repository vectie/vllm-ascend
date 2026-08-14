# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_minicpmo_causal_conv_pack(dtype: torch.dtype) -> None:
    enable_custom_op()
    torch.manual_seed(20260814)
    x = torch.randn(2, 50, 512, device="npu", dtype=dtype)
    cache = torch.randn(2, 512, 2, device="npu", dtype=dtype)

    packed, new_cache = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, cache)

    history = torch.cat((cache, x.transpose(1, 2)), dim=2)
    expected = torch.stack(
        [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
        dim=1,
    ).reshape(100, 1536)
    expected_cache = x[:, -2:, :].transpose(1, 2).contiguous()
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(new_cache, expected_cache, rtol=0, atol=0)
