# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_major", [False, True])
@pytest.mark.parametrize("frames", [50, 64])
def test_minicpmo_causal_conv_pack(
    dtype: torch.dtype,
    cache_major: bool,
    frames: int,
) -> None:
    enable_custom_op()
    torch.manual_seed(20260814)
    x = torch.randn(2, frames, 512, device="npu", dtype=dtype)
    channel_major_cache = torch.randn(2, 512, 2, device="npu", dtype=dtype)
    cache = (
        channel_major_cache.transpose(1, 2).contiguous()
        if cache_major
        else channel_major_cache
    )

    packed, new_cache = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, cache)

    history = torch.cat((channel_major_cache, x.transpose(1, 2)), dim=2)
    expected = torch.stack(
        [
            history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1)
            for offset in range(frames)
        ],
        dim=1,
    ).reshape(2 * frames, 1536)
    expected_cache = x[:, -2:, :].contiguous()
    if not cache_major:
        expected_cache = expected_cache.transpose(1, 2).contiguous()
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(new_cache, expected_cache, rtol=0, atol=0)
