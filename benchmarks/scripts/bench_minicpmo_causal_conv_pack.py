# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MiniCPM-o causal packing cache layouts on Ascend.

The cache-major candidate exercises the 910B/C two-slot MTE2/MTE3 pipeline;
the channel-major control retains the Gather-based compatibility path.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from vllm_ascend.utils import enable_custom_op

_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _reference(
    x: torch.Tensor,
    channel_major_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, channels = x.shape
    history = torch.cat((channel_major_cache, x.transpose(1, 2)), dim=2)
    packed = torch.stack(
        [
            history[:, :, offset : offset + 3]
            .transpose(1, 2)
            .reshape(batch, channels * 3)
            for offset in range(frames)
        ],
        dim=1,
    ).reshape(batch * frames, channels * 3)
    return packed, x[:, -2:, :].transpose(1, 2).contiguous()


def _measure_trial_us(fn: Callable[[], object], iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1_000


def _measure_serialized_us(fn: Callable[[], object]) -> float:
    start = time.perf_counter_ns()
    fn()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / 1_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")
    args = parser.parse_args()

    enable_custom_op()
    torch.manual_seed(20260817)
    dtype = _DTYPES[args.dtype]
    x = torch.randn(2, 50, 512, device="npu", dtype=dtype)
    channel_major = torch.randn(2, 512, 2, device="npu", dtype=dtype)
    cache_major = channel_major.transpose(1, 2).contiguous()

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, channel_major)

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, cache_major)

    baseline_packed, baseline_cache = baseline()
    candidate_packed, candidate_cache = candidate()
    expected_packed, expected_channel_major_cache = _reference(x, channel_major)
    torch.testing.assert_close(baseline_packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(
        baseline_cache,
        expected_channel_major_cache,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(candidate_packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_cache,
        expected_channel_major_cache.transpose(1, 2).contiguous(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(candidate_packed, baseline_packed, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_cache.transpose(1, 2),
        baseline_cache,
        rtol=0,
        atol=0,
    )

    for _ in range(args.warmups):
        baseline()
        candidate()
    torch.npu.synchronize()

    baseline_us: list[float] = []
    candidate_us: list[float] = []
    baseline_serialized_us: list[float] = []
    candidate_serialized_us: list[float] = []
    for trial in range(args.trials):
        if trial % 2:
            candidate_us.append(_measure_trial_us(candidate, args.iterations))
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
            candidate_serialized_us.append(_measure_serialized_us(candidate))
            baseline_serialized_us.append(_measure_serialized_us(baseline))
        else:
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
            candidate_us.append(_measure_trial_us(candidate, args.iterations))
            baseline_serialized_us.append(_measure_serialized_us(baseline))
            candidate_serialized_us.append(_measure_serialized_us(candidate))

    baseline_median = statistics.median(baseline_us)
    candidate_median = statistics.median(candidate_us)
    print(f"dtype={args.dtype}")
    print(f"channel_major_median_us={baseline_median:.3f}")
    print(f"cache_major_pipelined_median_us={candidate_median:.3f}")
    print(f"speedup={baseline_median / candidate_median:.4f}x")
    print(f"channel_major_min_us={min(baseline_us):.3f}")
    print(f"cache_major_pipelined_min_us={min(candidate_us):.3f}")
    print(
        "channel_major_serialized_median_us="
        f"{statistics.median(baseline_serialized_us):.3f}"
    )
    print(
        "cache_major_pipelined_serialized_median_us="
        f"{statistics.median(candidate_serialized_us):.3f}"
    )


if __name__ == "__main__":
    main()
