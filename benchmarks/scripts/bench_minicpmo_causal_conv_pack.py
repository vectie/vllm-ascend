# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MiniCPM-o causal packing cache layouts on Ascend."""

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from vllm_ascend.utils import enable_custom_op


def _measure_trial_us(fn: Callable[[], object], iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    enable_custom_op()
    torch.manual_seed(20260817)
    x = torch.randn(2, 50, 512, device="npu", dtype=torch.float32)
    channel_major = torch.randn(2, 512, 2, device="npu", dtype=torch.float32)
    cache_major = channel_major.transpose(1, 2).contiguous()

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, channel_major)

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, cache_major)

    baseline_packed, baseline_cache = baseline()
    candidate_packed, candidate_cache = candidate()
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
    for trial in range(args.trials):
        if trial % 2:
            candidate_us.append(_measure_trial_us(candidate, args.iterations))
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
        else:
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
            candidate_us.append(_measure_trial_us(candidate, args.iterations))

    baseline_median = statistics.median(baseline_us)
    candidate_median = statistics.median(candidate_us)
    print(f"channel_major_median_us={baseline_median:.3f}")
    print(f"cache_major_median_us={candidate_median:.3f}")
    print(f"speedup={baseline_median / candidate_median:.4f}x")
    print(f"channel_major_min_us={min(baseline_us):.3f}")
    print(f"cache_major_min_us={min(candidate_us):.3f}")


if __name__ == "__main__":
    main()
