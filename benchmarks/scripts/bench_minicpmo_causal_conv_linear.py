# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MiniCPM-o causal packing plus projection on Ascend."""

import argparse
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op


def _measure_trial_us(fn: Callable[[], object], iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1_000


def _warmup(fn: Callable[[], object], warmups: int) -> None:
    for _ in range(warmups):
        fn()
    torch.npu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    enable_custom_op()
    torch.manual_seed(20260815)
    x = torch.randn(2, 50, 512, device="npu", dtype=torch.float32)
    cache = torch.randn(2, 512, 2, device="npu", dtype=torch.float32)
    weight = torch.randn(512, 1536, device="npu", dtype=torch.float32) * 0.01
    bias = torch.randn(512, device="npu", dtype=torch.float32) * 0.01

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        packed, new_cache = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(x, cache)
        return F.linear(packed, weight, bias).reshape(2, 50, 512), new_cache

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_minicpmo_causal_conv_linear(x, cache, weight, bias)

    baseline_output, baseline_cache = baseline()
    fused_output, fused_cache = fused()
    torch.testing.assert_close(fused_output, baseline_output, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(fused_cache, baseline_cache, rtol=0, atol=0)

    _warmup(baseline, args.warmups)
    _warmup(fused, args.warmups)
    baseline_us = []
    fused_us = []
    for trial in range(args.trials):
        # Alternate order so thermal and frequency drift do not favor one path.
        if trial % 2 == 0:
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
            fused_us.append(_measure_trial_us(fused, args.iterations))
        else:
            fused_us.append(_measure_trial_us(fused, args.iterations))
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
    baseline_median = statistics.median(baseline_us)
    fused_median = statistics.median(fused_us)
    print(f"baseline_pack_linear_median_us={baseline_median:.3f}")
    print(f"fused_conv_linear_median_us={fused_median:.3f}")
    print(f"speedup={baseline_median / fused_median:.4f}x")
    print(f"baseline_min_us={min(baseline_us):.3f}")
    print(f"fused_min_us={min(fused_us):.3f}")


if __name__ == "__main__":
    main()
