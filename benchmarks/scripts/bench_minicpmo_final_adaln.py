# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MiniCPM-o final AdaLN plus projection on Ascend."""

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
    torch.manual_seed(20260820)
    hidden = torch.randn(2, 50, 512, device="npu", dtype=torch.float32)
    modulation = torch.randn(2, 1, 1024, device="npu", dtype=torch.float32) * 0.1
    weight = torch.randn(80, 512, device="npu", dtype=torch.float32) * 0.01
    bias = torch.randn(80, device="npu", dtype=torch.float32) * 0.01

    def baseline() -> torch.Tensor:
        shift, scale = modulation.chunk(2, dim=-1)
        normalized = F.layer_norm(hidden, (512,), eps=1e-6)
        return F.linear(torch.addcmul(normalized + shift, normalized, scale), weight, bias)

    def fused() -> torch.Tensor:
        return torch.ops._C_ascend.npu_minicpmo_final_adaln(hidden, modulation, weight, bias)

    expected = baseline()
    actual = fused()
    torch.npu.synchronize()
    error = (actual - expected).abs()
    print(f"max_abs_error={error.max().item():.8f}")
    print(f"mean_abs_error={error.mean().item():.8f}")

    _warmup(baseline, args.warmups)
    _warmup(fused, args.warmups)
    baseline_us = []
    fused_us = []
    for trial in range(args.trials):
        if trial % 2 == 0:
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
            fused_us.append(_measure_trial_us(fused, args.iterations))
        else:
            fused_us.append(_measure_trial_us(fused, args.iterations))
            baseline_us.append(_measure_trial_us(baseline, args.iterations))
    baseline_median = statistics.median(baseline_us)
    fused_median = statistics.median(fused_us)
    print(f"baseline_final_adaln_median_us={baseline_median:.3f}")
    print(f"fused_final_adaln_median_us={fused_median:.3f}")
    print(f"speedup={baseline_median / fused_median:.4f}x")
    print(f"baseline_min_us={min(baseline_us):.3f}")
    print(f"fused_min_us={min(fused_us):.3f}")


if __name__ == "__main__":
    main()
