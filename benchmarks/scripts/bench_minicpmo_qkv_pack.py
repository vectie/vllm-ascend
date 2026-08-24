# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MiniCPM-o QKV BSH-to-BNSD packing on Ascend."""

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


def _measure_us(fn: Callable[[], object], iterations: int) -> float:
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
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")
    args = parser.parse_args()

    enable_custom_op()
    torch.manual_seed(20260817)
    dtype = _DTYPES[args.dtype]
    inputs = [
        torch.randn(2, 50, 512, device="npu", dtype=dtype)
        for _ in range(3)
    ]

    def baseline() -> tuple[torch.Tensor, ...]:
        return tuple(value.reshape(2, 50, 8, 64).transpose(1, 2).contiguous() for value in inputs)

    def candidate() -> tuple[torch.Tensor, ...]:
        return torch.ops._C_ascend.npu_minicpmo_qkv_pack(*inputs)

    for actual, expected in zip(candidate(), baseline(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for _ in range(args.warmups):
        baseline()
        candidate()
    torch.npu.synchronize()

    baseline_us: list[float] = []
    candidate_us: list[float] = []
    for trial in range(args.trials):
        if trial % 2:
            candidate_us.append(_measure_us(candidate, args.iterations))
            baseline_us.append(_measure_us(baseline, args.iterations))
        else:
            baseline_us.append(_measure_us(baseline, args.iterations))
            candidate_us.append(_measure_us(candidate, args.iterations))

    baseline_median = statistics.median(baseline_us)
    candidate_median = statistics.median(candidate_us)
    print(f"dtype={args.dtype}")
    print(f"transpose_median_us={baseline_median:.3f}")
    print(f"qkv_pack_median_us={candidate_median:.3f}")
    print(f"speedup={baseline_median / candidate_median:.4f}x")
    print(f"transpose_min_us={min(baseline_us):.3f}")
    print(f"qkv_pack_min_us={min(candidate_us):.3f}")


if __name__ == "__main__":
    main()
