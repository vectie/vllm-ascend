# MiniCPM-o 4.5 Ascend 910C candidate matrix

This matrix keeps hardware-oriented candidates independent until a real 910C
run proves both operator correctness and whole-service benefit. Profiler runs
characterize the device; unprofiled runs decide benchmark acceptance.

## Reserved rounds

| Round | Candidate | Intended 910C effect | Default status |
|---|---|---|---|
| R1 | Hardware counter profile | Attribute idle AIC/AIV, MTE stalls, L2 pressure, and layout conversions | Diagnostic only |
| R2 | Pipelined causal pack | Replace per-tap DMA and `PIPE_ALL` with two UB rows and exact MTE2/MTE3 events | Candidate |
| R3 | Causal tiling and cache publication | Launch only non-empty AIVs and publish rolling cache from row owners without a GM reread | Candidate |
| R4 | Ping-pong whole-head QKV DMA | Replace 50 per-frame copies with one strided read and one contiguous write per head; overlap K/Q and V/K transfers | Candidate |

R2 applies to both cache layouts. Channel-major uses Gather only for the first
two rows of each batch; its remaining 48 rows use the same contiguous pipeline
as cache-major. Compatibility UB is allocated only on cores that own a prefix
row or publish the channel-major rolling cache.

## Operator gates

Build the extension for the target 910C image, then run the exact-output suite:

```bash
pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_minicpmo_causal_conv_pack.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_minicpmo_qkv_pack.py
```

Measure all supported compute dtypes without profiling:

```bash
for dtype in float32 bfloat16 float16; do
  python benchmarks/scripts/bench_minicpmo_causal_conv_pack.py --dtype "$dtype"
  python benchmarks/scripts/bench_minicpmo_qkv_pack.py --dtype "$dtype"
done
```

Record median, minimum, and serialized latency. Reject a candidate on any
non-zero output drift. A microbenchmark win is evidence for continuing to the
service gate, not permission to enable the operator by default.

## Hardware characterization

Enable counters only for a short, bounded trace:

```bash
export VLLM_ASCEND_PROFILER_L2_CACHE=1
export VLLM_ASCEND_PROFILER_OP_ATTR=1
```

Start vLLM with a torch profiler directory, collect the same fixed Seed-TTS
request, and inspect:

- AIV MTE2 and MTE3 utilization and wait time;
- AIV Vector utilization for channel-major prefix Gather;
- AIC/AIV overlap around the graph-visible Conv+MLP partition;
- L2 hit rate and GM bytes for causal pack and QKV layout conversion;
- `TransData` and `Transpose` time before and after the candidate;
- kernel count and host launch gaps.

Disable profiler and both counter switches before measuring RTF, TTFT, TTFP,
or request duration.

## Service gates

The narrow causal candidate can be exercised without enabling the rejected
BF16/fixed-KV bundle:

```bash
export VLLM_OMNI_MINICPMO45_NPU_DIT_CONV_MLP_GRAPH=1
export VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK=1
```

Add cache-major as a separate paired run:

```bash
export VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR=1
```

QKV pack remains an independent experiment:

```bash
export VLLM_OMNI_MINICPMO45_NPU_DIT_QKV_PACK=1
```

Run the official request at least five times per candidate in alternating
baseline/candidate order. Retain only when mean chunk RTF, full request RTF,
and P99 do not regress, and then run Daily-Omni, TTS-Seed, and Video-MME. No
candidate becomes a submission default before the accuracy drop is within the
competition's two-percentage-point limit.

## Result ledger

Fill this table only from artifacts produced on the same machine and image.

| Round | Image/CANN/torch_npu | Operator median | Mean chunk RTF | P99 chunk RTF | Accuracy | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | pending | pending | pending | pending | pending | pending |
| R2 | pending | pending | pending | pending | pending | pending |
| R2+R3 | pending | pending | pending | pending | pending | pending |
| R4 | pending | pending | pending | pending | pending | pending |
