/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

constexpr uint32_t kBatch = 2;
constexpr uint32_t kFrames = 50;
constexpr uint32_t kChannels = 512;
constexpr uint32_t kKernel = 3;
constexpr uint32_t kPackedChannels = kChannels * kKernel;
constexpr uint32_t kTotalRows = kBatch * kFrames;
constexpr float kInvChannels = 1.0f / static_cast<float>(kChannels);
constexpr float kLayerNormEpsilon = 1.0e-5f;
constexpr MatmulConfig kMatmulConfig{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                     0, 0, 0, 0, true};

using AType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using ConvMatmul = matmul::MatmulImpl<AType, BType, CType, BiasType, kMatmulConfig>;

class MinicpmoCausalConvBlockKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR hidden,
        GM_ADDR convInput,
        GM_ADDR cnnCache,
        GM_ADDR gateConv,
        GM_ADDR conv1Weight,
        GM_ADDR conv1Bias,
        GM_ADDR normWeight,
        GM_ADDR normBias,
        GM_ADDR conv2Weight,
        GM_ADDR conv2Bias,
        GM_ADDR hiddenOut,
        GM_ADDR newCache,
        GM_ADDR workspace,
        const MinicpmoCausalConvBlockTilingData* tiling)
    {
        tiling_ = tiling;
        hiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hidden), kTotalRows * kChannels);
        convInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(convInput), kTotalRows * kChannels);
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cnnCache), kBatch * kChannels * 4);
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gateConv), kBatch * kChannels);
        conv1WeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(conv1Weight), kChannels * kPackedChannels);
        conv1BiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(conv1Bias), kChannels);
        normWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normWeight), kChannels);
        normBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normBias), kChannels);
        conv2WeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(conv2Weight), kChannels * kPackedChannels);
        conv2BiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(conv2Bias), kChannels);
        hiddenOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hiddenOut), kTotalRows * kChannels);
        newCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newCache), kBatch * kChannels * 4);

        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        __gm__ float* work = reinterpret_cast<__gm__ float*>(userWorkspace);
        packed1Gm_.SetGlobalBuffer(work + tiling->packed1Offset, kTotalRows * kPackedChannels);
        conv1Gm_.SetGlobalBuffer(work + tiling->conv1Offset, kTotalRows * kChannels);
        activatedGm_.SetGlobalBuffer(work + tiling->activatedOffset, kTotalRows * kChannels);
        packed2Gm_.SetGlobalBuffer(work + tiling->packed2Offset, kTotalRows * kPackedChannels);
        conv2Gm_.SetGlobalBuffer(work + tiling->conv2Offset, kTotalRows * kChannels);

        if ASCEND_IS_AIC {
            mm_.Init(&tiling->mmTiling, &pipe_);
        } else {
            // On Ascend 220, GetBlockIdx() already folds the AIV sub-block ID
            // into a unique logical Vector-core index for 1:2 mixed launches.
            vectorCore_ = GetBlockIdx();
            vectorCoreCount_ = tiling->usedCoreNum * 2;
            pipe_.InitBuffer(rowBuffer_, kPackedChannels * sizeof(float));
            pipe_.InitBuffer(workBuffer_, kChannels * 3 * sizeof(float));
            pipe_.InitBuffer(parameterBuffer_, kChannels * 4 * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            Pack(convInputGm_, 0, packed1Gm_);
        }
        SyncAll<false>();

        if ASCEND_IS_AIC {
            Matmul(conv1WeightGm_, conv1BiasGm_, packed1Gm_, conv1Gm_);
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            NormalizeAndMish();
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            Pack(activatedGm_, kChannels, packed2Gm_);
        }
        SyncAll<false>();

        if ASCEND_IS_AIC {
            Matmul(conv2WeightGm_, conv2BiasGm_, packed2Gm_, conv2Gm_);
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            ResidualAndCache();
        }
    }

private:
    __aicore__ inline void Matmul(
        const GlobalTensor<float>& weight,
        const GlobalTensor<float>& bias,
        const GlobalTensor<float>& input,
        GlobalTensor<float>& output)
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t nStart = core * tiling_->nPerCore;
        mm_.SetOrgShape(kTotalRows, kChannels, kPackedChannels);
        mm_.SetSingleShape(kTotalRows, tiling_->nPerCore, kPackedChannels);
        mm_.SetTensorA(input, false);
        mm_.SetTensorB(weight[nStart * kPackedChannels], true);
        mm_.SetBias(bias[nStart]);
        mm_.template IterateAll<false>(output[nStart], 0);
    }

    __aicore__ inline void Pack(
        const GlobalTensor<float>& source,
        uint32_t cacheChannelOffset,
        const GlobalTensor<float>& packed)
    {
        LocalTensor<float> rowLocal = rowBuffer_.Get<float>();
        for (uint32_t row = vectorCore_; row < kTotalRows; row += vectorCoreCount_) {
            const uint32_t batch = row / kFrames;
            const uint32_t frame = row % kFrames;
            const uint32_t destination = row * kPackedChannels;
            if (frame >= 2) {
                DataCopy(rowLocal, source[(row - 2) * kChannels], kPackedChannels);
                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);
                DataCopy(packed[destination], rowLocal, kPackedChannels);
                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                continue;
            }

            const uint32_t cacheBase = batch * kChannels * 4 + cacheChannelOffset * 2;
            const uint32_t cachedTaps = 2 - frame;
            for (uint32_t tap = 0; tap < cachedTaps; ++tap) {
                const uint32_t cacheTap = frame + tap;
                for (uint32_t channel = 0; channel < kChannels; ++channel) {
                    rowLocal.SetValue(tap * kChannels + channel,
                                      cacheGm_.GetValue(cacheBase + channel * 2 + cacheTap));
                }
            }
            const uint32_t sourceTaps = frame + 1;
            DataCopy(rowLocal[cachedTaps * kChannels], source[batch * kFrames * kChannels],
                     sourceTaps * kChannels);
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            DataCopy(packed[destination], rowLocal, kPackedChannels);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void NormalizeAndMish()
    {
        LocalTensor<float> row = rowBuffer_.Get<float>();
        LocalTensor<float> temp = workBuffer_.Get<float>();
        LocalTensor<float> scratch = temp[kChannels];
        LocalTensor<float> reduce = temp[kChannels * 2];
        LocalTensor<float> params = parameterBuffer_.Get<float>();
        LocalTensor<float> gamma = params;
        LocalTensor<float> beta = params[kChannels];
        DataCopy(gamma, normWeightGm_, kChannels);
        DataCopy(beta, normBiasGm_, kChannels);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        for (uint32_t rowIndex = vectorCore_; rowIndex < kTotalRows; rowIndex += vectorCoreCount_) {
            DataCopy(row, conv1Gm_[rowIndex * kChannels], kChannels);
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);

            Muls(temp, row, kInvChannels, kChannels);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduce, temp, scratch, kChannels);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            const float mean = reduce.GetValue(0);
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            Adds(row, row, -mean, kChannels);
            PipeBarrier<PIPE_V>();

            Mul(temp, row, row, kChannels);
            PipeBarrier<PIPE_V>();
            Muls(temp, temp, kInvChannels, kChannels);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduce, temp, scratch, kChannels);
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            const float variance = reduce.GetValue(0);
            SetFlag<HardEvent::S_V>(1);
            WaitFlag<HardEvent::S_V>(1);
            Duplicate(temp, variance + kLayerNormEpsilon, kChannels);
            PipeBarrier<PIPE_V>();
            Rsqrt(temp, temp, kChannels);
            PipeBarrier<PIPE_V>();
            Mul(row, row, temp, kChannels);
            PipeBarrier<PIPE_V>();
            Mul(row, row, gamma, kChannels);
            PipeBarrier<PIPE_V>();
            Add(row, row, beta, kChannels);
            PipeBarrier<PIPE_V>();

            // Stable Mish: x * tanh(max(x, 0) + log1p(exp(-abs(x)))).
            Abs(temp, row, kChannels);
            PipeBarrier<PIPE_V>();
            Muls(temp, temp, -1.0f, kChannels);
            PipeBarrier<PIPE_V>();
            Exp(temp, temp, kChannels);
            PipeBarrier<PIPE_V>();
            Adds(temp, temp, 1.0f, kChannels);
            PipeBarrier<PIPE_V>();
            Ln(temp, temp, kChannels);
            PipeBarrier<PIPE_V>();
            Maxs(scratch, row, 0.0f, kChannels);
            PipeBarrier<PIPE_V>();
            Add(temp, temp, scratch, kChannels);
            PipeBarrier<PIPE_V>();
            Tanh(temp, temp, kChannels);
            PipeBarrier<PIPE_V>();
            Mul(row, row, temp, kChannels);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(activatedGm_[rowIndex * kChannels], row, kChannels);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void ResidualAndCache()
    {
        LocalTensor<float> row = rowBuffer_.Get<float>();
        LocalTensor<float> temp = workBuffer_.Get<float>();
        LocalTensor<float> params = parameterBuffer_.Get<float>();
        LocalTensor<float> gate0 = params[kChannels * 2];
        LocalTensor<float> gate1 = params[kChannels * 3];
        DataCopy(gate0, gateGm_, kChannels * 2);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        for (uint32_t rowIndex = vectorCore_; rowIndex < kTotalRows; rowIndex += vectorCoreCount_) {
            DataCopy(row, conv2Gm_[rowIndex * kChannels], kChannels);
            DataCopy(temp, hiddenGm_[rowIndex * kChannels], kChannels);
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);
            const LocalTensor<float> gate = rowIndex < kFrames ? gate0 : gate1;
            Mul(row, row, gate, kChannels);
            PipeBarrier<PIPE_V>();
            Add(row, row, temp, kChannels);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(hiddenOutGm_[rowIndex * kChannels], row, kChannels);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }

        if (vectorCore_ == 0) {
            UpdateCaches();
        }
    }

    __aicore__ inline void UpdateCaches()
    {
        LocalTensor<float> rows = workBuffer_.Get<float>();
        for (uint32_t batch = 0; batch < kBatch; ++batch) {
            const uint32_t rowBase = (batch * kFrames + kFrames - 2) * kChannels;
            const uint32_t cacheBase = batch * kChannels * 4;
            DataCopy(rows, convInputGm_[rowBase], kChannels * 2);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            for (uint32_t channel = 0; channel < kChannels; ++channel) {
                newCacheGm_.SetValue(cacheBase + channel * 2, rows.GetValue(channel));
                newCacheGm_.SetValue(cacheBase + channel * 2 + 1, rows.GetValue(kChannels + channel));
            }
            SetFlag<HardEvent::S_MTE2>(0);
            WaitFlag<HardEvent::S_MTE2>(0);
            DataCopy(rows, activatedGm_[rowBase], kChannels * 2);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            const uint32_t cache2Base = cacheBase + kChannels * 2;
            for (uint32_t channel = 0; channel < kChannels; ++channel) {
                newCacheGm_.SetValue(cache2Base + channel * 2, rows.GetValue(channel));
                newCacheGm_.SetValue(cache2Base + channel * 2 + 1, rows.GetValue(kChannels + channel));
            }
        }
    }

    const MinicpmoCausalConvBlockTilingData* tiling_ = nullptr;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> workBuffer_;
    TBuf<TPosition::VECCALC> parameterBuffer_;
    ConvMatmul mm_;
    GlobalTensor<float> hiddenGm_;
    GlobalTensor<float> convInputGm_;
    GlobalTensor<float> cacheGm_;
    GlobalTensor<float> gateGm_;
    GlobalTensor<float> conv1WeightGm_;
    GlobalTensor<float> conv1BiasGm_;
    GlobalTensor<float> normWeightGm_;
    GlobalTensor<float> normBiasGm_;
    GlobalTensor<float> conv2WeightGm_;
    GlobalTensor<float> conv2BiasGm_;
    GlobalTensor<float> hiddenOutGm_;
    GlobalTensor<float> newCacheGm_;
    GlobalTensor<float> packed1Gm_;
    GlobalTensor<float> conv1Gm_;
    GlobalTensor<float> activatedGm_;
    GlobalTensor<float> packed2Gm_;
    GlobalTensor<float> conv2Gm_;
    uint32_t vectorCore_ = 0;
    uint32_t vectorCoreCount_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void minicpmo_causal_conv_block(
    GM_ADDR hidden,
    GM_ADDR convInput,
    GM_ADDR cnnCache,
    GM_ADDR gateConv,
    GM_ADDR conv1Weight,
    GM_ADDR conv1Bias,
    GM_ADDR normWeight,
    GM_ADDR normBias,
    GM_ADDR conv2Weight,
    GM_ADDR conv2Bias,
    GM_ADDR hiddenOut,
    GM_ADDR newCache,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        MinicpmoCausalConvBlockKernel op;
        op.Init(hidden, convInput, cnnCache, gateConv, conv1Weight, conv1Bias, normWeight, normBias,
                conv2Weight, conv2Bias, hiddenOut, newCache, workspace, &tilingData);
        op.Process();
    }
}
