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
constexpr MatmulConfig kMatmulConfig{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                     0, 0, 0, 0, true};

using AType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using ConvMatmul = matmul::MatmulImpl<AType, BType, CType, BiasType, kMatmulConfig>;

class MinicpmoCausalConvLinearKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR cache,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR projected,
        GM_ADDR newCache,
        GM_ADDR workspace,
        const MinicpmoCausalConvLinearTilingData* tiling)
    {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), kTotalRows * kChannels);
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cache), kBatch * kChannels * 2);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), kChannels * kPackedChannels);
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), kChannels);
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected), kTotalRows * kChannels);
        newCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newCache), kBatch * kChannels * 2);

        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        __gm__ float* work = reinterpret_cast<__gm__ float*>(userWorkspace);
        packedGm_.SetGlobalBuffer(work + tiling->packedOffset, kTotalRows * kPackedChannels);

        if ASCEND_IS_AIC {
            mm_.Init(&tiling->mmTiling, &pipe_);
        } else {
            vectorCore_ = GetBlockIdx();
            vectorCoreCount_ = tiling->usedCoreNum * 2;
            pipe_.InitBuffer(rowBuffer_, kPackedChannels * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            Pack();
            if (vectorCore_ == 0) {
                UpdateCache();
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            Matmul();
        }
    }

private:
    __aicore__ inline void Matmul()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t nStart = core * tiling_->nPerCore;
        mm_.SetOrgShape(kTotalRows, kChannels, kPackedChannels);
        mm_.SetSingleShape(kTotalRows, tiling_->nPerCore, kPackedChannels);
        mm_.SetTensorA(packedGm_, false);
        mm_.SetTensorB(weightGm_[nStart * kPackedChannels], true);
        mm_.SetBias(biasGm_[nStart]);
        mm_.template IterateAll<false>(projectedGm_[nStart], 0);
    }

    __aicore__ inline void Pack()
    {
        LocalTensor<float> rowLocal = rowBuffer_.Get<float>();
        for (uint32_t row = vectorCore_; row < kTotalRows; row += vectorCoreCount_) {
            const uint32_t batch = row / kFrames;
            const uint32_t frame = row % kFrames;
            const uint32_t destination = row * kPackedChannels;
            if (frame >= 2) {
                DataCopy(rowLocal, xGm_[(row - 2) * kChannels], kPackedChannels);
                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);
                DataCopy(packedGm_[destination], rowLocal, kPackedChannels);
                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                continue;
            }

            const uint32_t cacheBase = batch * kChannels * 2;
            const uint32_t cachedTaps = 2 - frame;
            for (uint32_t tap = 0; tap < cachedTaps; ++tap) {
                const uint32_t cacheTap = frame + tap;
                for (uint32_t channel = 0; channel < kChannels; ++channel) {
                    rowLocal.SetValue(tap * kChannels + channel,
                                      cacheGm_.GetValue(cacheBase + channel * 2 + cacheTap));
                }
            }
            const uint32_t sourceTaps = frame + 1;
            DataCopy(rowLocal[cachedTaps * kChannels], xGm_[batch * kFrames * kChannels],
                     sourceTaps * kChannels);
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            DataCopy(packedGm_[destination], rowLocal, kPackedChannels);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void UpdateCache()
    {
        LocalTensor<float> rows = rowBuffer_.Get<float>();
        for (uint32_t batch = 0; batch < kBatch; ++batch) {
            const uint32_t rowBase = (batch * kFrames + kFrames - 2) * kChannels;
            const uint32_t cacheBase = batch * kChannels * 2;
            DataCopy(rows, xGm_[rowBase], kChannels * 2);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            for (uint32_t channel = 0; channel < kChannels; ++channel) {
                newCacheGm_.SetValue(cacheBase + channel * 2, rows.GetValue(channel));
                newCacheGm_.SetValue(cacheBase + channel * 2 + 1, rows.GetValue(kChannels + channel));
            }
        }
    }

    const MinicpmoCausalConvLinearTilingData* tiling_ = nullptr;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    ConvMatmul mm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> cacheGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<float> projectedGm_;
    GlobalTensor<float> newCacheGm_;
    GlobalTensor<float> packedGm_;
    uint32_t vectorCore_ = 0;
    uint32_t vectorCoreCount_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void minicpmo_causal_conv_linear(
    GM_ADDR x,
    GM_ADDR cache,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR projected,
    GM_ADDR newCache,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        MinicpmoCausalConvLinearKernel op;
        op.Init(x, cache, weight, bias, projected, newCache, workspace, &tilingData);
        op.Process();
    }
}
