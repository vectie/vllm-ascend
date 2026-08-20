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
constexpr uint32_t kOutputChannels = 80;
constexpr uint32_t kTotalRows = kBatch * kFrames;
constexpr float kInvChannels = 1.0f / static_cast<float>(kChannels);
constexpr float kLayerNormEpsilon = 1.0e-6f;
constexpr MatmulConfig kMatmulConfig{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                     0, 0, 0, 0, true};

using AType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using FinalMatmul = matmul::MatmulImpl<AType, BType, CType, BiasType, kMatmulConfig>;

class MinicpmoFinalAdalnKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR hidden,
        GM_ADDR modulation,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR projected,
        GM_ADDR workspace,
        const MinicpmoFinalAdalnTilingData* tiling)
    {
        tiling_ = tiling;
        hiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hidden), kTotalRows * kChannels);
        modulationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(modulation), kBatch * kChannels * 2);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), kOutputChannels * kChannels);
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), kOutputChannels);
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected), kTotalRows * kOutputChannels);

        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        __gm__ float* work = reinterpret_cast<__gm__ float*>(userWorkspace);
        modulatedGm_.SetGlobalBuffer(work + tiling->modulatedOffset, kTotalRows * kChannels);

        if ASCEND_IS_AIC {
            mm_.Init(&tiling->mmTiling, &pipe_);
        } else {
            vectorCore_ = GetBlockIdx();
            vectorCoreCount_ = tiling->usedCoreNum * 2;
            pipe_.InitBuffer(rowBuffer_, kChannels * sizeof(float));
            pipe_.InitBuffer(workBuffer_, kChannels * 3 * sizeof(float));
            pipe_.InitBuffer(parameterBuffer_, kBatch * kChannels * 2 * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            NormalizeAndModulate();
        }
        SyncAll<false>();

        if ASCEND_IS_AIC {
            const uint32_t core = GetBlockIdx();
            const uint32_t nStart = core * tiling_->nPerCore;
            mm_.SetOrgShape(kTotalRows, kOutputChannels, kChannels);
            mm_.SetSingleShape(kTotalRows, tiling_->nPerCore, kChannels);
            mm_.SetTensorA(modulatedGm_, false);
            mm_.SetTensorB(weightGm_[nStart * kChannels], true);
            mm_.SetBias(biasGm_[nStart]);
            mm_.template IterateAll<false>(projectedGm_[nStart], 0);
        }
    }

private:
    __aicore__ inline void NormalizeAndModulate()
    {
        LocalTensor<float> row = rowBuffer_.Get<float>();
        LocalTensor<float> work = workBuffer_.Get<float>();
        LocalTensor<float> scratch = work[kChannels];
        LocalTensor<float> reduce = work[kChannels * 2];
        LocalTensor<float> parameters = parameterBuffer_.Get<float>();
        DataCopy(parameters, modulationGm_, kBatch * kChannels * 2);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        for (uint32_t rowIndex = vectorCore_; rowIndex < kTotalRows; rowIndex += vectorCoreCount_) {
            DataCopy(row, hiddenGm_[rowIndex * kChannels], kChannels);
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);

            Muls(work, row, kInvChannels, kChannels);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduce, work, scratch, kChannels);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            const float mean = reduce.GetValue(0);
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            Adds(row, row, -mean, kChannels);
            PipeBarrier<PIPE_V>();

            Mul(work, row, row, kChannels);
            PipeBarrier<PIPE_V>();
            Muls(work, work, kInvChannels, kChannels);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduce, work, scratch, kChannels);
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            const float variance = reduce.GetValue(0);
            SetFlag<HardEvent::S_V>(1);
            WaitFlag<HardEvent::S_V>(1);
            Duplicate(work, variance + kLayerNormEpsilon, kChannels);
            PipeBarrier<PIPE_V>();
            Rsqrt(work, work, kChannels);
            PipeBarrier<PIPE_V>();
            Mul(row, row, work, kChannels);
            PipeBarrier<PIPE_V>();

            const uint32_t batch = rowIndex / kFrames;
            LocalTensor<float> shift = parameters[batch * kChannels * 2];
            LocalTensor<float> scale = parameters[batch * kChannels * 2 + kChannels];
            Mul(work, row, scale, kChannels);
            PipeBarrier<PIPE_V>();
            Add(row, row, work, kChannels);
            PipeBarrier<PIPE_V>();
            Add(row, row, shift, kChannels);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(modulatedGm_[rowIndex * kChannels], row, kChannels);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    const MinicpmoFinalAdalnTilingData* tiling_ = nullptr;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> workBuffer_;
    TBuf<TPosition::VECCALC> parameterBuffer_;
    FinalMatmul mm_;
    GlobalTensor<float> hiddenGm_;
    GlobalTensor<float> modulationGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<float> projectedGm_;
    GlobalTensor<float> modulatedGm_;
    uint32_t vectorCore_ = 0;
    uint32_t vectorCoreCount_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void minicpmo_final_adaln(
    GM_ADDR hidden,
    GM_ADDR modulation,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR projected,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        MinicpmoFinalAdalnKernel op;
        op.Init(hidden, modulation, weight, bias, projected, workspace, &tilingData);
        op.Process();
    }
}
