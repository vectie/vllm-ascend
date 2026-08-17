/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class MinicpmoQkvPackKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qOut,
                                GM_ADDR kOut, GM_ADDR vOut,
                                const MinicpmoQkvPackTilingData* tiling)
    {
        batch_ = tiling->batch;
        frames_ = tiling->frames;
        heads_ = tiling->heads;
        headDim_ = tiling->headDim;
        core_ = GetBlockIdx();
        const uint32_t elements = batch_ * frames_ * heads_ * headDim_;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q), elements);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k), elements);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v), elements);
        qOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(qOut), elements);
        kOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(kOut), elements);
        vOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(vOut), elements);
        pipe_.InitBuffer(headBuffer_, headDim_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const uint32_t batch = core_ / heads_;
        const uint32_t head = core_ % heads_;
        CopyTensor(qGm_, qOutGm_, batch, head);
        CopyTensor(kGm_, kOutGm_, batch, head);
        CopyTensor(vGm_, vOutGm_, batch, head);
    }

private:
    __aicore__ inline void CopyTensor(GlobalTensor<T>& input, GlobalTensor<T>& output,
                                      uint32_t batch, uint32_t head)
    {
        LocalTensor<T> local = headBuffer_.Get<T>();
        for (uint32_t frame = 0; frame < frames_; ++frame) {
            const uint32_t source = ((batch * frames_ + frame) * heads_ + head) * headDim_;
            const uint32_t destination = ((batch * heads_ + head) * frames_ + frame) * headDim_;
            DataCopy(local, input[source], headDim_);
            pipe_barrier(PIPE_ALL);
            DataCopy(output[destination], local, headDim_);
            pipe_barrier(PIPE_ALL);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> headBuffer_;
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> qOutGm_;
    GlobalTensor<T> kOutGm_;
    GlobalTensor<T> vOutGm_;
    uint32_t batch_ = 0;
    uint32_t frames_ = 0;
    uint32_t heads_ = 0;
    uint32_t headDim_ = 0;
    uint32_t core_ = 0;
};

extern "C" __global__ __aicore__ void minicpmo_qkv_pack(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR vOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }
    if (TILING_KEY_IS(1)) {
        MinicpmoQkvPackKernel<DTYPE_Q> op;
        op.Init(q, k, v, qOut, kOut, vOut, &tilingData);
        op.Process();
    }
}
