/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernel_operator.h"
#include "basic_api/kernel_vec_intf.h"

using namespace AscendC;

template <typename T>
class MinicpmoCausalConvPackKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cache, GM_ADDR packed, GM_ADDR newCache,
                                const MinicpmoCausalConvPackTilingData* tiling)
    {
        batch_ = tiling->batch;
        frames_ = tiling->frames;
        channels_ = tiling->channels;
        rowsPerCore_ = tiling->rowsPerCore;
        totalRows_ = batch_ * frames_;
        cacheMajor_ = tiling->cacheMajor != 0;
        core_ = GetBlockIdx();
        startRow_ = core_ * rowsPerCore_;
        endRow_ = min(startRow_ + rowsPerCore_, totalRows_);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalRows_ * channels_);
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cache), batch_ * channels_ * 2);
        packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(packed), totalRows_ * channels_ * 3);
        newCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(newCache), batch_ * channels_ * 2);
        pipe_.InitBuffer(frameBuffer_, channels_ * sizeof(T));
        pipe_.InitBuffer(cacheBuffer_, channels_ * 2 * sizeof(T));
        pipe_.InitBuffer(cacheLayoutBuffer_, channels_ * 2 * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = startRow_; row < endRow_; ++row) {
            const uint32_t batch = row / frames_;
            const int32_t frame = static_cast<int32_t>(row % frames_);
            if (!cacheMajor_ && frame < 2) {
                CopyChannelMajorPrefix(batch, frame, row);
                continue;
            }
            for (int32_t tap = 0; tap < 3; ++tap) {
                CopyTap(batch, frame + tap - 2, row, static_cast<uint32_t>(tap));
            }
        }
        if (core_ == 0) {
            UpdateCache();
        }
    }

private:
    __aicore__ inline void CopyLocalTap(const LocalTensor<T>& local, uint32_t row, uint32_t tap)
    {
        const uint32_t destination = row * channels_ * 3 + tap * channels_;
        DataCopy(packedGm_[destination], local, channels_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyChannelMajorPrefix(uint32_t batch, int32_t frame, uint32_t row)
    {
        LocalTensor<T> cache = cacheBuffer_.Get<T>();
        LocalTensor<T> frames = cacheLayoutBuffer_.Get<T>();
        const uint32_t cacheBase = batch * channels_ * 2;
        DataCopy(cache, cacheGm_[cacheBase], channels_ * 2);
        pipe_barrier(PIPE_ALL);
        DeInterleave(frames, frames[channels_], cache, channels_ * 2);
        pipe_barrier(PIPE_ALL);

        if (frame == 0) {
            CopyLocalTap(frames, row, 0);
            CopyLocalTap(frames[channels_], row, 1);
            CopyTap(batch, 0, row, 2);
        } else {
            CopyLocalTap(frames[channels_], row, 0);
            CopyTap(batch, 0, row, 1);
            CopyTap(batch, 1, row, 2);
        }
    }

    __aicore__ inline void CopyTap(uint32_t batch, int32_t sourceFrame, uint32_t row, uint32_t tap)
    {
        LocalTensor<T> local = frameBuffer_.Get<T>();
        if (sourceFrame >= 0) {
            const uint32_t source = (batch * frames_ + static_cast<uint32_t>(sourceFrame)) * channels_;
            DataCopy(local, xGm_[source], channels_);
        } else if (cacheMajor_) {
            const uint32_t cacheTap = static_cast<uint32_t>(sourceFrame + 2);
            const uint32_t source = batch * channels_ * 2 + cacheTap * channels_;
            DataCopy(local, cacheGm_[source], channels_);
        } else {
            const uint32_t cacheTap = static_cast<uint32_t>(sourceFrame + 2);
            const uint32_t cacheBase = batch * channels_ * 2;
            for (uint32_t channel = 0; channel < channels_; ++channel) {
                local.SetValue(channel, cacheGm_.GetValue(cacheBase + channel * 2 + cacheTap));
            }
        }
        pipe_barrier(PIPE_ALL);
        const uint32_t destination = row * channels_ * 3 + tap * channels_;
        DataCopy(packedGm_[destination], local, channels_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void UpdateCache()
    {
        LocalTensor<T> local = cacheBuffer_.Get<T>();
        LocalTensor<T> packed = cacheLayoutBuffer_.Get<T>();
        for (uint32_t batch = 0; batch < batch_; ++batch) {
            const uint32_t xBase = (batch * frames_ + frames_ - 2) * channels_;
            DataCopy(local, xGm_[xBase], channels_ * 2);
            pipe_barrier(PIPE_ALL);
            const uint32_t cacheBase = batch * channels_ * 2;
            if (cacheMajor_) {
                DataCopy(newCacheGm_[cacheBase], local, channels_ * 2);
                pipe_barrier(PIPE_ALL);
            } else {
                Interleave(packed, packed[channels_], local, local[channels_], channels_);
                pipe_barrier(PIPE_ALL);
                DataCopy(newCacheGm_[cacheBase], packed, channels_ * 2);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> frameBuffer_;
    TBuf<TPosition::VECCALC> cacheBuffer_;
    TBuf<TPosition::VECCALC> cacheLayoutBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> cacheGm_;
    GlobalTensor<T> packedGm_;
    GlobalTensor<T> newCacheGm_;
    uint32_t batch_ = 0;
    uint32_t frames_ = 0;
    uint32_t channels_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t totalRows_ = 0;
    uint32_t core_ = 0;
    uint32_t startRow_ = 0;
    uint32_t endRow_ = 0;
    bool cacheMajor_ = false;
};

extern "C" __global__ __aicore__ void minicpmo_causal_conv_pack(
    GM_ADDR x, GM_ADDR cache, GM_ADDR packed, GM_ADDR newCache, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }
    if (TILING_KEY_IS(1)) {
        MinicpmoCausalConvPackKernel<DTYPE_X> op;
        op.Init(x, cache, packed, newCache, &tilingData);
        op.Process();
    }
}
