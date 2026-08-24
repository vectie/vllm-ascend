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
        startRow_ = min(core_ * rowsPerCore_, totalRows_);
        endRow_ = min(startRow_ + rowsPerCore_, totalRows_);
        if (!cacheMajor_) {
            for (uint32_t row = startRow_; row < endRow_; ++row) {
                if (row % frames_ < 2) {
                    needsChannelMajorWorkspace_ = true;
                    break;
                }
            }
            // Core 0 also transposes and publishes the rolling cache.
            needsChannelMajorWorkspace_ =
                needsChannelMajorWorkspace_ || core_ == 0;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalRows_ * channels_);
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cache), batch_ * channels_ * 2);
        packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(packed), totalRows_ * channels_ * 3);
        newCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(newCache), batch_ * channels_ * 2);
        // Keep two complete three-tap rows in UB so MTE2 can fill row n+1
        // while MTE3 drains row n. Only the first two channel-major rows need
        // Gather; all remaining rows are contiguous regardless of cache layout.
        pipe_.InitBuffer(packedRowBuffers_, 2 * 3 * channels_ * sizeof(T));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            loadToStoreEvents_[slot] =
                GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
            storeToLoadEvents_[slot] =
                GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        }
        if (needsChannelMajorWorkspace_) {
            pipe_.InitBuffer(frameBuffer_, channels_ * sizeof(T));
            pipe_.InitBuffer(cacheBuffer_, channels_ * 2 * sizeof(T));
            pipe_.InitBuffer(cacheLayoutBuffer_, channels_ * 2 * sizeof(T));
            pipe_.InitBuffer(indexBuffer_, channels_ * 2 * sizeof(uint32_t));
        }
    }

    __aicore__ inline void Process()
    {
        if (cacheMajor_) {
            ProcessCacheMajor();
            ReleasePipelineEvents();
            return;
        }

        ProcessChannelMajor();
        if (core_ == 0) {
            UpdateCache();
        }
        ReleasePipelineEvents();
    }

private:
    __aicore__ inline void PrepareChannelMajorOffsets()
    {
        if (prefixOffsetsReady_) {
            return;
        }
        LocalTensor<int32_t> offsets = indexBuffer_.Get<int32_t>();
        CreateVecIndex(offsets, 0, channels_);
        pipe_barrier(PIPE_V);
        Muls(offsets, offsets, static_cast<int32_t>(2 * sizeof(T)), channels_);
        pipe_barrier(PIPE_V);
        prefixOffsetsReady_ = true;
    }

    __aicore__ inline void ProcessChannelMajor()
    {
        uint32_t row = startRow_;
        while (row < endRow_) {
            const uint32_t batch = row / frames_;
            const uint32_t frame = row % frames_;
            if (frame < 2) {
                CopyChannelMajorPrefix(
                    batch,
                    static_cast<int32_t>(frame),
                    row);
                ++row;
                continue;
            }

            // Stop at the batch boundary; the next batch's first two rows
            // need cache Gather before contiguous input rows resume.
            const uint32_t batchEnd = (batch + 1) * frames_;
            const uint32_t segmentEnd = min(endRow_, batchEnd);
            ProcessContiguousRows(row, segmentEnd);
            row = segmentEnd;
        }
    }

    __aicore__ inline void ProcessContiguousRows(
        uint32_t beginRow,
        uint32_t endRow)
    {
        const uint32_t rowElements = 3 * channels_;
        const uint32_t rowCount = endRow - beginRow;
        LocalTensor<T> buffers = packedRowBuffers_.Get<T>();
        for (uint32_t ordinal = 0; ordinal < rowCount; ++ordinal) {
            const uint32_t slot = ordinal & 1U;
            if (ordinal >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[slot]);
            }
            const uint32_t row = beginRow + ordinal;
            const uint32_t batch = row / frames_;
            const uint32_t frame = row % frames_;
            const uint32_t source =
                (batch * frames_ + frame - 2) * channels_;
            LocalTensor<T> local = buffers[slot * rowElements];
            DataCopy(local, xGm_[source], rowElements);
            SetFlag<HardEvent::MTE2_MTE3>(loadToStoreEvents_[slot]);
            WaitFlag<HardEvent::MTE2_MTE3>(loadToStoreEvents_[slot]);
            DataCopy(packedGm_[row * rowElements], local, rowElements);
            SetFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[slot]);
        }
        const uint32_t liveSlots = min(rowCount, static_cast<uint32_t>(2));
        for (uint32_t slot = 0; slot < liveSlots; ++slot) {
            WaitFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[slot]);
        }
    }

    __aicore__ inline void ProcessCacheMajor()
    {
        const uint32_t rowElements = 3 * channels_;
        const uint32_t localRowCount = endRow_ - startRow_;
        LocalTensor<T> buffers = packedRowBuffers_.Get<T>();
        for (uint32_t ordinal = 0; ordinal < localRowCount; ++ordinal) {
            const uint32_t row = startRow_ + ordinal;
            const uint32_t slot = ordinal & 1U;
            if (ordinal >= 2) {
                // Reuse only after the previous MTE3 transfer from this UB
                // slot has completed.  The other slot remains independent.
                WaitFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[slot]);
            }

            LocalTensor<T> local = buffers[slot * rowElements];
            const uint32_t batch = row / frames_;
            const uint32_t frame = row % frames_;
            const uint32_t xBatchBase = batch * frames_ * channels_;
            const uint32_t cacheBase = batch * channels_ * 2;
            if (frame == 0) {
                // [cache[-2], cache[-1], x[0]]
                DataCopy(local, cacheGm_[cacheBase], 2 * channels_);
                DataCopy(local[2 * channels_], xGm_[xBatchBase], channels_);
            } else if (frame == 1) {
                // [cache[-1], x[0], x[1]]
                DataCopy(local, cacheGm_[cacheBase + channels_], channels_);
                DataCopy(local[channels_], xGm_[xBatchBase], 2 * channels_);
            } else {
                // All three taps are contiguous in the canonical BTD input.
                // Move the complete row with one DMA instead of materializing
                // each tap through the same single UB frame buffer.
                const uint32_t source =
                    xBatchBase + (frame - 2) * channels_;
                DataCopy(local, xGm_[source], rowElements);
            }

            SetFlag<HardEvent::MTE2_MTE3>(loadToStoreEvents_[slot]);
            WaitFlag<HardEvent::MTE2_MTE3>(loadToStoreEvents_[slot]);
            DataCopy(packedGm_[row * rowElements], local, rowElements);
            if (frame >= frames_ - 2) {
                // The newest frame is already resident at tap 2. Publish the
                // two rolling-cache frames from their owning row cores instead
                // of rereading both frames from GM on core 0 after packing.
                const uint32_t cacheFrame = frame - (frames_ - 2);
                DataCopy(
                    newCacheGm_[cacheBase + cacheFrame * channels_],
                    local[2 * channels_],
                    channels_);
            }
            SetFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[slot]);
        }

        // Complete the final outstanding write in each live slot before event
        // IDs are released.
        const uint32_t liveSlots = min(localRowCount, static_cast<uint32_t>(2));
        for (uint32_t slot = 0; slot < liveSlots; ++slot) {
            WaitFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[slot]);
        }
    }

    __aicore__ inline void ReleasePipelineEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(
                loadToStoreEvents_[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(
                storeToLoadEvents_[slot]);
        }
    }

    __aicore__ inline void CopyLocalTap(const LocalTensor<T>& local, uint32_t row, uint32_t tap)
    {
        const uint32_t destination = row * channels_ * 3 + tap * channels_;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(packedGm_[destination], local, channels_);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void CopyChannelMajorPrefix(uint32_t batch, int32_t frame, uint32_t row)
    {
        PrepareChannelMajorOffsets();
        LocalTensor<T> cache = cacheBuffer_.Get<T>();
        LocalTensor<T> local = frameBuffer_.Get<T>();
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        const uint32_t cacheBase = batch * channels_ * 2;
        DataCopy(cache, cacheGm_[cacheBase], channels_ * 2);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        if (frame == 0) {
            Gather(local, cache, offsets, static_cast<uint32_t>(0), channels_);
            CopyLocalTap(local, row, 0);
            Gather(local, cache, offsets, static_cast<uint32_t>(sizeof(T)), channels_);
            CopyLocalTap(local, row, 1);
            CopyTap(batch, 0, row, 2);
        } else {
            Gather(local, cache, offsets, static_cast<uint32_t>(sizeof(T)), channels_);
            CopyLocalTap(local, row, 0);
            CopyTap(batch, 0, row, 1);
            CopyTap(batch, 1, row, 2);
        }
    }

    __aicore__ inline void CopyTap(uint32_t batch, int32_t sourceFrame, uint32_t row, uint32_t tap)
    {
        LocalTensor<T> local = frameBuffer_.Get<T>();
        // Prefix rows consume cache through CopyChannelMajorPrefix; all taps
        // reaching this helper are contiguous input frames.
        const uint32_t source =
            (batch * frames_ + static_cast<uint32_t>(sourceFrame)) * channels_;
        DataCopy(local, xGm_[source], channels_);
        SetFlag<HardEvent::MTE2_MTE3>(loadToStoreEvents_[0]);
        WaitFlag<HardEvent::MTE2_MTE3>(loadToStoreEvents_[0]);
        const uint32_t destination = row * channels_ * 3 + tap * channels_;
        DataCopy(packedGm_[destination], local, channels_);
        SetFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(storeToLoadEvents_[0]);
    }

    __aicore__ inline void UpdateCache()
    {
        LocalTensor<T> local = cacheBuffer_.Get<T>();
        LocalTensor<T> packed = cacheLayoutBuffer_.Get<T>();
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        // The transpose map is batch invariant. Build it once on the only
        // core that publishes newCache instead of repeating 1024 scalar UB
        // writes for every batch item.
        for (uint32_t output = 0; output < channels_ * 2; ++output) {
            const uint32_t frame = output & 1;
            const uint32_t channel = output >> 1;
            offsets.SetValue(output, (frame * channels_ + channel) * sizeof(T));
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        for (uint32_t batch = 0; batch < batch_; ++batch) {
            const uint32_t xBase = (batch * frames_ + frames_ - 2) * channels_;
            DataCopy(local, xGm_[xBase], channels_ * 2);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            const uint32_t cacheBase = batch * channels_ * 2;
            Gather(
                packed,
                local,
                offsets,
                static_cast<uint32_t>(0),
                channels_ * 2);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(newCacheGm_[cacheBase], packed, channels_ * 2);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> frameBuffer_;
    TBuf<TPosition::VECCALC> cacheBuffer_;
    TBuf<TPosition::VECCALC> cacheLayoutBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> packedRowBuffers_;
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
    bool needsChannelMajorWorkspace_ = false;
    bool prefixOffsetsReady_ = false;
    TEventID loadToStoreEvents_[2] = {0, 0};
    TEventID storeToLoadEvents_[2] = {0, 0};
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
