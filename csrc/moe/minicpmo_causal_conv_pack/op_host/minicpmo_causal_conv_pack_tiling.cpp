/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>

#include "minicpmo_causal_conv_pack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* xDesc = context->GetInputShape(0);
    const auto* cacheDesc = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheDesc);
    const gert::Shape& x = xDesc->GetStorageShape();
    const gert::Shape& cache = cacheDesc->GetStorageShape();
    OP_CHECK_IF(x.GetDimNum() != 3 || cache.GetDimNum() != 3,
                OP_LOGE(context, "x and cache must be rank 3"), return ge::GRAPH_FAILED);

    const uint32_t batch = static_cast<uint32_t>(x.GetDim(0));
    const uint32_t frames = static_cast<uint32_t>(x.GetDim(1));
    const uint32_t channels = static_cast<uint32_t>(x.GetDim(2));
    OP_CHECK_IF(batch != 2 || frames != 50 || channels != 512,
                OP_LOGE(context, "expected x [2,50,512], got [%u,%u,%u]", batch, frames, channels),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDim(0) != batch || cache.GetDim(1) != channels || cache.GetDim(2) != 2,
                OP_LOGE(context, "expected cache [2,512,2]"), return ge::GRAPH_FAILED);

    const uint32_t totalRows = batch * frames;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCoreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(availableCoreNum == 0,
                OP_LOGE(context, "no vector cores are available"), return ge::GRAPH_FAILED);
    const uint32_t usedCoreNum = std::min(totalRows, availableCoreNum);
    const uint32_t rowsPerCore = (totalRows + usedCoreNum - 1) / usedCoreNum;

    MinicpmoCausalConvPackTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_frames(frames);
    tiling.set_channels(channels);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<MinicpmoCausalConvPackCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalCoreNum = platform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MinicpmoCausalConvPack)
    .Tiling(TilingFunc)
    .TilingParse<MinicpmoCausalConvPackCompileInfo>(TilingPrepare);

}  // namespace optiling
