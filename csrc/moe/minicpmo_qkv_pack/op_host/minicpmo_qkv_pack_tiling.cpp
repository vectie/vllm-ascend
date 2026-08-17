/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>

#include "minicpmo_qkv_pack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const gert::Shape& input = inputDesc->GetStorageShape();
    OP_CHECK_IF(input.GetDimNum() != 3, OP_LOGE(context, "q must be rank 3"),
                return ge::GRAPH_FAILED);
    const uint32_t batch = static_cast<uint32_t>(input.GetDim(0));
    const uint32_t frames = static_cast<uint32_t>(input.GetDim(1));
    const uint32_t channels = static_cast<uint32_t>(input.GetDim(2));
    OP_CHECK_IF(batch != 2 || frames != 50 || channels != 512,
                OP_LOGE(context, "expected q/k/v [2,50,512]"), return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t usedCoreNum = std::min(16U, platform.GetCoreNumAiv());
    OP_CHECK_IF(usedCoreNum != 16, OP_LOGE(context, "requires 16 vector cores"),
                return ge::GRAPH_FAILED);

    MinicpmoQkvPackTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_frames(frames);
    tiling.set_heads(8);
    tiling.set_headDim(64);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<MinicpmoQkvPackCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalCoreNum = platform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MinicpmoQkvPack)
    .Tiling(TilingFunc)
    .TilingParse<MinicpmoQkvPackCompileInfo>(TilingPrepare);

}  // namespace optiling
