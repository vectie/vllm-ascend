/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <initializer_list>

#include "minicpmo_final_adaln_tiling.h"
#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {

constexpr uint32_t kBatch = 2;
constexpr uint32_t kFrames = 50;
constexpr uint32_t kChannels = 512;
constexpr uint32_t kOutputChannels = 80;
constexpr uint32_t kPreferredAicCores = 5;
constexpr uint64_t kSystemWorkspace = 16UL * 1024UL * 1024UL;

bool ShapeEquals(const gert::Shape& shape, std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : expected) {
        if (shape.GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* hiddenDesc = context->GetInputShape(0);
    const auto* modulationDesc = context->GetInputShape(1);
    const auto* weightDesc = context->GetInputShape(2);
    const auto* biasDesc = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, modulationDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);

    OP_CHECK_IF(!ShapeEquals(hiddenDesc->GetStorageShape(), {kBatch, kFrames, kChannels}),
                OP_LOGE(context, "hidden must be [2,50,512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(modulationDesc->GetStorageShape(), {kBatch, 1, kChannels * 2}),
                OP_LOGE(context, "modulation must be [2,1,1024]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(weightDesc->GetStorageShape(), {kOutputChannels, kChannels}),
                OP_LOGE(context, "weight must be [80,512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(biasDesc->GetStorageShape(), {kOutputChannels}),
                OP_LOGE(context, "bias must be [80]"), return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableAic = platform.GetCoreNumAic();
    OP_CHECK_IF(availableAic == 0, OP_LOGE(context, "no Cube cores are available"), return ge::GRAPH_FAILED);
    uint32_t usedCoreNum = std::min(availableAic, kPreferredAicCores);
    while (usedCoreNum > 1 && kOutputChannels % usedCoreNum != 0) {
        --usedCoreNum;
    }
    const uint32_t nPerCore = kOutputChannels / usedCoreNum;

    MinicpmoFinalAdalnTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_nPerCore(nPerCore);
    tiling.set_modulatedOffset(0);

    constexpr uint64_t totalRows = kBatch * kFrames;
    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    mm.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, true);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    mm.SetBias(true);
    mm.SetShape(totalRows, nPerCore, kChannels);
    mm.SetOrgShape(totalRows, kOutputChannels, kChannels);
    mm.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(mm.GetTiling(tiling.mmTiling) == -1,
                OP_LOGE(context, "failed to generate FP32 Matmul tiling"), return ge::GRAPH_FAILED);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = kSystemWorkspace + totalRows * kChannels * sizeof(float);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<MinicpmoFinalAdalnCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalAicCoreNum = platform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MinicpmoFinalAdaln)
    .Tiling(TilingFunc)
    .TilingParse<MinicpmoFinalAdalnCompileInfo>(TilingPrepare);

}  // namespace optiling
