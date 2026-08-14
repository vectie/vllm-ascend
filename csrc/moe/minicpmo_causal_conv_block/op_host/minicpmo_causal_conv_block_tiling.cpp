/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <initializer_list>

#include "minicpmo_causal_conv_block_tiling.h"
#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {

constexpr uint32_t kBatch = 2;
constexpr uint32_t kFrames = 50;
constexpr uint32_t kChannels = 512;
constexpr uint32_t kKernel = 3;
constexpr uint32_t kPreferredAicCores = 16;
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
    const auto* convInputDesc = context->GetInputShape(1);
    const auto* cacheDesc = context->GetInputShape(2);
    const auto* gateDesc = context->GetInputShape(3);
    const auto* conv1WeightDesc = context->GetInputShape(4);
    const auto* vectorDesc = context->GetInputShape(5);
    const auto* conv2WeightDesc = context->GetInputShape(8);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, convInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, conv1WeightDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vectorDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, conv2WeightDesc);

    const gert::Shape& hidden = hiddenDesc->GetStorageShape();
    const gert::Shape& convInput = convInputDesc->GetStorageShape();
    const gert::Shape& cache = cacheDesc->GetStorageShape();
    const gert::Shape& gate = gateDesc->GetStorageShape();
    const gert::Shape& conv1Weight = conv1WeightDesc->GetStorageShape();
    const gert::Shape& vector = vectorDesc->GetStorageShape();
    const gert::Shape& conv2Weight = conv2WeightDesc->GetStorageShape();
    OP_CHECK_IF(!ShapeEquals(hidden, {kBatch, kFrames, kChannels}) ||
                    !ShapeEquals(convInput, {kBatch, kFrames, kChannels}),
                OP_LOGE(context, "hidden and conv_input must be [2,50,512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(cache, {kBatch, kChannels * 2, 2}),
                OP_LOGE(context, "cnn_cache must be [2,1024,2]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(gate, {kBatch, 1, kChannels}),
                OP_LOGE(context, "gate_conv must be [2,1,512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(conv1Weight, {kChannels, kChannels * kKernel}) ||
                    !ShapeEquals(conv2Weight, {kChannels, kChannels * kKernel}),
                OP_LOGE(context, "flattened Conv weights must be [512,1536]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ShapeEquals(vector, {kChannels}),
                OP_LOGE(context, "bias and normalization vectors must be [512]"), return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableAic = platform.GetCoreNumAic();
    OP_CHECK_IF(availableAic == 0, OP_LOGE(context, "no Cube cores are available"), return ge::GRAPH_FAILED);
    uint32_t usedCoreNum = std::min(availableAic, kPreferredAicCores);
    while (usedCoreNum > 1 && kChannels % usedCoreNum != 0) {
        --usedCoreNum;
    }
    const uint32_t nPerCore = kChannels / usedCoreNum;

    MinicpmoCausalConvBlockTilingData tiling;
    constexpr uint64_t totalRows = kBatch * kFrames;
    constexpr uint64_t packedElements = totalRows * kChannels * kKernel;
    constexpr uint64_t matrixElements = totalRows * kChannels;
    uint64_t offset = 0;
    tiling.set_packed1Offset(offset);
    offset += packedElements;
    tiling.set_conv1Offset(offset);
    offset += matrixElements;
    tiling.set_activatedOffset(offset);
    offset += matrixElements;
    tiling.set_packed2Offset(offset);
    offset += packedElements;
    tiling.set_conv2Offset(offset);
    offset += matrixElements;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_nPerCore(nPerCore);
    tiling.set_totalRows(totalRows);

    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    mm.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, true);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    mm.SetBias(true);
    mm.SetShape(totalRows, nPerCore, kChannels * kKernel);
    mm.SetOrgShape(totalRows, kChannels, kChannels * kKernel);
    mm.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(mm.GetTiling(tiling.mmTiling) == -1,
                OP_LOGE(context, "failed to generate FP32 Matmul tiling"), return ge::GRAPH_FAILED);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = kSystemWorkspace + offset * sizeof(float);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<MinicpmoCausalConvBlockCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalAicCoreNum = platform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MinicpmoCausalConvBlock)
    .Tiling(TilingFunc)
    .TilingParse<MinicpmoCausalConvBlockCompileInfo>(TilingPrepare);

}  // namespace optiling
