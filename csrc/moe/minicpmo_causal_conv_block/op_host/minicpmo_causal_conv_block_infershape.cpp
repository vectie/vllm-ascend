/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape4MinicpmoCausalConvBlock(gert::InferShapeContext* context)
{
    const gert::Shape* hidden = context->GetInputShape(0);
    const gert::Shape* cache = context->GetInputShape(2);
    gert::Shape* hiddenOut = context->GetOutputShape(0);
    gert::Shape* newCache = context->GetOutputShape(1);
    if (hidden == nullptr || cache == nullptr || hiddenOut == nullptr || newCache == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *hiddenOut = *hidden;
    *newCache = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MinicpmoCausalConvBlock(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinicpmoCausalConvBlock)
    .InferShape(InferShape4MinicpmoCausalConvBlock)
    .InferDataType(InferDataType4MinicpmoCausalConvBlock);

}  // namespace ops
