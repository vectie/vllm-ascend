/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape4MinicpmoCausalConvLinear(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    const gert::Shape* cache = context->GetInputShape(1);
    gert::Shape* projected = context->GetOutputShape(0);
    gert::Shape* newCache = context->GetOutputShape(1);
    if (x == nullptr || cache == nullptr || projected == nullptr || newCache == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *projected = *x;
    *newCache = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MinicpmoCausalConvLinear(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinicpmoCausalConvLinear)
    .InferShape(InferShape4MinicpmoCausalConvLinear)
    .InferDataType(InferDataType4MinicpmoCausalConvLinear);

}  // namespace ops
