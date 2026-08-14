/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape4MinicpmoCausalConvPack(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    const gert::Shape* cache = context->GetInputShape(1);
    if (x == nullptr || cache == nullptr || x->GetDimNum() != 3 || cache->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* packed = context->GetOutputShape(0);
    gert::Shape* newCache = context->GetOutputShape(1);
    if (packed == nullptr || newCache == nullptr) {
        return ge::GRAPH_FAILED;
    }
    packed->SetDimNum(2);
    packed->SetDim(0, x->GetDim(0) * x->GetDim(1));
    packed->SetDim(1, x->GetDim(2) * 3);
    *newCache = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MinicpmoCausalConvPack(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinicpmoCausalConvPack)
    .InferShape(InferShape4MinicpmoCausalConvPack)
    .InferDataType(InferDataType4MinicpmoCausalConvPack);

}  // namespace ops
