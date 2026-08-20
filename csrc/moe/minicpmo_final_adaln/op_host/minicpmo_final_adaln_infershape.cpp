/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape4MinicpmoFinalAdaln(gert::InferShapeContext* context)
{
    const gert::Shape* hidden = context->GetInputShape(0);
    const gert::Shape* weight = context->GetInputShape(2);
    gert::Shape* projected = context->GetOutputShape(0);
    if (hidden == nullptr || weight == nullptr || projected == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *projected = *hidden;
    projected->SetDim(2, weight->GetDim(0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MinicpmoFinalAdaln(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinicpmoFinalAdaln)
    .InferShape(InferShape4MinicpmoFinalAdaln)
    .InferDataType(InferDataType4MinicpmoFinalAdaln);

}  // namespace ops
