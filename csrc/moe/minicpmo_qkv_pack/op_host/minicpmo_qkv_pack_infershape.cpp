/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape4MinicpmoQkvPack(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(0);
    if (input == nullptr || input->GetDimNum() != 3 || input->GetDim(2) != 512) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 3; ++index) {
        gert::Shape* output = context->GetOutputShape(index);
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        output->SetDimNum(4);
        output->SetDim(0, input->GetDim(0));
        output->SetDim(1, 8);
        output->SetDim(2, input->GetDim(1));
        output->SetDim(3, 64);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MinicpmoQkvPack(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    for (size_t index = 0; index < 3; ++index) {
        context->SetOutputDataType(index, dtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinicpmoQkvPack)
    .InferShape(InferShape4MinicpmoQkvPack)
    .InferDataType(InferDataType4MinicpmoQkvPack);

}  // namespace ops
