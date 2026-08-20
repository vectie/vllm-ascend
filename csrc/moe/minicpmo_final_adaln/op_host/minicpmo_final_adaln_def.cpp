/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_def_registry.h"

namespace ops {

class MinicpmoFinalAdaln : public OpDef {
public:
    explicit MinicpmoFinalAdaln(const char* name) : OpDef(name)
    {
        for (const char* input : {"hidden", "modulation", "weight", "bias"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        }
        this->Output("projected")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(MinicpmoFinalAdaln);

}  // namespace ops
