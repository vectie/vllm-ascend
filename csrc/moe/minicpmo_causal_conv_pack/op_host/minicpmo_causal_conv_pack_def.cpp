/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_def_registry.h"

namespace ops {

class MinicpmoCausalConvPack : public OpDef {
public:
    explicit MinicpmoCausalConvPack(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("packed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("new_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(MinicpmoCausalConvPack);

}  // namespace ops
