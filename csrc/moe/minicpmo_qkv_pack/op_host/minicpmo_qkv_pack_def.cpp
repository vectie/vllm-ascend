/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_def_registry.h"

namespace ops {

class MinicpmoQkvPack : public OpDef {
public:
    explicit MinicpmoQkvPack(const char* name) : OpDef(name)
    {
        for (const char* input : {"q", "k", "v"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        }
        for (const char* output : {"q_out", "k_out", "v_out"}) {
            this->Output(output)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .FormatList({ge::FORMAT_ND});
        }

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

OP_ADD(MinicpmoQkvPack);

}  // namespace ops
