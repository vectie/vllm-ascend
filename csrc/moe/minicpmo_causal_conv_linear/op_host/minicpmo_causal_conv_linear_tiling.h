/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MINICPMO_CAUSAL_CONV_LINEAR_TILING_H
#define MINICPMO_CAUSAL_CONV_LINEAR_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MinicpmoCausalConvLinearTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, nPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
    TILING_DATA_FIELD_DEF(uint64_t, packedOffset);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);
END_TILING_DATA_DEF;

struct MinicpmoCausalConvLinearCompileInfo {
    uint32_t totalAicCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(MinicpmoCausalConvLinear, MinicpmoCausalConvLinearTilingData)

}  // namespace optiling

#endif
