/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MINICPMO_FINAL_ADALN_TILING_H
#define MINICPMO_FINAL_ADALN_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MinicpmoFinalAdalnTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, nPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, modulatedOffset);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);
END_TILING_DATA_DEF;

struct MinicpmoFinalAdalnCompileInfo {
    uint32_t totalAicCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(MinicpmoFinalAdaln, MinicpmoFinalAdalnTilingData)

}  // namespace optiling

#endif
