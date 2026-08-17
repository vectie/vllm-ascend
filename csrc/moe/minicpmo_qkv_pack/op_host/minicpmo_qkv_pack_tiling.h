/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MINICPMO_QKV_PACK_TILING_H
#define MINICPMO_QKV_PACK_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MinicpmoQkvPackTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, frames);
    TILING_DATA_FIELD_DEF(uint32_t, heads);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

struct MinicpmoQkvPackCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(MinicpmoQkvPack, MinicpmoQkvPackTilingData)

}  // namespace optiling

#endif
