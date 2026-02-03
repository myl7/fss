// SPDX-License-Identifier: Apache-2.0
/**
 * @file util.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 */

#pragma once
#include <cuda_runtime.h>

namespace fss::util {

__host__ __device__ int4 Xor(int4 lhs, int4 rhs) {
    return {lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w};
}

__host__ __device__ int4 SetLsb(int4 val, bool bit) {
    if (bit) val.w |= 1;
    else val.w &= ~1;
    return val;
}

__host__ __device__ bool GetLsb(int4 val) {
    return (val.w & 1) != 0;
}

}  // namespace fss::util
