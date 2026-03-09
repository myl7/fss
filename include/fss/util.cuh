// SPDX-License-Identifier: Apache-2.0
/**
 * @file util.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>

namespace fss::util {

__host__ __device__ int4 Xor(int4 lhs, int4 rhs) {
    return {lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w};
}

__host__ __device__ cuda::std::array<int4, 2> Xor(
    cuda::std::span<const int4, 2> lhs, cuda::std::span<const int4, 2> rhs) {
    return {Xor(lhs[0], rhs[0]), Xor(lhs[1], rhs[1])};
}

__host__ __device__ cuda::std::array<int4, 4> Xor(
    cuda::std::span<const int4, 4> lhs, cuda::std::span<const int4, 4> rhs) {
    return {Xor(lhs[0], rhs[0]), Xor(lhs[1], rhs[1]), Xor(lhs[2], rhs[2]), Xor(lhs[3], rhs[3])};
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
