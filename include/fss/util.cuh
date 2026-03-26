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
#include <omp.h>

namespace fss::util {

__host__ __device__ inline int4 Xor(int4 lhs, int4 rhs) {
    return {lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w};
}

__host__ __device__ inline cuda::std::array<int4, 2> Xor(
    cuda::std::span<const int4, 2> lhs, cuda::std::span<const int4, 2> rhs) {
    return {Xor(lhs[0], rhs[0]), Xor(lhs[1], rhs[1])};
}

__host__ __device__ inline cuda::std::array<int4, 4> Xor(
    cuda::std::span<const int4, 4> lhs, cuda::std::span<const int4, 4> rhs) {
    return {Xor(lhs[0], rhs[0]), Xor(lhs[1], rhs[1]), Xor(lhs[2], rhs[2]), Xor(lhs[3], rhs[3])};
}

__host__ __device__ inline int4 SetLsb(int4 val, bool bit) {
    if (bit) val.w |= 1;
    else val.w &= ~1;
    return val;
}

__host__ __device__ inline bool GetLsb(int4 val) {
    return (val.w & 1) != 0;
}

inline int ResolveParDepth(int par_depth) {
    if (par_depth >= 0) return par_depth;
    int d = 0, threads = omp_get_max_threads();
    while ((1 << d) < threads) ++d;
    return d;
}

template <typename In>
__host__ __device__ inline int4 Pack(In val) {
    int4 buf = {0, 0, 0, 0};
    if constexpr (sizeof(In) <= 4) {
        buf.x = static_cast<int>(val);
    } else if constexpr (sizeof(In) <= 8) {
        auto v = static_cast<uint64_t>(val);
        buf.x = static_cast<int>(v & 0xFFFFFFFF);
        buf.y = static_cast<int>((v >> 32) & 0xFFFFFFFF);
    } else {
        auto v = static_cast<__uint128_t>(val);
        buf.x = static_cast<int>(v & 0xFFFFFFFF);
        buf.y = static_cast<int>((v >> 32) & 0xFFFFFFFF);
        buf.z = static_cast<int>((v >> 64) & 0xFFFFFFFF);
        buf.w = static_cast<int>((v >> 96) & 0xFFFFFFFF);
    }
    return buf;
}

}  // namespace fss::util
