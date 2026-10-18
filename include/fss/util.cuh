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
#include <cstddef>
#include <cstdint>
#include <climits>

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

// Resolve a tree split depth without shifting beyond a machine word.
inline int ResolveParDepth(int par_depth, int useful_depth = sizeof(size_t) * CHAR_BIT - 1) {
  if (par_depth < 0) {
    par_depth = 0;
    unsigned threads = static_cast<unsigned>(omp_get_max_threads());
    size_t partitions = 1;
    while (partitions < threads && par_depth < useful_depth) {
      partitions *= 2;
      ++par_depth;
    }
  }
  return par_depth < useful_depth ? par_depth : useful_depth;
}

inline size_t ResolveParPartitions(int par_depth, size_t count) {
  if (count == 0) return 0;
  int depth = ResolveParDepth(par_depth);
  size_t partitions = 1;
  while (depth-- > 0 && partitions < count) {
    if (partitions > count / 2) return count;
    partitions *= 2;
  }
  return partitions;
}

inline int ResolveParThreads(int par_depth, size_t count) {
  size_t partitions = ResolveParPartitions(par_depth, count);
  int threads = omp_get_max_threads();
  return partitions < static_cast<size_t>(threads) ? static_cast<int>(partitions) : threads;
}

// Each scheduled partition owns a contiguous range. A depth of zero never
// enters an OpenMP region, including when called from a shallow fallback path.
template <typename Function>
inline void ParallelFor(int par_depth, size_t count, Function function) {
  size_t partitions = ResolveParPartitions(par_depth, count);
  if (partitions <= 1) {
    for (size_t i = 0; i < count; ++i) function(i);
    return;
  }
  int threads = ResolveParThreads(par_depth, count);
#pragma omp parallel for num_threads(threads)
  for (size_t partition = 0; partition < partitions; ++partition) {
    size_t begin = partition * (count / partitions) + (partition < count % partitions ? partition : count % partitions);
    size_t end = begin + count / partitions + (partition < count % partitions);
    for (size_t i = begin; i < end; ++i) function(i);
  }
}

// Stored seed words are signed int32 values. Increment their bit pattern modulo
// 2^32, then convert back without an out-of-range signed conversion.
__host__ __device__ inline int IncrementWord(int value) {
  uint32_t bits = static_cast<uint32_t>(value) + uint32_t{1};
  return bits <= static_cast<uint32_t>(INT_MAX) ? static_cast<int>(bits)
                                             : -1 - static_cast<int>(UINT32_MAX - bits);
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
