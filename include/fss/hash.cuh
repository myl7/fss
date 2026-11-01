// SPDX-License-Identifier: Apache-2.0
/**
 * @file hash.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <concepts>

/**
 * Collision-resistant hash interface.
 * @ingroup fss_api
 */
template <typename Hash>
concept Hashable = requires(Hash hash, cuda::std::span<const int4, 4> msg) {
  { hash.Hash(msg) } -> std::same_as<cuda::std::array<int4, 2>>;
};

/**
 * Collision-resistant and XOR-collision-resistant hash interface.
 * @ingroup fss_api
 */
template <typename Hash>
concept XorHashable = requires(Hash hash, cuda::std::tuple<int4, const int4> msg) {
  { hash.Hash(msg) } -> std::same_as<cuda::std::array<int4, 4>>;
};

/**
 * Circular-correlation-robust (CCR) hash interface for HalfTree schemes.
 * @ingroup fss_api
 *
 * Hash() maps a 128-bit block to a 128-bit block and must satisfy circular
 * correlation robustness, beyond the expression requirements checked here.
 * HalfTree applies its instance seed S separately, as Hash(S xor x).
 * This requirement is distinct from Hashable and XorHashable.
 */
template <typename Hash>
concept CcrHashable = requires(Hash hash, int4 block) {
  { hash.Hash(block) } -> std::same_as<int4>;
};

// -----------------------------------------------------------------------------

namespace fss::detail {

template <typename Hash>
cuda::std::array<int4, 8> HashBatch(Hash &hash, const cuda::std::array<int4, 8> &blocks) {
  cuda::std::array<int4, 8> out;
  for (int i = 0; i < 8; ++i) out[i] = hash.Hash(blocks[i]);
  return out;
}

}  // namespace fss::detail
