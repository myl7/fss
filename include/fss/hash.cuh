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
 * Optional four-message batch extension of Hashable.
 * @ingroup fss_api
 *
 * Hash4() borrows four independent 64B messages through
 * cuda::std::span<const cuda::std::array<int4, 4>, 4> and maps them to four
 * 32B digests, indexed as [message_index][output_index]. Custom hash backends
 * opt in by providing Hash4(msgs); no registration, inheritance, or
 * specialization is required. Host evaluators use this method when available
 * and otherwise call Hash() four times.
 *
 * For every j in [0, 4), Hash4(msgs)[j] must equal Hash(msgs[j]) word for
 * word, evaluated with the same object and configuration. Hash4() must be
 * callable on the host. It borrows the messages only for the duration of the
 * call, must not modify them, and must not retain pointers or references to
 * them; the returned digests are owned by the caller. Hash4() must not change
 * the backend hash configuration, and scalar Hash() calls made afterwards with
 * the same object remain valid. Concurrent-call requirements are the same as
 * for the existing scalar Hash(); this concept adds no thread-safety promise.
 *
 * This concept checks only the expression shape and cannot verify these
 * semantic or host-callability requirements.
 *
 * @tparam Hash User-provided or library-provided hash type.
 */
template <typename Hash>
concept HasHash4 = Hashable<Hash> &&
    requires(Hash hash, cuda::std::span<const cuda::std::array<int4, 4>, 4> msgs) {
  { hash.Hash4(msgs) } -> std::same_as<cuda::std::array<cuda::std::array<int4, 2>, 4>>;
};

/**
 * Collision-resistant and XOR-collision-resistant hash interface.
 *
 * VDPF and VDMPF pass the packed coordinate as the first block and the
 * clamped leaf seed as the second block. All 128 bits of the first block
 * and the 127 nonreserved bits of the second block are message data.
 * Bit 0 of the second block's fourth word is reserved; the schemes pass it
 * as zero. The built-in tuple backends use it for the two digest tags.
 * A custom backend must preserve these message bits over the instantiated
 * scheme's coordinate domain and satisfy the scheme's collision-resistance
 * and XOR-collision-resistance assumptions.
 * This concept checks only the expression and return type,
 * not these semantic or cryptographic properties.
 *
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
