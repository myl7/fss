// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <concepts>
#include <cuda/std/span>

/**
 * Pseudorandom generator (PRG) interface.
 * @ingroup fss_api
 *
 * Required to be cryptographically secure.
 *
 * @tparam mul Seed expansion factor. An int4 seed contains 16 B, and Gen()
 * returns mul * 16 B.
 */
template <typename Prg, int mul>
concept Prgable = requires(Prg prg, int4 seed) {
  { prg.Gen(seed) } -> std::same_as<cuda::std::array<int4, mul>>;
};

/**
 * Optional selected-half extension of Prgable for positive even mul.
 *
 * GenHalf(seed, false) returns the first mul / 2 outputs of Gen(seed).
 * GenHalf(seed, true) returns the last mul / 2 outputs in the same order.
 * Both halves must be bit-exact with Gen() under the same PRG configuration.
 * GenHalf() has the same execution-space, lifetime, and concurrency requirements
 * as Gen(). Custom PRGs opt in by providing the member, without registration.
 *
 * @tparam Prg User-provided or library-provided PRG type.
 * @tparam mul Positive even seed expansion factor, as in Prgable.
 * @ingroup fss_api
 */
template <typename Prg, int mul>
concept HasGenHalf = mul > 0 && mul % 2 == 0 && Prgable<Prg, mul> && requires(Prg prg, int4 seed, bool right) {
  { prg.GenHalf(seed, right) } -> std::same_as<cuda::std::array<int4, mul / 2>>;
};

/**
 * Optional eight-seed batch extension of Prgable.
 *
 * Gen8() borrows eight seeds through cuda::std::span<const int4, 8>.
 * Custom PRGs opt in by providing Gen8(seeds). No registration, inheritance,
 * or specialization is required. Full-domain host evaluators use this method
 * when available and otherwise call Gen() eight times.
 *
 * The result is indexed as [seed_index][output_index]. For every j in [0, 8),
 * Gen8(seeds)[j] must equal Gen(seeds[j]) from the same PRG configuration.
 * This semantic requirement is the implementer's responsibility.
 * Gen8() must be callable on the host for host batch evaluation.
 *
 * Host-side full-domain evaluation of Dcf and Dpf expands eight seeds per
 * call. It calls Gen8() when the PRG provides it and calls Gen() eight times
 * otherwise. Dmpf takes the same path through its inner Dpf when bucket_bits
 * is at least 6. Point evaluation, key generation, and GPU evaluation never
 * call Gen8().
 *
 * Gen8() can help when the backend expands eight seeds faster than the
 * generic eight-call fallback does. For the library's Aes128MmoRaw on one
 * x86 host with one thread, Gen8() made Dcf's full-domain evaluation faster.
 * It made no difference to Dpf's that the measurement could separate from
 * its own bias.
 *
 * Gen8() must be bit-exact with Gen(). Full-domain evaluation uses Gen8() and
 * point evaluation uses Gen(), so a mismatch makes the two disagree without
 * any error.
 *
 * @tparam Prg User-provided or library-provided PRG type.
 * @tparam mul Seed expansion factor, as in Prgable.
 * @ingroup fss_api
 */
template <typename Prg, int mul>
concept HasGen8 = Prgable<Prg, mul> && requires(Prg prg, cuda::std::span<const int4, 8> seeds) {
  { prg.Gen8(seeds) } -> std::same_as<cuda::std::array<cuda::std::array<int4, mul>, 8>>;
};

// -----------------------------------------------------------------------------

namespace fss::detail {

template <int mul, typename Prg>
  requires(mul > 0 && mul % 2 == 0 && Prgable<Prg, mul>)
__host__ __device__ inline cuda::std::array<int4, mul / 2> GenHalf(Prg &prg, int4 seed, bool right) {
  if constexpr (HasGenHalf<Prg, mul>) {
    return prg.GenHalf(seed, right);
  } else {
    auto full = prg.Gen(seed);
    cuda::std::array<int4, mul / 2> out{};
    int offset = right ? mul / 2 : 0;
    for (int i = 0; i < mul / 2; ++i) out[i] = full[i + offset];
    return out;
  }
}

// Keep the optional batched entry point outside the public PRG requirement.
template <int mul, typename Prg>
cuda::std::array<cuda::std::array<int4, mul>, 8> GenBatch(Prg &prg, cuda::std::span<const int4, 8> seeds) {
  if constexpr (HasGen8<Prg, mul>) {
    return prg.Gen8(seeds);
  } else {
    cuda::std::array<cuda::std::array<int4, mul>, 8> out{};
    for (int j = 0; j < 8; ++j) out[j] = prg.Gen(seeds[j]);
    return out;
  }
}

}  // namespace fss::detail
