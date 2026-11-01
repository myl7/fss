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
