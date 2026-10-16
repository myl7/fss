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

/**
 * Pseudorandom generator (PRG) interface.
 *
 * Required to be cryptographically secure.
 *
 * @tparam mul Requires the PRG to output mul times of the seed size, i.e., mul * 16 B.
 */
template <typename Prg, int mul>
concept Prgable = requires(Prg prg, int4 seed) {
  { prg.Gen(seed) } -> std::same_as<cuda::std::array<int4, mul>>;
};

namespace fss::detail {

/**
 * Detection of a Prgable PRG that also provides an 8-way batched Gen8().
 *
 * Gen8() applies the same construction as Gen() to 8 independent seeds at once,
 * so for every seed j, out[j] is bit-exact with Gen(seeds[j]) element-wise.
 */
template <typename Prg, int mul>
concept HasGen8 = requires(Prg prg, cuda::std::array<int4, 8> seeds) {
  { prg.Gen8(seeds) } -> std::same_as<cuda::std::array<cuda::std::array<int4, mul>, 8>>;
};

/**
 * Expands 8 seeds at once: forwards to Gen8() when the PRG provides it and
 * loops Gen() over the seeds otherwise. The two forms agree bit-exactly, so
 * callers get 8-way batching without PRG capability dispatch.
 *
 * @tparam mul See Prgable mul.
 */
template <int mul, typename Prg>
cuda::std::array<cuda::std::array<int4, mul>, 8> GenBatch(Prg &prg, const cuda::std::array<int4, 8> &seeds) {
  if constexpr (HasGen8<Prg, mul>) {
    return prg.Gen8(seeds);
  } else {
    cuda::std::array<cuda::std::array<int4, mul>, 8> out{};
    for (int j = 0; j < 8; ++j) out[j] = prg.Gen(seeds[j]);
    return out;
  }
}

}  // namespace fss::detail
