// SPDX-License-Identifier: Apache-2.0
/**
 * @file prp.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Pseudorandom permutation (PRP) interface.
 */

#pragma once
#include <cuda_runtime.h>
#include <concepts>

/**
 * Small-domain pseudorandom permutation (PRP) interface.
 * @ingroup fss_api
 *
 * Required to be cryptographically secure.
 *
 * Permu(seed, x, domain_size) is a keyed permutation on [0, domain_size).
 * The seed is a 16-byte key.
 */
template <typename Prp>
concept Permutable = requires(Prp prp, int4 seed, __uint128_t x, __uint128_t domain_size) {
  { prp.Permu(seed, x, domain_size) } -> std::same_as<__uint128_t>;
};
