// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
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
 * @tparam mul Requires the PRG to output `mul` times of the seed size, i.e., `mul * 16` B.
 */
template <typename Prg, int mul>
concept Prgable = requires(Prg prg, int4 seed) {
    { prg.Gen(seed) } -> std::same_as<cuda::std::array<int4, mul>>;
};
