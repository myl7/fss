// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * @brief Pseudorandom generator (PRG) interface
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <concepts>

// TODO: Clamped bit

template <typename Prg, int mul>
concept Prgable = requires(Prg prg, int4 seed) {
    { prg.Gen(seed) } -> std::same_as<cuda::std::array<int4, mul>>;
};
