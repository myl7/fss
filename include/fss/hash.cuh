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
 */
template <typename Hash>
concept Hashable = requires(Hash hash, cuda::std::span<const int4, 4> msg) {
    { hash.Hash(msg) } -> std::same_as<cuda::std::array<int4, 2>>;
};

/**
 * Collision-resistant and XOR-collision-resistant hash interface.
 */
template <typename Hash>
concept XorHashable = requires(Hash hash, cuda::std::tuple<int4, const int4> msg) {
    { hash.Hash(msg) } -> std::same_as<cuda::std::array<int4, 4>>;
};
