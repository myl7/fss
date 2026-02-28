// SPDX-License-Identifier: Apache-2.0
/**
 * @file group.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 */

#pragma once
#include <cuda_runtime.h>
#include <concepts>
#include <type_traits>

/**
 * Group interface.
 *
 * For the output domain of DPF/DCF.
 *
 * `From()`: Convert a clamped 16B to the group element.
 *
 * - **Parameters**
 *     - **buf**: Clamped 16B. Users implementing it should assert the clamped bit to be 0.
 *
 * `Into()`: Convert the group element to a clamped 16B.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's least significant bit (LSB) to 0, corresponding to $\lambda = 127$.
 * We call setting the last word's LSB to 0 as **clamping**, which is adapted from libsodium's documentation.
 * We call this LSB as the clamped bit.
 * This output domain is large enough for most applications.
 * Larger output domain of DPF can be implemented with Spectrum's large message transformation (@ref spectrum "1") by applying a PRG to outputs.
 * For general cases, you may repeat the scheme or modify the source code.
 * When modifying the source code, you need to care about the stack usage on CPU and the register usage on GPU, which is the primary reason why we fix the output domain size.
 *
 * ## References
 *
 * 1. 	Zachary Newman, Sacha Servan-Schreiber, Srinivas Devadas: Spectrum: High-bandwidth Anonymous Broadcast. NSDI 2022: 229-248. <https://www.usenix.org/conference/nsdi22/presentation/newman>. @anchor spectrum
 */
template <typename Group>
concept Groupable =
    std::is_default_constructible_v<Group> && requires(Group lhs, Group rhs, int4 buf) {
        { lhs + rhs } -> std::same_as<Group>;
        { -lhs } -> std::same_as<Group>;
        { Group::From(buf) } -> std::same_as<Group>;
        { lhs.Into() } -> std::same_as<int4>;
    };
