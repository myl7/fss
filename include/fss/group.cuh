// SPDX-License-Identifier: Apache-2.0
/**
 * @file group.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * @brief Group interface
 */

#pragma once
#include <cuda_runtime.h>
#include <concepts>
#include <type_traits>

// TODO: Template for gtests
// TODO: Clamped bit
// TODO: Assert clamped bit
// TODO: Save extra bit at clamped bit

template <typename Group>
concept Groupable =
    std::is_default_constructible_v<Group> && requires(Group lhs, Group rhs, int4 buf) {
        { lhs + rhs } -> std::same_as<Group>;
        { -lhs } -> std::same_as<Group>;
        { Group::From(buf) } -> std::same_as<Group>;
        { lhs.Into() } -> std::same_as<int4>;
    };
