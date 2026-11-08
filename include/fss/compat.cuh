// SPDX-License-Identifier: Apache-2.0
/**
 * @file compat.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Single build-time switch for the deprecated v1 Cw-form compatibility entries.
 *
 * FSS_ENABLE_V1_COMPAT is the only definition site of the switch. It is a numeric
 * macro: the default 1 keeps the deprecated Cw-form overloads and their companion
 * types available, and an explicit 0 removes them so only the current
 * split-component APIs remain.
 *
 * Define it to the same value in every translation unit of a program, including
 * ones that only include FSS headers indirectly. The scheme classes gain or lose
 * members with this macro, so mixed settings create distinct class definitions
 * and violate the one-definition rule.
 */

#pragma once

#ifndef FSS_ENABLE_V1_COMPAT
#define FSS_ENABLE_V1_COMPAT 1
#endif
