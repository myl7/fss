// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file fss_decl.h
 *
 * # 1bit Truncation for Output Group
 *
 * For a group element, though it is stored as @ref kLambda bytes, only @ref kLambda * 8 - 1 bits are used.
 * Its MSB is always 0, and set to 0 if any input is not.
 * Taking DPF as an example, we do this because DPF requires a PRG that has lambda bytes -> 2 * lambda + 2 bytes, and we make lambda = kLambda * 8 - 1.
 * We choose MSB to be unset because we can truncate it during adding, which is easy to handle.
 */

#pragma once

#include <fss_prelude.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * PRG.
 * @param out Output whose len depends on usage.
 * For DPF, its len = 2 * @ref kLambda.
 * For DCF, its len = 4 * @ref kLambda.
 * @param seed Input whose len = @ref kLambda
 */
HOST_DEVICE void prg(uint8_t *out, const uint8_t *seed);

/**
 * `val` = `val` + `rhs`, and `rhs` is unchanged.
 * `val` and `rhs` are group elements and little-endian.
 * Their MSB are always 0 (input/output). See @ref fss_decl.h for details.
 */
HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs);

/**
 * `val` = -`val`.
 * `val` is a group element and little-endian.
 * Its MSB is always 0 (input/output). See @ref fss_decl.h for details.
 */
HOST_DEVICE void group_neg(uint8_t *val);

/**
 * `val` = 0.
 * `val` is a group element and little-endian.
 * Its MSB is always 0 (input/output). See @ref fss_decl.h for details.
 */
HOST_DEVICE void group_zero(uint8_t *val);

#ifdef __cplusplus
}
#endif
