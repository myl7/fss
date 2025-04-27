// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file prg.h
 */

#pragma once

#include <fss/prelude.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * PRG.
 * @param out Output
 * @param out_len Output len.
 * For DPF, its len = 2 * lambda.
 * For DCF, its len = 4 * lambda.
 * @param seed Input whose len = lambda
 */
FSS_CUDA_HOST_DEVICE void prg(uint8_t *out, int out_len, const uint8_t *seed);

/**
 * Init PRG.
 * Same state and seed give same output.
 * This is not called by the library. Users can leave it empty if not needed.
 * @param state
 * @param state_len Len of `state`.
 * For DPF, its len should >= 2 * lambda.
 * For DCF, its len should >= 4 * lambda.
 */
void prg_init(const uint8_t *state, int state_len);

/**
 * Free PRG
 */
void prg_free();

#ifdef __cplusplus
}
#endif
