// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file dpf.h
 */

#pragma once

#include <fss_decl.h>

#define kDpfCwLen (kLambda + 1)

/**
 * Point function.
 * Output = `beta` when input = `alpha`, otherwise output = 0.
 */
typedef struct {
  Bits alpha;
  /**
   * Little-endian @ref kLambda bytes viewed as a group element.
   * Its MSB is ignored and assumed to be 0. See @ref fss_decl.h for details.
   */
  uint8_t *beta;
} PointFunc;

/**
 * DPF key.
 * DPF key + `s0s[0]` and DPF key + `s0s[1]` are 2 shares given to 2 parties.
 * Designed for easy serialization.
 */
typedef struct {
  /**
   * Correction words whose len = @ref kDpfCwLen * @ref kLambda
   */
  uint8_t *cws;
  /**
   * Last correction word whose len = @ref kLambda
   */
  uint8_t *cw_np1;
} DpfKey;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DPF keygen.
 * @param k Output allocated already. See @ref DpfKey for allocation and no need to init.
 * @param pf
 * @param sbuf Buffer whose len >= 6 * @ref kLambda.
 * `s0s` as input is stored at first 2 * @ref kLambda bytes.
 * No need to init other bytes.
 */
HOST_DEVICE void dpf_gen(DpfKey k, PointFunc pf, uint8_t *sbuf);

/**
 * DPF eval at 1 input point.
 * @param sbuf Buffer whose len >= 3 * @ref kLambda.
 * `s0s[b]` as input is stored at first @ref kLambda bytes.
 * Output is stored at first @ref kLambda bytes.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref fss_decl.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dpf_gen()
 * @param x Evaluated input point. Like `alpha` of @ref PointFunc.
 */
HOST_DEVICE void dpf_eval(uint8_t *sbuf, uint8_t b, DpfKey k, Bits x);

// TODO: Move all alloc info to fss_prelude.h.
// eval_full_domain: Each node allocs 2 * lambda.
// gen/eval: No allocation happens in it.

/**
 * DPF full domain eval i.e. eval at all input points.
 * @param sbuf Buffer whose len >= 2 ^ `x_bitlen` * @ref kLambda.
 * `s0s[b]` as input is stored at first @ref kLambda bytes.
 * Output is contiguously stored at each @ref kLambda bytes of `sbuf`.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref fss_decl.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dpf_gen()
 * @param x_bitlen Bitlen of input points, resulting in 2 ^ `x_bitlen` input points in total
 */
void dpf_eval_full_domain(uint8_t *sbuf, uint8_t b, DpfKey k, int x_bitlen);

#ifdef __cplusplus
}
#endif
