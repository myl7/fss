// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file dcf.h
 */

#pragma once

#include <fss_decl.h>

#define kDcfCwLen (kLambda * 2 + 1)

enum Bound {
  /**
   * Output = `beta` when input < `alpha`, otherwise output = 0
   */
  kLtAlpha,
  /**
   * Output = `beta` when input > `alpha`, otherwise output = 0
   */
  kGtAlpha,
};

/**
 * Comparison function.
 * See @ref Bound for its def based on `bound`.
 */
typedef struct {
  Bits alpha;
  /**
   * Little-endian @ref kLambda bytes viewed as a group element.
   * Its MSB is ignored and assumed to be 0. See @ref fss_decl.h for details.
   */
  uint8_t *beta;
  enum Bound bound;
} CmpFunc;

/**
 * DCF key.
 * DCF key + `s0s[0]` and DCF key + `s0s[1]` are 2 shares given to 2 parties.
 * Designed for easy serialization.
 */
typedef struct {
  /**
   * Correction words whose len = @ref kDcfCwLen * @ref kLambda
   */
  uint8_t *cws;
  /**
   * Last correction word whose len = @ref kLambda
   */
  uint8_t *cw_np1;
} DcfKey;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DCF keygen.
 * @param k Output allocated already. See @ref DcfKey for allocation and no need to init.
 * @param cf
 * @param sbuf Buffer whose len >= 10 * @ref kLambda.
 * `s0s` as input is stored at first 2 * @ref kLambda bytes.
 * No need to init other bytes.
 */
HOST_DEVICE void dcf_gen(DcfKey k, CmpFunc cf, uint8_t *sbuf);

/**
 * DCF eval at 1 input point.
 * @param sbuf Buffer whose len >= 6 * @ref kLambda.
 * `s0s[b]` as input is stored at first @ref kLambda bytes.
 * Output is stored at first @ref kLambda bytes.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref fss_decl.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dcf_gen()
 * @param x Evaluated input point. Like `alpha` of @ref CmpFunc.
 */
HOST_DEVICE void dcf_eval(uint8_t *sbuf, uint8_t b, DcfKey k, Bits x);

/**
 * DCF full domain eval i.e. eval at all input points.
 * @param sbuf Buffer whose len >= 2 ^ `x_bitlen` * @ref kLambda.
 * `s0s[b]` as input is stored at first @ref kLambda bytes.
 * Output is contiguously stored at each @ref kLambda bytes of `sbuf`.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref fss_decl.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dcf_gen()
 * @param x_bitlen Bitlen of input points, resulting in 2 ^ `x_bitlen` input points in total
 */
void dcf_eval_full_domain(uint8_t *sbuf, uint8_t b, DcfKey k, int x_bitlen);

#ifdef __cplusplus
}
#endif
