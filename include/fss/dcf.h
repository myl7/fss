// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file dcf.h
 */

#pragma once

#include <fss/prelude.h>
#include <fss/group.h>
#include <fss/prg.h>

#define kDcfCwLen (kLambda * 2 + 1)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DCF keygen.
 * @param k Output allocated already. See @ref Key for allocation and no need to init.
 * @param cf
 * @param sbuf Buffer whose len >= 10 * lambda.
 * `s0s` as input is stored at first 2 * lambda bytes.
 * No need to init other bytes.
 */
FSS_CUDA_HOST_DEVICE void dcf_gen(Key k, CmpFunc cf, uint8_t *sbuf);

/**
 * DCF eval at 1 input point.
 * @param sbuf Buffer whose len >= 6 * lambda.
 * `s0s[b]` as input is stored at first lambda bytes.
 * Output is stored at first lambda bytes.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref group.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dcf_gen()
 * @param x Evaluated input point. Like `alpha` of @ref CmpFunc.
 */
FSS_CUDA_HOST_DEVICE void dcf_eval(uint8_t *sbuf, uint8_t b, Key k, Bits x);

/**
 * DCF full domain eval i.e. eval at all input points.
 * @param sbuf Buffer whose len >= 2 ^ `x_bitlen` * lambda.
 * `s0s[b]` as input is stored at first lambda bytes.
 * Output is contiguously stored at each lambda bytes of `sbuf`.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref group.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dcf_gen()
 * @param x_bitlen Bitlen of input points, resulting in 2 ^ `x_bitlen` input points in total
 */
void dcf_eval_full_domain(uint8_t *sbuf, uint8_t b, Key k, int x_bitlen);

#ifdef __cplusplus
}
#endif
