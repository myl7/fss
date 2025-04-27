// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file dpf.h
 */

#pragma once

#include <fss/group.h>
#include <fss/prg.h>

#define kDpfCwLen (kLambda + 1)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DPF keygen.
 * @param k Output allocated already. See @ref Key for allocation and no need to init.
 * @param pf
 * @param sbuf Buffer whose len >= 6 * lambda.
 * `s0s` as input is stored at first 2 * lambda bytes.
 * No need to init other bytes.
 */
FSS_CUDA_HOST_DEVICE void dpf_gen(Key k, PointFunc pf, uint8_t *sbuf);

/**
 * DPF eval at 1 input point.
 * @param sbuf Buffer whose len >= 3 * lambda.
 * `s0s[b]` as input is stored at first lambda bytes.
 * Output is stored at first lambda bytes.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref group.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dpf_gen()
 * @param x Evaluated input point. Like `alpha` of @ref PointFunc.
 */
FSS_CUDA_HOST_DEVICE void dpf_eval(uint8_t *sbuf, uint8_t b, Key k, Bits x);

// TODO: Move all alloc info to fss_prelude.h.
// eval_full_domain: Each node allocs 2 * lambda.
// gen/eval: No allocation happens in it.

/**
 * DPF full domain eval i.e. eval at all input points.
 * @param sbuf Buffer whose len >= 2 ^ `x_bitlen` * lambda.
 * `s0s[b]` as input is stored at first lambda bytes.
 * Output is contiguously stored at each lambda bytes of `sbuf`.
 * Output is little-endian and viewed as a group element.
 * Output's MSB is always 0. See @ref group.h for details.
 * No need to init other bytes.
 * @param b Party bit, 0/1
 * @param k Gen by @ref dpf_gen()
 * @param x_bitlen Bitlen of input points, resulting in 2 ^ `x_bitlen` input points in total
 */
void dpf_eval_full_domain(uint8_t *sbuf, uint8_t b, Key k, int x_bitlen);

#ifdef __cplusplus
}
#endif
