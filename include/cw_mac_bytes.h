// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file cw_mac_bytes.h
 */

#pragma once

#include <fss_prelude.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO: Methods to gen keys/pubkeys

/**
 * Gen Carter-Wegman MAC and share it into 2 shares.
 * @param t0 Output share 0
 * @param t1 Output share 1
 * @param sbufs0 Eval output points of party 0.
 * Len of each output point is @ref kLambda.
 * Full domain eval output `sbuf` can be directly used.
 * @param sbufs1 Like `sbufs0` of party 1
 * @param sbuf_num Num of output points
 * @param sbuf_step Step between output points.
 * `sbuf_step` > @ref kLambda allows gaps between output points.
 * @param keys Field scalar elements as keys corresponding to each input point.
 * Holding keys of a set of input points allows their output points to be non-zeros.
 * Gen each key with `crypto_core_ristretto255_scalar_random()`.
 */
void gen_cw_mac(uint8_t *t0, uint8_t *t1, const uint8_t *sbufs0, const uint8_t *sbufs1, int sbuf_num, int sbuf_step,
  const uint8_t *keys);

/**
 * Commit Carter-Wegman MAC share into a commitment to be exchanged.
 * @param beta Output commitment
 * @param b Party bit, 0/1
 * @param t Gen by @ref gen_cw_mac()
 * @param sbufs Same as `sbufs0` of @ref gen_cw_mac()
 * @param sbuf_num Same as `sbuf_num` of @ref gen_cw_mac()
 * @param sbuf_step Same as `sbuf_step` of @ref gen_cw_mac()
 * @param pubkeys Field elements `g^key` as public keys corresponding to each input point.
 * Gen each pubkey with `crypto_scalarmult_ristretto255_base()` for `key` of @ref gen_cw_mac().
 */
void commit_cw_mac(uint8_t *beta, uint8_t b, const uint8_t *t, const uint8_t *sbufs, int sbuf_num, int sbuf_step,
  const uint8_t *pubkeys);

/**
 * Verify Carter-Wegman MAC commitment.
 * @param beta0 Commitment of party 0.
 * Commitments as field elements, the method stores beta0 + beta1 in `beta0` in addition to returning.
 * @param beta1 Commitment of party 1
 * @return 1 if valid, otherwise 0
 */
int verify_cw_mac(uint8_t *beta0, const uint8_t *beta1);

#ifdef __cplusplus
}
#endif
