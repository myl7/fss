// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file cw_mac_bytes.h
 */

#pragma once

#include <fss/prelude.h>
#include <sodium.h>

/**
 * Len of each write key of `wkeys` of @ref gen_cw_mac()
 */
#define kCwMacWkeyLen crypto_core_ristretto255_SCALARBYTES

/**
 * Len of each public write key of `pub_wkeys` of @ref commit_cw_mac()
 */
#define kCwMacPubWkeyLen crypto_core_ristretto255_BYTES

/**
 * Len of MAC of @ref gen_cw_mac()
 */
#define kCwMacLen crypto_core_ristretto255_BYTES

/**
 * Len of commitment of @ref commit_cw_mac()
 */
#define kCwMacCommitLen crypto_core_ristretto255_BYTES

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gen write key
 * @param wkey Output write key
 */
void gen_wkey(uint8_t *wkey);

/**
 * Gen public write key
 * @param pub_wkey Output public write key
 * @param wkey Write key
 */
void gen_pub_wkey(uint8_t *pub_wkey, const uint8_t *wkey);

/**
 * Gen Carter-Wegman MAC and share it into 2 shares.
 * @param t0 Output share 0
 * @param t1 Output share 1
 * @param sbufs0 Eval output points of party 0.
 * Len of each output point is lambda.
 * Full domain eval output `sbuf` can be directly used.
 * @param sbufs1 Like `sbufs0` of party 1
 * @param sbuf_num Num of output points
 * @param sbuf_step Step between output points.
 * `sbuf_step` > lambda allows gaps between output points.
 * @param wkeys Field scalar elements as write keys corresponding to each input point.
 * Holding write keys of a set of input points allows their output points to be non-zeros.
 * Gen by @ref gen_wkey().
 */
void gen_cw_mac(uint8_t *t0, uint8_t *t1, const uint8_t *sbufs0, const uint8_t *sbufs1, int sbuf_num, int sbuf_step,
  const uint8_t *wkeys);

/**
 * Commit Carter-Wegman MAC share into a commitment to be exchanged.
 * @param beta Output commitment
 * @param b Party bit, 0/1
 * @param t Gen by @ref gen_cw_mac()
 * @param sbufs Same as `sbufs0` of @ref gen_cw_mac()
 * @param sbuf_num Same as `sbuf_num` of @ref gen_cw_mac()
 * @param sbuf_step Same as `sbuf_step` of @ref gen_cw_mac()
 * @param pub_wkeys Field elements `g^key` as public write keys corresponding to each input point.
 * Gen by @ref gen_pub_wkey() from write keys.
 */
void commit_cw_mac(uint8_t *beta, uint8_t b, const uint8_t *t, const uint8_t *sbufs, int sbuf_num, int sbuf_step,
  const uint8_t *pub_wkeys);

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
