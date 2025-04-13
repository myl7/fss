// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <cw_mac_bytes.h>
#include <string.h>
#include <assert.h>
#include <sodium.h>

void hash_sbuf(uint8_t *scalar, const uint8_t *sbuf) {
  assert(crypto_hash_sha512_BYTES * 8 >= 317);
  uint8_t hash[crypto_hash_sha512_BYTES];
  crypto_hash_sha512(hash, sbuf, kLambda);
  crypto_core_ristretto255_scalar_reduce(scalar, hash);
}

void gen_cw_mac(uint8_t *t0, uint8_t *t1, const uint8_t *sbufs0, const uint8_t *sbufs1, int sbuf_num, int sbuf_step,
  const uint8_t *keys) {
  uint8_t t[crypto_core_ristretto255_SCALARBYTES];
  memset(t, 0, crypto_core_ristretto255_SCALARBYTES);
  uint8_t scalar0[crypto_core_ristretto255_SCALARBYTES];
  uint8_t scalar1[crypto_core_ristretto255_SCALARBYTES];

  for (int i = 0; i < sbuf_num; i++) {
    const uint8_t *sbuf0 = sbufs0 + i * sbuf_step;
    const uint8_t *sbuf1 = sbufs1 + i * sbuf_step;
    const uint8_t *key = keys + i * crypto_core_ristretto255_SCALARBYTES;
    hash_sbuf(scalar0, sbuf0);
    hash_sbuf(scalar1, sbuf1);
    crypto_core_ristretto255_scalar_negate(scalar1, scalar1);
    crypto_core_ristretto255_scalar_add(scalar0, scalar0, scalar1);
    crypto_core_ristretto255_scalar_mul(scalar0, key, scalar0);
    crypto_core_ristretto255_scalar_add(t, t, scalar0);
  }
  crypto_core_ristretto255_scalar_random(scalar1);
  crypto_core_ristretto255_scalar_sub(scalar0, t, scalar1);
  crypto_scalarmult_ristretto255_base(t0, scalar0);
  crypto_scalarmult_ristretto255_base(t1, scalar1);
}

void commit_cw_mac(uint8_t *beta, uint8_t b, const uint8_t *t, const uint8_t *sbufs, int sbuf_num, int sbuf_step,
  const uint8_t *pubkeys) {
  uint8_t t_delta[crypto_core_ristretto255_BYTES];
  memset(t_delta, 0, crypto_core_ristretto255_BYTES);
  uint8_t scalar[crypto_core_ristretto255_SCALARBYTES];
  uint8_t t_delta_i[crypto_core_ristretto255_BYTES];

  for (int i = 0; i < sbuf_num; i++) {
    const uint8_t *sbuf = sbufs + i * sbuf_step;
    const uint8_t *pubkey = pubkeys + i * crypto_core_ristretto255_BYTES;
    hash_sbuf(scalar, sbuf);
    if (b) crypto_core_ristretto255_scalar_negate(scalar, scalar);
    int _ = crypto_scalarmult_ristretto255(t_delta_i, scalar, pubkey);
    crypto_core_ristretto255_add(t_delta, t_delta, t_delta_i);
  }
  crypto_core_ristretto255_sub(beta, t_delta, t);
}

int verify_cw_mac(uint8_t *beta0, const uint8_t *beta1) {
  crypto_core_ristretto255_add(beta0, beta0, beta1);
  for (int i = 0; i < crypto_core_ristretto255_BYTES; i++) {
    if (beta0[i] != 0) return 0;
  }
  return 1;
}
