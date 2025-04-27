// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#ifndef kBlocks
  #define kBlocks 1
#endif

#include <fss/prg.h>
#include <string.h>
#include <assert.h>
#include <openssl/evp.h>
#include "../utils.h"

EVP_CIPHER_CTX *gOpensslCtxs[kBlocks][kLambda / 16];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda % 16 == 0);
  assert(state_len >= 2 * kBlocks * kLambda);
  for (int i = 0; i < kBlocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      int ret;
      EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
      assert(ctx != NULL);
      ret = EVP_EncryptInit_ex2(ctx, EVP_aes_256_ecb(), state + i * 2 * kLambda + j * 32, NULL, NULL);
      assert(ret == 1);
      ret = EVP_CIPHER_CTX_set_padding(ctx, 0);
      assert(ret == 1);
      gOpensslCtxs[i][j] = ctx;
    }
  }
}

void prg_free() {
  for (int i = 0; i < kBlocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      EVP_CIPHER_CTX_free(gOpensslCtxs[i][j]);
    }
  }
}

// `p(x)` of Hirose
static inline void p(uint8_t *block) {
  for (int i = 0; i < 16; i++) {
    block[i] = ~block[i];
  }
}

void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  uint8_t buf[16];
  assert(out_len % kLambda == 0);
  assert(out_len <= 2 * kBlocks * kLambda);
  int blocks = out_len / (2 * kLambda);
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      int cipher_len = 0;
      EVP_EncryptUpdate(gOpensslCtxs[i][j], out + i * 2 * kLambda + j * 16, &cipher_len, seed + j * 16, 16);
      assert(cipher_len == 16);
      xor_bytes(out + i * 2 * kLambda + j * 16, seed + j * 16, 16);

      cipher_len = 0;
      memcpy(buf, seed + j * 16, 16);
      p(buf);
      EVP_EncryptUpdate(gOpensslCtxs[i][j], out + i * 2 * kLambda + kLambda + j * 16, &cipher_len, buf, 16);
      assert(cipher_len == 16);
      xor_bytes(out + i * 2 * kLambda + kLambda + j * 16, buf, 16);
    }
  }
}
