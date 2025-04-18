// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#ifndef kBlocks
#define kBlocks 2
#endif

#include <fss_decl.h>
#include <string.h>
#include <assert.h>
#include <openssl/evp.h>
#include "../utils.h"

EVP_CIPHER_CTX *gOpensslCtxs[kBlocks][kLambda / 16];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda % 16 == 0);
  assert(state_len >= kBlocks * kLambda);
  for (int i = 0; i < kBlocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      int ret;
      EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
      assert(ctx != NULL);
      ret = EVP_EncryptInit_ex2(ctx, EVP_aes_128_ecb(), state + i * kLambda + j * 16, NULL, NULL);
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

void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  assert(out_len % kLambda == 0);
  assert(out_len <= kBlocks * kLambda);
  int blocks = out_len / kLambda;
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      int cipher_len;
      EVP_EncryptUpdate(gOpensslCtxs[i][j], out + i * kLambda + j * 16, &cipher_len, seed + j * 16, 16);
      assert(cipher_len == 16);
    }
  }
  for (int i = 0; i < blocks; i++) {
    xor_bytes(out + i * kLambda, seed, kLambda);
  }
}
