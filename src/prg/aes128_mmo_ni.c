// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#ifndef kBlocks
#define kBlocks 2
#endif

#include <fss_decl.h>
#include <string.h>
#include <assert.h>
#include "aes-brute-force/aes_ni.h"
#include "../utils.h"

__m128i gKeySchedules[kBlocks][kLambda / 16][kRounds + 1];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda % 16 == 0);
  assert(state_len >= kBlocks * kLambda);
  for (int i = 0; i < kBlocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      aes128_load_key_enc_only(state + i * kLambda + j * 16, gKeySchedules[i][j]);
    }
  }
}

void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  assert(out_len % kLambda == 0);
  assert(out_len <= kBlocks * kLambda);
  int blocks = out_len / kLambda;
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      aes128_enc(gKeySchedules[i][j], seed + j * 16, out + i * kLambda + j * 16);
    }
  }
  for (int i = 0; i < blocks; i++) {
    xor_bytes(out + i * kLambda, seed, kLambda);
  }
}
