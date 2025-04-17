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

__m128i gKeySchedules[kBlocks][kRounds + 1];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda == 16);
  assert(state_len >= kBlocks * 16);
  for (int i = 0; i < kBlocks; i++) {
    aes128_load_key_enc_only(state + i * 16, gKeySchedules[i]);
  }
}

void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  assert(out_len % 16 == 0);
  assert(out_len / 16 <= kBlocks);
  for (int i = 0; i < out_len / 16; i++) {
    aes128_enc(gKeySchedules[i], seed, out + i * 16);
    xor_bytes(out + i * 16, seed, 16);
  }
}
