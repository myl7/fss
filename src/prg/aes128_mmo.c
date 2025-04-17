// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#ifndef kBlocks
#define kBlocks 2
#endif

#include <fss_decl.h>
#include <string.h>
#include <assert.h>
#include "../utils.h"

extern void KeyExpansion(uint8_t *RoundKey, const uint8_t *Key);
extern void encrypt(uint8_t *state, const uint8_t *RoundKey);

uint8_t gRoundKeys[kBlocks][176];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda == 16);
  assert(state_len >= kBlocks * 16);
  for (int i = 0; i < kBlocks; i++) {
    KeyExpansion(gRoundKeys[i], state + i * 16);
  }
}

void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  assert(out_len % 16 == 0);
  assert(out_len / 16 <= kBlocks);
  for (int i = 0; i < out_len / 16; i++) {
    memcpy(out + i * 16, seed, 16);
    encrypt(out + i * 16, gRoundKeys[i]);
    xor_bytes(out + i * 16, seed, 16);
  }
}
