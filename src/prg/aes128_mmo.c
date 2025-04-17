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

uint8_t gRoundKeys[kBlocks][kLambda / 16][176];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda % 16 == 0);
  assert(state_len >= kBlocks * kLambda);
  for (int i = 0; i < kBlocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      KeyExpansion(gRoundKeys[i][j], state + i * kLambda + j * 16);
    }
  }
}

void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  assert(out_len % kLambda == 0);
  assert(out_len <= kBlocks * kLambda);
  int blocks = out_len / kLambda;
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < kLambda / 16; j++) {
      memcpy(out + i * kLambda + j * 16, seed + j * 16, 16);
      encrypt(out + i * kLambda + j * 16, gRoundKeys[i][j]);
    }
  }
  for (int i = 0; i < blocks; i++) {
    xor_bytes(out + i * kLambda, seed, kLambda);
  }
}
