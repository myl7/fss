// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#ifndef BLOCK_NUM
#define BLOCK_NUM 2
#endif

#include <fss_decl.h>
#include <string.h>
#include <assert.h>
#include "aes-brute-force/aes_ni.h"
#include "../utils.h"

__m128i gKeySchedules[BLOCK_NUM][20];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda == 16);
  assert(state_len == BLOCK_NUM * 16);
  for (int i = 0; i < BLOCK_NUM; i++) {
    aes128_load_key(state + i * 16, gKeySchedules[i]);
  }
}

void prg(uint8_t *out, const uint8_t *seed) {
  for (int i = 0; i < BLOCK_NUM; i++) {
    aes128_enc(gKeySchedules[i], seed, out + i * 16);
    xor_bytes(out + i * 16, seed, 16);
  }
}
