// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <dpf_api.h>
#include <string.h>
#include <assert.h>
#include "../utils.h"

extern void KeyExpansion(uint8_t *RoundKey, const uint8_t *Key);
extern void encrypt(uint8_t *state, const uint8_t *RoundKey);

uint8_t gRoundKeys[2][176];

void prg_init(const uint8_t *state, int state_len) {
  assert(state_len == 32);
  assert(kLambda == 16);
  KeyExpansion(gRoundKeys[0], state);
  KeyExpansion(gRoundKeys[1], state + 16);
}

void prg(uint8_t *out, const uint8_t *seed) {
  memcpy(out, seed, 16);
  memcpy(out + 16, seed, 16);

  encrypt(out, gRoundKeys[0]);
  encrypt(out + 16, gRoundKeys[1]);

  xor_bytes(out, seed, 16);
  xor_bytes(out + 16, seed, 16);
}
