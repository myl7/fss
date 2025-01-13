// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <dpf_api.h>
#include <string.h>
#include <assert.h>
#include "aes-brute-force/aes_ni.h"
#include "utils.h"

__m128i gKeySchedules[2][20];

void prg_init(const uint8_t keys[2][16]) {
  assert(kLambda == 16);
  aes128_load_key(keys[0], gKeySchedules[0]);
  aes128_load_key(keys[1], gKeySchedules[1]);
}

void prg(uint8_t *out, const uint8_t *seed) {
  aes128_enc(gKeySchedules[0], seed, out);
  aes128_enc(gKeySchedules[1], seed, out + 16);

  xor_bytes(out, seed, 16);
  xor_bytes(out + 16, seed, 16);
}
