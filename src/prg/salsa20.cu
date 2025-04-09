// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)
// Based on https://en.wikipedia.org/wiki/Salsa20

#ifndef BLOCK_NUM
#define BLOCK_NUM 1
#endif

#include <fss_decl.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d) (b ^= ROTL(a + d, 7), c ^= ROTL(b + a, 9), d ^= ROTL(c + b, 13), a ^= ROTL(d + c, 18))
#define ROUNDS 12

HOST_DEVICE void salsa20_block(uint32_t x[16], uint32_t in[16], uint64_t pos, const uint32_t nonce[2]) {
  in[0] = 'e' | ('x' << 8) | ('p' << 16) | ('a' << 24);
  in[5] = 'n' | ('d' << 8) | (' ' << 16) | ('1' << 24);
  in[10] = '6' | ('-' << 8) | ('b' << 16) | ('y' << 24);
  in[15] = 't' | ('e' << 8) | (' ' << 16) | ('k' << 24);

  in[6] = nonce[0];
  in[7] = nonce[1];
  in[8] = pos & 0xFFFFFFFF;
  in[9] = pos >> 32;

  for (int i = 0; i < 16; i++) {
    x[i] = in[i];
  }

  for (int i = 0; i < ROUNDS; i += 2) {
    // Odd round
    QR(x[0], x[4], x[8], x[12]);   // column 1
    QR(x[5], x[9], x[13], x[1]);   // column 2
    QR(x[10], x[14], x[2], x[6]);  // column 3
    QR(x[15], x[3], x[7], x[11]);  // column 4
    // Even round
    QR(x[0], x[1], x[2], x[3]);      // row 1
    QR(x[5], x[6], x[7], x[4]);      // row 2
    QR(x[10], x[11], x[8], x[9]);    // row 3
    QR(x[15], x[12], x[13], x[14]);  // row 4
  }

  for (int i = 0; i < 16; i++) {
    x[i] += in[i];
  }
}

HOST_DEVICE void salsa20_expand_key(uint32_t x[16]) {
  x[11] = x[1];
  x[12] = x[2];
  x[13] = x[3];
  x[14] = x[4];
}

DEVICE_CONST uint32_t gNonces[BLOCK_NUM][2];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda == 16);
  assert(state_len == BLOCK_NUM * 8);
#ifdef __CUDACC__
  cudaMemcpyToSymbol(gNonces, state, state_len);
#else
  memcpy(gNonces, state, state_len);
#endif
}

HOST_DEVICE void prg(uint8_t *out, const uint8_t *seed) {
  uint32_t in[16];
  const uint32_t *seed_int = (const uint32_t *)seed;
  in[1] = seed_int[0];
  in[2] = seed_int[1];
  in[3] = seed_int[2];
  in[4] = seed_int[3];
  salsa20_expand_key(in);

  for (int i = 0; i < BLOCK_NUM; i++) {
    uint32_t x[16];
    salsa20_block(x, in, 0, gNonces[i]);
    memcpy(out + i * 32, x, 32);
  }
}
