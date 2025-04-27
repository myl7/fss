// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)
// Based on https://en.wikipedia.org/wiki/Salsa20

#ifndef kRounds
  #define kRounds 20
#endif

#ifndef kBlocks
  #define kBlocks 1
#endif

#include <fss/prg.h>
#include <assert.h>
#include <string.h>
#ifdef __CUDACC__
  #include <cuda_runtime.h>
#endif

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
// clang-format off
#define QR(a, b, c, d) ( \
  b ^= ROTL(a + d, 7), \
  c ^= ROTL(b + a, 9), \
  d ^= ROTL(c + b, 13), \
  a ^= ROTL(d + c, 18))
// clang-format on

FSS_CUDA_HOST_DEVICE void salsa_block(uint32_t x[16], uint32_t in[16], uint64_t pos, const uint32_t nonce[2]) {
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

  for (int i = 0; i < kRounds; i += 2) {
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

FSS_CUDA_HOST_DEVICE void salsa_expand_key(uint32_t x[16]) {
  x[11] = x[1];
  x[12] = x[2];
  x[13] = x[3];
  x[14] = x[4];
}

FSS_CUDA_CONSTANT uint32_t gNonces[kBlocks][(kLambda + 31) / 32][2];

void prg_init(const uint8_t *state, int state_len) {
  assert(kLambda % 16 == 0);
  assert(state_len >= kBlocks * kLambda / 4);
#ifdef __CUDACC__
  cudaMemcpyToSymbol(gNonces, state, state_len);
#else
  memcpy(gNonces, state, state_len);
#endif
}

void prg_free() {}

FSS_CUDA_HOST_DEVICE void prg(uint8_t *out, int out_len, const uint8_t *seed) {
  assert(out_len % (2 * kLambda) == 0);
  assert(out_len <= kBlocks * kLambda * 2);
  int blocks = out_len / kLambda / 2;
  int left_block = kLambda / 16 % 2;
  uint32_t in[16];
  uint32_t x[16];
  for (int i = 0; i < blocks; i++) {
    int j = 0;
    for (; j < kLambda / 32; j++) {
      const uint32_t *seed_int = (const uint32_t *)seed + j * 32;
      in[1] = seed_int[0];
      in[2] = seed_int[1];
      in[3] = seed_int[2];
      in[4] = seed_int[3];
      in[11] = seed_int[4];
      in[12] = seed_int[5];
      in[13] = seed_int[6];
      in[14] = seed_int[7];
      salsa_block(x, in, 0, gNonces[i][j]);
      memcpy(out + i * 2 * kLambda + j * 32, x, 32);
      memcpy(out + i * 2 * kLambda + kLambda + j * 32, x + 8, 32);
    }
    if (left_block) {
      assert(j * 32 + 16 == kLambda);
      const uint32_t *seed_int = (const uint32_t *)seed + j * 32;
      in[1] = seed_int[0];
      in[2] = seed_int[1];
      in[3] = seed_int[2];
      in[4] = seed_int[3];
      salsa_expand_key(in);
      salsa_block(x, in, 0, gNonces[i][j]);
      memcpy(out + i * 2 * kLambda + j * 32, x, 16);
      memcpy(out + i * 2 * kLambda + kLambda + j * 32, x + 8, 16);
    }
  }
}
