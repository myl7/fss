// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)
// Based on https://en.wikipedia.org/wiki/Salsa20#ChaCha_variant

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
  a += b, d ^= a, d = ROTL(d, 16), \
  c += d, b ^= c, b = ROTL(b, 12), \
  a += b, d ^= a, d = ROTL(d, 8), \
  c += d, b ^= c, b = ROTL(b, 7))
// clang-format on

FSS_CUDA_HOST_DEVICE void chacha_block(uint32_t x[16], uint32_t in[16], uint64_t pos, const uint32_t nonce[2]) {
  in[0] = 'e' | ('x' << 8) | ('p' << 16) | ('a' << 24);
  in[1] = 'n' | ('d' << 8) | (' ' << 16) | ('3' << 24);
  in[2] = '2' | ('-' << 8) | ('b' << 16) | ('y' << 24);
  in[3] = 't' | ('e' << 8) | (' ' << 16) | ('k' << 24);

  in[12] = pos & 0xFFFFFFFF;
  in[13] = pos >> 32;
  in[14] = nonce[0];
  in[15] = nonce[1];

  for (int i = 0; i < 16; i++) {
    x[i] = in[i];
  }

  for (int i = 0; i < kRounds; i += 2) {
    // Column rounds
    QR(x[0], x[4], x[8], x[12]);
    QR(x[1], x[5], x[9], x[13]);
    QR(x[2], x[6], x[10], x[14]);
    QR(x[3], x[7], x[11], x[15]);
    // Diagonal rounds
    QR(x[0], x[5], x[10], x[15]);
    QR(x[1], x[6], x[11], x[12]);
    QR(x[2], x[7], x[8], x[13]);
    QR(x[3], x[4], x[9], x[14]);
  }

  for (int i = 0; i < 16; i++) {
    x[i] += in[i];
  }
}

FSS_CUDA_HOST_DEVICE void chacha_expand_key(uint32_t x[16]) {
  x[8] = x[4];
  x[9] = x[5];
  x[10] = x[6];
  x[11] = x[7];
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
      in[4] = seed_int[0];
      in[5] = seed_int[1];
      in[6] = seed_int[2];
      in[7] = seed_int[3];
      in[8] = seed_int[4];
      in[9] = seed_int[5];
      in[10] = seed_int[6];
      in[11] = seed_int[7];
      chacha_block(x, in, 0, gNonces[i][j]);
      memcpy(out + i * 2 * kLambda + j * 32, x, 32);
      memcpy(out + i * 2 * kLambda + kLambda + j * 32, x + 8, 32);
    }
    if (left_block) {
      assert(j * 32 + 16 == kLambda);
      const uint32_t *seed_int = (const uint32_t *)seed + j * 32;
      in[4] = seed_int[0];
      in[5] = seed_int[1];
      in[6] = seed_int[2];
      in[7] = seed_int[3];
      chacha_expand_key(in);
      chacha_block(x, in, 0, gNonces[i][j]);
      memcpy(out + i * 2 * kLambda + j * 32, x, 16);
      memcpy(out + i * 2 * kLambda + kLambda + j * 32, x + 8, 16);
    }
  }
}
