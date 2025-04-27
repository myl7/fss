// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <fss/prelude.h>

// Get least significant bit first from little-endian bytes
FSS_CUDA_HOST_DEVICE static inline uint8_t get_bit_lsb(const uint8_t *bytes_le, int i) {
  return (bytes_le[i / 8] >> (i % 8)) & 1;
}

// Set least significant bit first to little-endian bytes
FSS_CUDA_HOST_DEVICE static inline void set_bit_lsb(uint8_t *bytes_le, int i, uint8_t bit) {
  if (bit) {
    bytes_le[i / 8] |= 1 << (i % 8);
  } else {
    bytes_le[i / 8] &= ~(1 << (i % 8));
  }
}

FSS_CUDA_HOST_DEVICE static inline void xor_bytes(uint8_t *val, const uint8_t *rhs, int len) {
  for (int i = 0; i < len; i++) {
    val[i] ^= rhs[i];
  }
}
