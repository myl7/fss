// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <stdint.h>
#include <fss_prelude.h>

HOST_DEVICE static inline uint8_t get_bit(const uint8_t *bytes, int i) {
  return (bytes[i / 8] >> (7 - i % 8)) & 1;
}

HOST_DEVICE static inline void set_bit(uint8_t *bytes, int i, uint8_t bit) {
  if (bit) {
    bytes[i / 8] |= 1 << (7 - i % 8);
  } else {
    bytes[i / 8] &= ~(1 << (7 - i % 8));
  }
}

HOST_DEVICE static inline void xor_bytes(uint8_t *val, const uint8_t *rhs, int len) {
  for (int i = 0; i < len; i++) {
    val[i] ^= rhs[i];
  }
}
