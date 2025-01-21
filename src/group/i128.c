// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <dpf_api.h>

HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs) {
  __int128_t *val128 = (__int128_t *)val;
  const __int128_t *rhs128 = (const __int128_t *)rhs;
  *val128 += *rhs128;
}

HOST_DEVICE void group_neg(uint8_t *val) {
  __int128_t *val128 = (__int128_t *)val;
  *val128 = -*val128;
}
