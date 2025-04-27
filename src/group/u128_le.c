// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <fss/group.h>
#include <string.h>
#include "../utils.h"

#if kLambda != 16
#error "kLambda must be 16 for u128_le group"
#endif

FSS_CUDA_HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs) {
  __uint128_t *val127 = (__uint128_t *)val;
  __uint128_t *rhs127 = (__uint128_t *)rhs;
  *val127 += *rhs127;
  set_bit_lsb(val, 127, 0);
}

FSS_CUDA_HOST_DEVICE void group_neg(uint8_t *val) {
  __uint128_t *val127 = (__uint128_t *)val;
  *val127 = -(*val127);
  set_bit_lsb(val, 127, 0);
}

FSS_CUDA_HOST_DEVICE void group_zero(uint8_t *val) {
  memset(val, 0, sizeof(__uint128_t));
}
