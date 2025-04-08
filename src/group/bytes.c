// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <fss_decl.h>
#include "../utils.h"

HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs) {
  xor_bytes(val, rhs, kLambda);
}

HOST_DEVICE void group_neg(uint8_t *val) {}
