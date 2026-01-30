// SPDX-License-Identifier: Apache-2.0
/**
 * @file bytes.c
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 */

#include <fss/group.h>
#include <string.h>
#include "../utils.h"

FSS_CUDA_HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs) {
  xor_bytes(val, rhs, kLambda);
}

FSS_CUDA_HOST_DEVICE void group_neg(uint8_t *val) {}

FSS_CUDA_HOST_DEVICE void group_zero(uint8_t *val) {
  memset(val, 0, kLambda);
}
