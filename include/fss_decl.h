// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <fss_prelude.h>

#ifdef __cplusplus
extern "C" {
#endif

HOST_DEVICE void prg(uint8_t *out, const uint8_t *seed);

HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs);
HOST_DEVICE void group_neg(uint8_t *val);
HOST_DEVICE void group_zero(uint8_t *val);

#ifdef __cplusplus
}
#endif
