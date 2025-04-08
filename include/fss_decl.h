// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <stdint.h>

#ifndef kLambda
#define kLambda 16
#endif

#if __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#ifdef __cplusplus
extern "C" {
#endif

HOST_DEVICE void prg(uint8_t *out, const uint8_t *seed);

HOST_DEVICE void group_add(uint8_t *val, const uint8_t *rhs);
HOST_DEVICE void group_neg(uint8_t *val);

#ifdef __cplusplus
}
#endif
