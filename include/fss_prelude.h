// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <stdint.h>

#ifndef kLambda
#define kLambda 16
#endif

#define kBetaBitlen (kLambda * 8 - 1)

#if __CUDACC__
#define HOST_DEVICE __host__ __device__
#define DEVICE_CONST __constant__
#else
#define HOST_DEVICE
#define DEVICE_CONST
#endif

typedef struct {
  uint8_t *bytes;
  int bitlen;
} Bits;
