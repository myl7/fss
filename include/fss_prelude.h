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

typedef struct {
  const uint8_t *bytes;
  int bitlen;
} Bits;
