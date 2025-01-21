// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <stdint.h>
#include <dpf_api.h>

#define kCwLen (kLambda + 1)

typedef struct {
  const uint8_t *bytes;
  int bitlen;
} Bits;

typedef struct {
  Bits alpha;
  const uint8_t *beta;
} PointFunc;

typedef struct {
  uint8_t *cws;
  uint8_t *cw_np1;
} DpfKey;

HOST_DEVICE void dpf_gen(DpfKey k, PointFunc pf, uint8_t *sbuf);
HOST_DEVICE void dpf_eval(uint8_t *sbuf, uint8_t b, const DpfKey k, Bits x);
