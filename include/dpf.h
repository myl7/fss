// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <stdint.h>
#include <dpf_api.h>

typedef struct {
  uint8_t *bytes;
  int bitlen;
} Bits;

typedef struct {
  Bits alpha;
  uint8_t *beta;
} PointFunc;

typedef struct {
  uint8_t **cws;
  uint8_t *cw_np1;
} DpfKey;

int dpf_gen(DpfKey k, const PointFunc pf, const uint8_t *s0s);
int dpf_eval(uint8_t *y, uint8_t b, const DpfKey k, Bits x);
