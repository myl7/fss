// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <fss_decl.h>

#define kDpfCwLen (kLambda + 1)

typedef struct {
  Bits alpha;
  uint8_t *beta;
} PointFunc;

typedef struct {
  uint8_t *cws;
  uint8_t *cw_np1;
} DpfKey;

#ifdef __cplusplus
extern "C" {
#endif

HOST_DEVICE void dpf_gen(DpfKey k, PointFunc pf, uint8_t *sbuf);
HOST_DEVICE void dpf_eval(uint8_t *sbuf, uint8_t b, DpfKey k, Bits x);

#ifdef __cplusplus
}
#endif
