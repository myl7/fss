// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <fss_decl.h>

#define kDcfCwLen (kLambda * 2 + 1)

enum Bound {
  kGtAlpha,
  kLtAlpha,
};

typedef struct {
  Bits alpha;
  const uint8_t *beta;
  enum Bound bound;
} CmpFunc;

typedef struct {
  uint8_t *cws;
  uint8_t *cw_np1;
} DcfKey;

#ifdef __cplusplus
extern "C" {
#endif

HOST_DEVICE void dcf_gen(DcfKey k, CmpFunc cf, uint8_t *sbuf);
HOST_DEVICE void dcf_eval(uint8_t *sbuf, uint8_t b, DcfKey k, Bits x);

#ifdef __cplusplus
}
#endif
