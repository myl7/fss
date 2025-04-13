// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file cw_mac_bytes.h
 */

#pragma once

#include <fss_prelude.h>

#ifdef __cplusplus
extern "C" {
#endif

void gen_cw_mac(uint8_t *t0, uint8_t *t1, const uint8_t *sbufs0, const uint8_t *sbufs1, int sbuf_num, int sbuf_step,
  const uint8_t *keys);
void commit_cw_mac(uint8_t *beta, uint8_t b, const uint8_t *t, const uint8_t *sbufs, int sbuf_num, int sbuf_step,
  const uint8_t *pubkeys);
int verify_cw_mac(uint8_t *beta0, const uint8_t *beta1);

#ifdef __cplusplus
}
#endif
