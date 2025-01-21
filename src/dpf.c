// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <dpf.h>
#include <string.h>
#include "utils.h"

HOST_DEVICE static inline void load_st(uint8_t *s, uint8_t *t) {
  *t = get_bit(s, kLambda * 8 - 1);
  set_bit(s, kLambda * 8 - 1, 0);
}

HOST_DEVICE static inline void set_st(uint8_t *s, uint8_t t) {
  set_bit(s, kLambda * 8 - 1, t);
}

HOST_DEVICE static inline void load_sst(uint8_t *ss, uint8_t *t0, uint8_t *t1) {
  load_st(ss, t0);
  load_st(ss + kLambda, t1);
}

HOST_DEVICE static inline void set_cwt(uint8_t *cw, uint8_t tl, uint8_t tr) {
  cw[kLambda] = tl << 1 | tr;
}

HOST_DEVICE static inline void get_cwt(const uint8_t *cw, uint8_t *tl, uint8_t *tr) {
  *tl = cw[kLambda] >> 1;
  *tr = cw[kLambda] & 1;
}

HOST_DEVICE void dpf_gen(DpfKey k, PointFunc pf, uint8_t *sbuf) {
  uint8_t *ss = sbuf;
  uint8_t *s0 = ss;
  uint8_t *s1 = ss + kLambda;
  uint8_t t0, t1;
  load_st(s0, &t0);
  t1 = t0 ^ 1;
  set_st(s1, t1);

  uint8_t *s0s = sbuf + kLambda * 2;
  uint8_t *s0l = s0s;
  uint8_t *s0r = s0s + kLambda;
  uint8_t *s1s = sbuf + kLambda * 4;
  uint8_t *s1l = s1s;
  uint8_t *s1r = s1s + kLambda;
  uint8_t t0l, t0r, t1l, t1r;

  for (int i = 0; i < pf.alpha.bitlen; i++) {
    prg(s0s, s0);
    prg(s1s, s1);
    load_sst(s0s, &t0l, &t0r);
    load_sst(s1s, &t1l, &t1r);

    uint8_t alpha_i = get_bit(pf.alpha.bytes, i);
    uint8_t *cw = k.cws + i * kCwLen;

    uint8_t *s0_lose = alpha_i ? s0l : s0r;
    uint8_t *s1_lose = alpha_i ? s1l : s1r;

    uint8_t *s_cw = cw;
    memcpy(s_cw, s0_lose, kLambda);
    xor_bytes(s_cw, s1_lose, kLambda);
    uint8_t tl_cw, tr_cw;
    tl_cw = t0l ^ t1l ^ alpha_i ^ 1;
    tr_cw = t0r ^ t1r ^ alpha_i;
    set_cwt(cw, tl_cw, tr_cw);

    uint8_t *s0_keep = alpha_i ? s0r : s0l;
    uint8_t *s1_keep = alpha_i ? s1r : s1l;
    uint8_t t0_keep = alpha_i ? t0r : t0l;
    uint8_t t1_keep = alpha_i ? t1r : t1l;
    uint8_t t_keep_cw = alpha_i ? tr_cw : tl_cw;

    memcpy(s0, s0_keep, kLambda);
    if (t0) xor_bytes(s0, s_cw, kLambda);
    memcpy(s1, s1_keep, kLambda);
    if (t1) xor_bytes(s1, s_cw, kLambda);
    if (t0) t0 = t0_keep ^ t_keep_cw;
    else t0 = t0_keep;
    if (t1) t1 = t1_keep ^ t_keep_cw;
    else t1 = t1_keep;
  }

  memcpy(k.cw_np1, pf.beta, kLambda);
  group_neg(s0);
  group_add(k.cw_np1, s0);
  group_add(k.cw_np1, s1);
  if (t1) group_neg(k.cw_np1);
}

HOST_DEVICE void dpf_eval(uint8_t *sbuf, uint8_t b, DpfKey k, Bits x) {
  uint8_t *s = sbuf;
  uint8_t t;
  load_st(s, &t);

  uint8_t *ss = sbuf + kLambda;
  uint8_t *sl = ss;
  uint8_t *sr = ss + kLambda;
  uint8_t tl, tr;

  for (int i = 0; i < x.bitlen; i++) {
    uint8_t *cw = k.cws + i * kCwLen;

    prg(ss, s);
    load_sst(ss, &tl, &tr);
    if (t) {
      xor_bytes(sl, cw, kLambda);
      xor_bytes(sr, cw, kLambda);
      uint8_t cw_tl, cw_tr;
      get_cwt(cw, &cw_tl, &cw_tr);
      tl ^= cw_tl;
      tr ^= cw_tr;
    }

    uint8_t x_i = get_bit(x.bytes, i);
    memcpy(s, x_i ? sr : sl, kLambda);
    t = x_i ? tr : tl;
  }

  if (t) group_add(s, k.cw_np1);
  if (b) group_neg(s);
}
