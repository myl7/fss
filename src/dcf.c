// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <dcf.h>
#include <string.h>
#include "utils.h"

// Load the 1bit t from MSB, so we can truncate during adding
HOST_DEVICE static inline void load_st(uint8_t *s, uint8_t *t) {
  *t = get_bit_lsb(s, kLambda * 8 - 1);
  set_bit_lsb(s, kLambda * 8 - 1, 0);
}

HOST_DEVICE static inline void load_sst(uint8_t *ss, uint8_t *t0, uint8_t *t1) {
  load_st(ss, t0);
  load_st(ss + kLambda, t1);
}

HOST_DEVICE static inline void load_svst(uint8_t *svs, uint8_t *t0, uint8_t *t1) {
  load_st(svs, t0);
  set_bit_lsb(svs + kLambda, kLambda * 8 - 1, 0);
  load_st(svs + kLambda * 2, t1);
  set_bit_lsb(svs + kLambda * 3, kLambda * 8 - 1, 0);
}

// Save the 2bit tl tr in an extra byte
HOST_DEVICE static inline void set_cwt(uint8_t *cw, uint8_t tl, uint8_t tr) {
  cw[kLambda * 2] = tl << 1 | tr;
}

HOST_DEVICE static inline void get_cwt(const uint8_t *cw, uint8_t *tl, uint8_t *tr) {
  *tl = cw[kLambda * 2] >> 1;
  *tr = cw[kLambda * 2] & 1;
}

// | s0 | s1 | s0l | v0l | s0r | v0r | s1l | v1l | s1r | v1r |
// | ss      | sv0s                  | sv1s                  |
HOST_DEVICE void dcf_gen(DcfKey k, CmpFunc cf, uint8_t *sbuf) {
  uint8_t *ss = sbuf;
  uint8_t *s0 = ss;
  uint8_t *s1 = ss + kLambda;
  uint8_t *v = k.cw_np1;
  group_zero(v);
  uint8_t t0, t1;
  load_sst(ss, &t0, &t1);
  t0 = 0;
  t1 = 1;

  uint8_t *sv0s = sbuf + kLambda * 2;
  uint8_t *s0l = sv0s;
  uint8_t *v0l = sv0s + kLambda;
  uint8_t *s0r = sv0s + kLambda * 2;
  uint8_t *v0r = sv0s + kLambda * 3;
  uint8_t *sv1s = sbuf + kLambda * 6;
  uint8_t *s1l = sv1s;
  uint8_t *v1l = sv1s + kLambda;
  uint8_t *s1r = sv1s + kLambda * 2;
  uint8_t *v1r = sv1s + kLambda * 3;
  uint8_t t0l, t0r, t1l, t1r;

  for (int i = 0; i < cf.alpha.bitlen; i++) {
    prg(sv0s, s0);
    prg(sv1s, s1);
    load_svst(sv0s, &t0l, &t0r);
    load_svst(sv1s, &t1l, &t1r);

    // Actually get MSB first
    uint8_t alpha_i = get_bit_lsb(cf.alpha.bytes, cf.alpha.bitlen - i - 1);
    uint8_t *cw = k.cws + i * kDcfCwLen;

    uint8_t *s0_lose = alpha_i ? s0l : s0r;
    uint8_t *s1_lose = alpha_i ? s1l : s1r;

    uint8_t *v0_lose = alpha_i ? v0l : v0r;
    uint8_t *v1_lose = alpha_i ? v1l : v1r;
    uint8_t *v0_keep = alpha_i ? v0r : v0l;
    uint8_t *v1_keep = alpha_i ? v1r : v1l;

    uint8_t *s_cw = cw;
    memcpy(s_cw, s0_lose, kLambda);
    xor_bytes(s_cw, s1_lose, kLambda);

    uint8_t *v_cw = cw + kLambda;
    memcpy(v_cw, v1_lose, kLambda);
    group_neg(v0_lose);
    group_add(v_cw, v0_lose);
    memcpy(v0_lose, v, kLambda);
    group_neg(v0_lose);
    group_add(v_cw, v0_lose);
    if (t1) group_neg(v_cw);
    memcpy(v0_lose, cf.beta, kLambda);
    set_bit_lsb(v0_lose, kLambda * 8 - 1, 0);
    if (t1) group_neg(v0_lose);
    switch (cf.bound) {
      case kLtAlpha:
        if (alpha_i) group_add(v_cw, v0_lose);
        break;
      case kGtAlpha:
        if (!alpha_i) group_add(v_cw, v0_lose);
        break;
    }
    group_neg(v1_keep);
    group_add(v, v1_keep);
    group_add(v, v0_keep);
    memcpy(v0_lose, v_cw, kLambda);
    if (t1) group_neg(v0_lose);
    group_add(v, v0_lose);

    uint8_t tl_cw, tr_cw;
    tl_cw = t0l ^ t1l ^ alpha_i ^ 1;
    tr_cw = t0r ^ t1r ^ alpha_i;
    set_cwt(cw, tl_cw, tr_cw);

    uint8_t *s0_keep = alpha_i ? s0r : s0l;
    uint8_t *s1_keep = alpha_i ? s1r : s1l;
    uint8_t t0_keep = alpha_i ? t0r : t0l;
    uint8_t t1_keep = alpha_i ? t1r : t1l;
    uint8_t t_cw_keep = alpha_i ? tr_cw : tl_cw;

    memcpy(s0, s0_keep, kLambda);
    if (t0) xor_bytes(s0, s_cw, kLambda);
    memcpy(s1, s1_keep, kLambda);
    if (t1) xor_bytes(s1, s_cw, kLambda);

    if (t0) t0 = t0_keep ^ t_cw_keep;
    else t0 = t0_keep;
    if (t1) t1 = t1_keep ^ t_cw_keep;
    else t1 = t1_keep;
  }

  group_neg(s0);
  group_add(s1, s0);
  group_neg(v);
  group_add(s1, v);
  if (t1) group_neg(s1);
  memcpy(k.cw_np1, s1, kLambda);
}

// | s | v | sl | vl | sr | vr |
// |       | svs               |
HOST_DEVICE void dcf_eval(uint8_t *sbuf, uint8_t b, DcfKey k, Bits x) {
  uint8_t *s = sbuf;
  uint8_t *v = sbuf + kLambda;
  group_zero(v);
  uint8_t t;
  load_st(s, &t);
  t = b;

  uint8_t *svs = sbuf + kLambda * 2;
  uint8_t *sl = svs;
  uint8_t *vl = svs + kLambda;
  uint8_t *sr = svs + kLambda * 2;
  uint8_t *vr = svs + kLambda * 3;
  uint8_t tl, tr;

  for (int i = 0; i < x.bitlen; i++) {
    const uint8_t *cw = k.cws + i * kDcfCwLen;
    const uint8_t *s_cw = cw;
    const uint8_t *v_cw = cw + kLambda;
    uint8_t tl_cw, tr_cw;
    get_cwt(cw, &tl_cw, &tr_cw);

    prg(svs, s);
    load_svst(svs, &tl, &tr);
    if (t) {
      xor_bytes(sl, s_cw, kLambda);
      xor_bytes(sr, s_cw, kLambda);
      tl ^= tl_cw;
      tr ^= tr_cw;
    }

    // Actually get MSB first
    uint8_t x_i = get_bit_lsb(x.bytes, x.bitlen - i - 1);

    uint8_t *v_delta = x_i ? vr : vl;
    if (t) group_add(v_delta, v_cw);
    if (b) group_neg(v_delta);
    group_add(v, v_delta);

    memcpy(s, x_i ? sr : sl, kLambda);
    t = x_i ? tr : tl;
  }

  if (t) group_add(s, k.cw_np1);
  if (b) group_neg(s);
  group_add(v, s);
  memcpy(s, v, kLambda);
}
