// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <fss/dcf.h>
#include <string.h>
#include "utils.h"

// Load the 1bit t from MSB, so we can truncate during adding
FSS_CUDA_HOST_DEVICE static inline void load_st(uint8_t *s, uint8_t *t) {
  *t = get_bit_lsb(s, kLambda * 8 - 1);
  set_bit_lsb(s, kLambda * 8 - 1, 0);
}

static inline void set_st(uint8_t *s, uint8_t t) {
  set_bit_lsb(s, kLambda * 8 - 1, t);
}

FSS_CUDA_HOST_DEVICE static inline void load_sst(uint8_t *ss, uint8_t *t0, uint8_t *t1) {
  load_st(ss, t0);
  load_st(ss + kLambda, t1);
}

FSS_CUDA_HOST_DEVICE static inline void load_svst(uint8_t *svs, uint8_t *t0, uint8_t *t1) {
  load_st(svs, t0);
  set_bit_lsb(svs + kLambda, kLambda * 8 - 1, 0);
  load_st(svs + kLambda * 2, t1);
  set_bit_lsb(svs + kLambda * 3, kLambda * 8 - 1, 0);
}

// Save the 2bit tl tr in an extra byte
FSS_CUDA_HOST_DEVICE static inline void set_cwt(uint8_t *cw, uint8_t tl, uint8_t tr) {
  cw[kLambda * 2] = tl << 1 | tr;
}

FSS_CUDA_HOST_DEVICE static inline void get_cwt(const uint8_t *cw, uint8_t *tl, uint8_t *tr) {
  *tl = cw[kLambda * 2] >> 1;
  *tr = cw[kLambda * 2] & 1;
}

// | s0 | s1 | s0l | v0l | s0r | v0r | s1l | v1l | s1r | v1r |
// | ss      | sv0s                  | sv1s                  |
FSS_CUDA_HOST_DEVICE void dcf_gen(Key k, CmpFunc cf, uint8_t *sbuf) {
  uint8_t *ss = sbuf;
  uint8_t *s0 = ss;
  uint8_t *s1 = ss + kLambda;
  uint8_t *v = k.cw_np1;
  group_zero(v);
  uint8_t t0, t1;
  load_sst(ss, &t0, &t1);
  t0 = 0;
  t1 = 1;
  Point p = cf.point;

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

  for (int i = 0; i < p.alpha.bitlen; i++) {
    prg(sv0s, 4 * kLambda, s0);
    prg(sv1s, 4 * kLambda, s1);
    load_svst(sv0s, &t0l, &t0r);
    load_svst(sv1s, &t1l, &t1r);

    // Actually get MSB first
    uint8_t alpha_i = get_bit_lsb(p.alpha.bytes, p.alpha.bitlen - i - 1);
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
    memcpy(v0_lose, p.beta, kLambda);
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
FSS_CUDA_HOST_DEVICE void dcf_eval(uint8_t *sbuf, uint8_t b, Key k, Bits x) {
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

    prg(svs, 4 * kLambda, s);
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

#include <assert.h>
#include <stdlib.h>
#include <omp.h>

void dcf_eval_full_domain_node(int depth, uint8_t *sbufl, uint8_t *sbufr, uint8_t b, Key k, uint8_t *vs_alloc) {
  uint8_t *s = sbufl;
  uint8_t *v = sbufl + kLambda;
  uint8_t t;
  load_st(s, &t);

  uint8_t *svs = (uint8_t *)malloc(kLambda * 4);
  assert(svs != NULL);
  uint8_t *sl = svs;
  uint8_t *vl = svs + kLambda;
  uint8_t *sr = svs + kLambda * 2;
  uint8_t *vr = svs + kLambda * 3;
  uint8_t tl, tr;

  const uint8_t *cw = k.cws + depth * kDcfCwLen;
  const uint8_t *s_cw = cw;
  const uint8_t *v_cw = cw + kLambda;
  uint8_t tl_cw, tr_cw;
  get_cwt(cw, &tl_cw, &tr_cw);

  prg(svs, 4 * kLambda, s);
  load_svst(svs, &tl, &tr);
  if (t) {
    xor_bytes(sl, s_cw, kLambda);
    xor_bytes(sr, s_cw, kLambda);
    tl ^= tl_cw;
    tr ^= tr_cw;
  }

  uint8_t *vl_out = v;
  uint8_t *vr_out = sbufr + kLambda;
  if (vs_alloc) {
    vl_out = vs_alloc;
    memcpy(vl_out, v, kLambda);
    vr_out = vs_alloc + kLambda;
    memcpy(vr_out, v, kLambda);
  } else {
    memcpy(vr_out, v, kLambda);
  }

  if (t) {
    group_add(vl, v_cw);
    group_add(vr, v_cw);
  }
  if (b) {
    group_neg(vl);
    group_neg(vr);
  }
  group_add(vl_out, vl);
  group_add(vr_out, vr);

  // sr uses the space of v, so set sr after using v
  memcpy(sbufl, sl, kLambda);
  set_st(sbufl, tl);
  memcpy(sbufr, sr, kLambda);
  set_st(sbufr, tr);
  free(svs);
}

void dcf_eval_full_domain_leaf(uint8_t *sbuf, uint8_t b, Key k, uint8_t *v) {
  uint8_t *s = sbuf;
  uint8_t t;
  load_st(s, &t);

  if (t) group_add(s, k.cw_np1);
  if (b) group_neg(s);
  group_add(v, s);
  memcpy(s, v, kLambda);
}

void dcf_eval_full_domain_subtree(
  int depth, uint8_t *sbuf, size_t l, size_t r, uint8_t b, Key k, int x_bitlen, uint8_t *v_alloc, int par_depth) {
  assert(kLambda * (1ULL << (x_bitlen - depth)) == r - l);

  if (depth == x_bitlen) {
    dcf_eval_full_domain_leaf(sbuf + l, b, k, v_alloc);
    return;
  } else {
    assert(v_alloc == NULL);
  }

  size_t mid = (l + r) / 2;

  if (depth == x_bitlen - 1) {
    v_alloc = (uint8_t *)malloc(2 * kLambda);
    assert(v_alloc != NULL);
    dcf_eval_full_domain_node(depth, sbuf + l, sbuf + mid, b, k, v_alloc);
  } else {
    dcf_eval_full_domain_node(depth, sbuf + l, sbuf + mid, b, k, v_alloc);
  }

  uint8_t *vl_alloc = v_alloc ? v_alloc : NULL;
  uint8_t *vr_alloc = v_alloc ? v_alloc + kLambda : NULL;
  if (depth < par_depth) {
#pragma omp parallel
#pragma omp single
    {
#pragma omp task
      { dcf_eval_full_domain_subtree(depth + 1, sbuf, l, mid, b, k, x_bitlen, vl_alloc, par_depth); }
#pragma omp task
      { dcf_eval_full_domain_subtree(depth + 1, sbuf, mid, r, b, k, x_bitlen, vr_alloc, par_depth); }
#pragma omp taskwait
    }
  } else {
    dcf_eval_full_domain_subtree(depth + 1, sbuf, l, mid, b, k, x_bitlen, vl_alloc, par_depth);
    dcf_eval_full_domain_subtree(depth + 1, sbuf, mid, r, b, k, x_bitlen, vr_alloc, par_depth);
  }

  if (depth == x_bitlen - 1) {
    free(v_alloc);
  }
}

void dcf_eval_full_domain(uint8_t *sbuf, uint8_t b, Key k, int x_bitlen) {
  uint8_t *s = sbuf;
  uint8_t *v = sbuf + kLambda;
  group_zero(v);
  uint8_t t = b;
  set_st(s, t);

  int threads = omp_get_max_threads();
  int par_depth = 0;
  while ((1 << par_depth) <= threads) {
    par_depth++;
  }
  par_depth--;

  size_t sbuf_len = kLambda * (1ULL << x_bitlen);
  dcf_eval_full_domain_subtree(0, sbuf, 0, sbuf_len, b, k, x_bitlen, NULL, par_depth);
}
