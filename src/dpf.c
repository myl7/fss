// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#include <fss/dpf.h>
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

// Save the 2bit tl tr in an extra byte
FSS_CUDA_HOST_DEVICE static inline void set_cwt(uint8_t *cw, uint8_t tl, uint8_t tr) {
  cw[kLambda] = tl << 1 | tr;
}

FSS_CUDA_HOST_DEVICE static inline void get_cwt(const uint8_t *cw, uint8_t *tl, uint8_t *tr) {
  *tl = cw[kLambda] >> 1;
  *tr = cw[kLambda] & 1;
}

// | s0 | s1 | s0l | s0r | s1l | s1r |
// | ss      | s0s      | s1s        |
FSS_CUDA_HOST_DEVICE void dpf_gen(Key k, PointFunc pf, uint8_t *sbuf) {
  uint8_t *ss = sbuf;
  uint8_t *s0 = ss;
  uint8_t *s1 = ss + kLambda;
  uint8_t t0, t1;
  load_sst(ss, &t0, &t1);
  t0 = 0;
  t1 = 1;
  Point p = pf.point;

  uint8_t *s0s = sbuf + kLambda * 2;
  uint8_t *s0l = s0s;
  uint8_t *s0r = s0s + kLambda;
  uint8_t *s1s = sbuf + kLambda * 4;
  uint8_t *s1l = s1s;
  uint8_t *s1r = s1s + kLambda;
  uint8_t t0l, t0r, t1l, t1r;

  for (int i = 0; i < p.alpha.bitlen; i++) {
    prg(s0s, 2 * kLambda, s0);
    prg(s1s, 2 * kLambda, s1);
    load_sst(s0s, &t0l, &t0r);
    load_sst(s1s, &t1l, &t1r);

    // Actually get MSB first
    uint8_t alpha_i = get_bit_lsb(p.alpha.bytes, p.alpha.bitlen - i - 1);
    uint8_t *cw = k.cws + i * kDpfCwLen;

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

  memcpy(k.cw_np1, p.beta, kLambda);
  set_bit_lsb(k.cw_np1, kLambda * 8 - 1, 0);
  group_neg(s0);
  group_add(k.cw_np1, s0);
  group_add(k.cw_np1, s1);
  if (t1) group_neg(k.cw_np1);
}

// | s | sl | sr |
// |   | ss      |
FSS_CUDA_HOST_DEVICE void dpf_eval(uint8_t *sbuf, uint8_t b, Key k, Bits x) {
  uint8_t *s = sbuf;
  uint8_t t;
  load_st(s, &t);
  t = b;

  uint8_t *ss = sbuf + kLambda;
  uint8_t *sl = ss;
  uint8_t *sr = ss + kLambda;
  uint8_t tl, tr;

  for (int i = 0; i < x.bitlen; i++) {
    const uint8_t *cw = k.cws + i * kDpfCwLen;
    const uint8_t *s_cw = cw;
    uint8_t tl_cw, tr_cw;
    get_cwt(cw, &tl_cw, &tr_cw);

    prg(ss, 2 * kLambda, s);
    load_sst(ss, &tl, &tr);
    if (t) {
      xor_bytes(sl, s_cw, kLambda);
      xor_bytes(sr, s_cw, kLambda);
      tl ^= tl_cw;
      tr ^= tr_cw;
    }

    // Actually get MSB first
    uint8_t x_i = get_bit_lsb(x.bytes, x.bitlen - i - 1);

    memcpy(s, x_i ? sr : sl, kLambda);
    t = x_i ? tr : tl;
  }

  if (t) group_add(s, k.cw_np1);
  if (b) group_neg(s);
}

#include <assert.h>
#include <stdlib.h>
#include <omp.h>

void dpf_eval_full_domain_node(int depth, uint8_t *sbufl, uint8_t *sbufr, uint8_t b, Key k) {
  uint8_t *s = sbufl;
  uint8_t t;
  load_st(s, &t);

  uint8_t *ss = (uint8_t *)malloc(kLambda * 2);
  assert(ss != NULL);
  uint8_t *sl = ss;
  uint8_t *sr = ss + kLambda;
  uint8_t tl, tr;

  const uint8_t *cw = k.cws + depth * kDpfCwLen;
  const uint8_t *s_cw = cw;
  uint8_t tl_cw, tr_cw;
  get_cwt(cw, &tl_cw, &tr_cw);

  prg(ss, 2 * kLambda, s);
  load_sst(ss, &tl, &tr);
  if (t) {
    xor_bytes(sl, s_cw, kLambda);
    xor_bytes(sr, s_cw, kLambda);
    tl ^= tl_cw;
    tr ^= tr_cw;
  }

  memcpy(sbufl, sl, kLambda);
  set_st(sbufl, tl);
  memcpy(sbufr, sr, kLambda);
  set_st(sbufr, tr);
  free(ss);
}

void dpf_eval_full_domain_leaf(uint8_t *sbuf, uint8_t b, Key k) {
  uint8_t *s = sbuf;
  uint8_t t;
  load_st(s, &t);

  if (t) group_add(s, k.cw_np1);
  if (b) group_neg(s);
}

void dpf_eval_full_domain_subtree(
  int depth, uint8_t *sbuf, size_t l, size_t r, uint8_t b, Key k, int x_bitlen, int par_depth) {
  assert(kLambda * (1ULL << (x_bitlen - depth)) == r - l);

  if (depth == x_bitlen) {
    dpf_eval_full_domain_leaf(sbuf + l, b, k);
    return;
  }

  size_t mid = (l + r) / 2;
  dpf_eval_full_domain_node(depth, sbuf + l, sbuf + mid, b, k);

  if (depth < par_depth) {
#pragma omp parallel
#pragma omp single
    {
#pragma omp task
      { dpf_eval_full_domain_subtree(depth + 1, sbuf, l, mid, b, k, x_bitlen, par_depth); }
#pragma omp task
      { dpf_eval_full_domain_subtree(depth + 1, sbuf, mid, r, b, k, x_bitlen, par_depth); }
#pragma omp taskwait
    }
  } else {
    dpf_eval_full_domain_subtree(depth + 1, sbuf, l, mid, b, k, x_bitlen, par_depth);
    dpf_eval_full_domain_subtree(depth + 1, sbuf, mid, r, b, k, x_bitlen, par_depth);
  }
}

void dpf_eval_full_domain(uint8_t *sbuf, uint8_t b, Key k, int x_bitlen) {
  uint8_t *s = sbuf;
  uint8_t t = b;
  set_st(s, t);

  int threads = omp_get_max_threads();
  int par_depth = 0;
  while ((1 << par_depth) <= threads) {
    par_depth++;
  }
  par_depth--;

  size_t sbuf_len = kLambda * (1ULL << x_bitlen);
  dpf_eval_full_domain_subtree(0, sbuf, 0, sbuf_len, b, k, x_bitlen, par_depth);
}
