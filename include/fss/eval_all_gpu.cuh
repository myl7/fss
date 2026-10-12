// SPDX-License-Identifier: Apache-2.0
/**
 * @file eval_all_gpu.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief GPU full-domain evaluation (EvalAll) for DPF, HalfTree DPF, packed HalfTree DPF,
 * DCF, VDPF (with host-side Prove), and Grotto DCF (Preprocess and EvalAll).
 *
 * The tree-expansion strategy follows the hybrid approach of GPU-DPF
 * (facebookresearch/GPU-DPF, dpf_hybrid.cu, Z=128): expand the tree
 * toward a frontier of 2^z nodes, then expand each frontier node's
 * subtree down to the leaves. Single kernel launch.
 *
 * Structure (no redundant expansion):
 *  - Phase 0: each block walks the b1 top levels to its own subtree
 *    root (2^b1 blocks, b1 serial steps per block — negligible).
 *  - Phase 1: the block breadth-parallel-expands its subtree from
 *    depth b1 to depth z in shared memory (one __syncthreads per
 *    level, every node expanded exactly once).
 *  - Phase 2: each thread DFS-expands its own depth-z subtree and
 *    converts the last level into leaf outputs.
 *
 * This keeps the total PRG work at ~2^(n+1) calls (the tree size)
 * instead of the ~n * N of point evals, and avoids the redundant
 * per-thread path re-expansion of a naive breadth-to-z / DFS split.
 *
 * ## References
 *
 * 1. Facebookresearch GPU-DPF: https://github.com/facebookresearch/GPU-DPF
 * 2. EzPC GPU-MPC: https://github.com/mpc-msri/EzPC (see doc/bench_third_parties.md)
 */

#pragma once
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <fss/dcf.cuh>
#include <fss/dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/vdpf.cuh>

namespace fss::gpu {

namespace detail {

__device__ constexpr int4 kZero4 = {0, 0, 0, 0};

// --- HalfTree DPF ---

template <typename Prg>
__device__ __forceinline__ void HtExpandNode(int4 n, int4 cw_s, int4 hash_key, Prg prg, int4 &left, int4 &right) {
  bool t = util::GetLsb(n);
  int4 h = prg.Gen(util::Xor(hash_key, n))[0];
  left = util::Xor(h, t ? cw_s : kZero4);
  right = util::Xor(left, n);
}

// One step of a HalfTree path walk: only the chosen child is computed.
template <typename Prg>
__device__ __forceinline__ int4 HtPathStep(int4 n, bool x_bit, int4 cw_s, int4 hash_key, Prg prg) {
  bool t = util::GetLsb(n);
  int4 h = prg.Gen(util::Xor(hash_key, n))[0];
  return util::Xor(util::Xor(h, x_bit ? n : kZero4), t ? cw_s : kZero4);
}

template <int in_bits, int z, int b1, typename Group, typename Prg>
__global__ void HalfTreeDpfEvalAllKernel(bool b, int4 s0,
    const typename HalfTreeDpf<in_bits, Group, Prg, uint>::Cw *cws, int4 ocw, int4 *ys, int4 hash_key, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 n = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (blockIdx.x >> (b1 - 1 - i)) & 1;
      n = HtPathStep(n, x_bit, cws[i].s, hash_key, prg);
    }
    s_front[0] = n;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      HtExpandNode(n, cws[b1 + i].s, hash_key, prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z (levels z..in_bits-2, normal rule),
  // then convert the last level into leaf outputs.
  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits - 1; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 n = nodes[k];
      bool t = util::GetLsb(n);
      int4 h = prg.Gen(util::Xor(hash_key, n))[0];
      int4 left = util::Xor(h, t ? cws[lev].s : kZero4);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = util::Xor(left, n);
    }
    count *= 2;
  }

  int4 hcw = util::SetLsb(cws[in_bits - 1].s, false);
  bool lcw_0 = util::GetLsb(cws[in_bits - 1].s);
  bool lcw_1 = cws[in_bits - 1].extra;
  auto ocw_group = Group::From(ocw);

  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 n = nodes[k];
    bool t = util::GetLsb(n);

    int4 h0 = prg.Gen(util::Xor(hash_key, util::SetLsb(n, false)))[0];
    int4 h1 = prg.Gen(util::Xor(hash_key, util::SetLsb(n, true)))[0];

    int4 high0 = util::SetLsb(h0, false);
    bool low0 = util::GetLsb(h0);
    int4 high1 = util::SetLsb(h1, false);
    bool low1 = util::GetLsb(h1);

    if (t) {
      high0 = util::Xor(high0, hcw);
      low0 = low0 ^ lcw_0;
      high1 = util::Xor(high1, hcw);
      low1 = low1 ^ lcw_1;
    }

    auto y0 = Group::From(high0);
    if (low0) y0 = y0 + ocw_group;
    if (b) y0 = -y0;
    auto y1 = Group::From(high1);
    if (low1) y1 = y1 + ocw_group;
    if (b) y1 = -y1;

    ys[out_base + 2 * k] = y0.Into();
    ys[out_base + 2 * k + 1] = y1.Into();
  }
}

// --- DPF ---

template <typename Cw, typename Prg>
__device__ __forceinline__ void DpfExpandNode(int4 st, const Cw &cw, Prg prg, int4 &left, int4 &right) {
  bool t = util::GetLsb(st);
  int4 s = util::SetLsb(st, false);
  int4 s_cw = cw.s;
  bool tl_cw = util::GetLsb(s_cw);
  s_cw = util::SetLsb(s_cw, false);
  bool tr_cw = cw.tr;

  auto [sl, sr] = prg.Gen(s);
  bool tl = util::GetLsb(sl);
  sl = util::SetLsb(sl, false);
  bool tr = util::GetLsb(sr);
  sr = util::SetLsb(sr, false);

  if (t) {
    sl = util::Xor(sl, s_cw);
    sr = util::Xor(sr, s_cw);
    tl = tl ^ tl_cw;
    tr = tr ^ tr_cw;
  }

  left = util::SetLsb(sl, tl);
  right = util::SetLsb(sr, tr);
}

// One step of a DPF path walk: only the chosen child is computed.
template <typename Cw, typename Prg>
__device__ __forceinline__ int4 DpfPathStep(int4 st, const Cw &cw, Prg prg, bool x_bit) {
  bool t = util::GetLsb(st);
  int4 s = util::SetLsb(st, false);
  int4 s_cw = cw.s;
  bool tl_cw = util::GetLsb(s_cw);
  s_cw = util::SetLsb(s_cw, false);
  bool tr_cw = cw.tr;

  auto [sl, sr] = prg.Gen(s);
  bool tl = util::GetLsb(sl);
  sl = util::SetLsb(sl, false);
  bool tr = util::GetLsb(sr);
  sr = util::SetLsb(sr, false);

  if (t) {
    sl = util::Xor(sl, s_cw);
    sr = util::Xor(sr, s_cw);
    tl = tl ^ tl_cw;
    tr = tr ^ tr_cw;
  }

  // Pick the child without computing both full nodes.
  return x_bit ? util::SetLsb(sr, tr) : util::SetLsb(sl, tl);
}

template <int in_bits, int z, int b1, typename Group, typename Prg>
__global__ void DpfEvalAllKernel(
    bool b, int4 s0, const typename Dpf<in_bits, Group, Prg, uint>::Cw *cws, int4 *ys, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (blockIdx.x >> (b1 - 1 - i)) & 1;
      st = DpfPathStep(st, cws[i], prg, x_bit);
    }
    s_front[0] = st;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      DpfExpandNode(n, cws[b1 + i], prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z, then convert to leaves.
  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      DpfExpandNode(nodes[k], cws[lev], prg, left, right);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = right;
    }
    count *= 2;
  }

  int4 v_cw_np1 = cws[in_bits].s;
  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 st_cur = nodes[k];
    bool t_cur = util::GetLsb(st_cur);
    int4 s_cur = util::SetLsb(st_cur, false);
    auto y = Group::From(s_cur);
    if (t_cur) y = y + Group::From(v_cw_np1);
    if (b) y = -y;
    ys[out_base + k] = y.Into();
  }
}

// --- Batched variants: nkeys keys evaluated in one launch. ---
// cws is flat, key-major: key k's words start at cws + k * stride.
// Each key's 2^in_bits outputs land at ys + k * (1 << in_bits).
// The grid covers nkeys * 2^b1 blocks; block (key, bid) walks the
// same subtree as bid in the single-key kernel, but for key.

template <int in_bits, int z, int b1, typename Group, typename Prg>
__global__ void DpfEvalAllBatchKernel(bool b, const int4 *s0s,
    const typename Dpf<in_bits, Group, Prg, uint>::Cw *cws, int4 *ys, int nkeys, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  constexpr int kBlocks = 1 << b1;
  const int key = blockIdx.x / kBlocks;
  const int bid = blockIdx.x - key * kBlocks;
  const int tid = bid * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;
  const auto *kcws = cws + key * (in_bits + 1);
  int4 *kys = ys + key * (1 << in_bits);

  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0s[key], b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (bid >> (b1 - 1 - i)) & 1;
      st = DpfPathStep(st, kcws[i], prg, x_bit);
    }
    s_front[0] = st;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      DpfExpandNode(n, kcws[b1 + i], prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      DpfExpandNode(nodes[k], kcws[lev], prg, left, right);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = right;
    }
    count *= 2;
  }

  int4 v_cw_np1 = kcws[in_bits].s;
  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 st_cur = nodes[k];
    bool t_cur = util::GetLsb(st_cur);
    int4 s_cur = util::SetLsb(st_cur, false);
    auto y = Group::From(s_cur);
    if (t_cur) y = y + Group::From(v_cw_np1);
    if (b) y = -y;
    kys[out_base + k] = y.Into();
  }
}

template <int in_bits, int z, int b1, typename Group, typename Prg>
__global__ void HalfTreeDpfEvalAllBatchKernel(bool b, const int4 *s0s,
    const typename HalfTreeDpf<in_bits, Group, Prg, uint>::Cw *cws, const int4 *ocws, int4 *ys,
    int nkeys, int4 hash_key, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  constexpr int kBlocks = 1 << b1;
  const int key = blockIdx.x / kBlocks;
  const int bid = blockIdx.x - key * kBlocks;
  const int tid = bid * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;
  const auto *kcws = cws + key * in_bits;
  int4 *kys = ys + key * (1 << in_bits);

  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 n = util::SetLsb(s0s[key], b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (bid >> (b1 - 1 - i)) & 1;
      n = HtPathStep(n, x_bit, kcws[i].s, hash_key, prg);
    }
    s_front[0] = n;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      HtExpandNode(n, kcws[b1 + i].s, hash_key, prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits - 1; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 n = nodes[k];
      bool t = util::GetLsb(n);
      int4 h = prg.Gen(util::Xor(hash_key, n))[0];
      int4 left = util::Xor(h, t ? kcws[lev].s : kZero4);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = util::Xor(left, n);
    }
    count *= 2;
  }

  int4 hcw = util::SetLsb(kcws[in_bits - 1].s, false);
  bool lcw_0 = util::GetLsb(kcws[in_bits - 1].s);
  bool lcw_1 = kcws[in_bits - 1].extra;
  auto ocw_group = Group::From(ocws[key]);

  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 n = nodes[k];
    bool t = util::GetLsb(n);

    int4 h0 = prg.Gen(util::Xor(hash_key, util::SetLsb(n, false)))[0];
    int4 h1 = prg.Gen(util::Xor(hash_key, util::SetLsb(n, true)))[0];

    int4 high0 = util::SetLsb(h0, false);
    bool low0 = util::GetLsb(h0);
    int4 high1 = util::SetLsb(h1, false);
    bool low1 = util::GetLsb(h1);

    if (t) {
      high0 = util::Xor(high0, hcw);
      low0 = low0 ^ lcw_0;
      high1 = util::Xor(high1, hcw);
      low1 = low1 ^ lcw_1;
    }

    auto y0 = Group::From(high0);
    if (low0) y0 = y0 + ocw_group;
    if (b) y0 = -y0;
    auto y1 = Group::From(high1);
    if (low1) y1 = y1 + ocw_group;
    if (b) y1 = -y1;

    kys[out_base + 2 * k] = y0.Into();
    kys[out_base + 2 * k + 1] = y1.Into();
  }
}

// --- PackedHalfTree DPF ---

// The kernels mirror the HalfTree frontier kernels with two differences: the
// expanded depth is kDepth = in_bits - log2(kLanes) (the last log2(kLanes)
// levels never run, each level-kDepth node IS one packed output block), and the
// leaf conversion is a single domain-separated hash: leaf = PRG(hash_key ^ node)
// ^ (t ? fcw : 0), see PackedHalfTreeDpf::ConvertLeaf. There is no Group, no
// ocw, and no party negation of outputs since packed shares XOR-reconstruct.

template <int in_bits, int out_bits, int z, int b1, typename Prg>
__global__ void PackedHalfTreeDpfEvalAllKernel(bool b, int4 s0,
    const typename PackedHalfTreeDpf<in_bits, out_bits, Prg, uint>::Cw *cws, int4 fcw, int4 *ys, int4 hash_key,
    Prg prg) {
  constexpr int kDepth = PackedHalfTreeDpf<in_bits, out_bits, Prg, uint>::kDepth;
  constexpr int D = kDepth - z;
  constexpr int kSub = z - b1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 n = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (blockIdx.x >> (b1 - 1 - i)) & 1;
      n = HtPathStep(n, x_bit, cws[i].s, hash_key, prg);
    }
    s_front[0] = n;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      HtExpandNode(n, cws[b1 + i].s, hash_key, prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z (levels z..kDepth-1).
  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < kDepth; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 n = nodes[k];
      bool t = util::GetLsb(n);
      int4 h = prg.Gen(util::Xor(hash_key, n))[0];
      int4 left = util::Xor(h, t ? cws[lev].s : kZero4);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = util::Xor(left, n);
    }
    count *= 2;
  }

  // Leaf conversion: one level-kDepth node per packed output block.
  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 n = nodes[k];
    int4 leaf = prg.Gen(util::Xor(hash_key, n))[0];
    if (util::GetLsb(n)) leaf = util::Xor(leaf, fcw);
    ys[out_base + k] = leaf;
  }
}

template <int in_bits, int out_bits, int z, int b1, typename Prg>
__global__ void PackedHalfTreeDpfEvalAllBatchKernel(bool b, const int4 *s0s,
    const typename PackedHalfTreeDpf<in_bits, out_bits, Prg, uint>::Cw *cws, const int4 *fcws, int4 *ys, int nkeys,
    int4 hash_key, Prg prg) {
  constexpr int kDepth = PackedHalfTreeDpf<in_bits, out_bits, Prg, uint>::kDepth;
  constexpr int D = kDepth - z;
  constexpr int kSub = z - b1;
  constexpr int kBlocks = 1 << b1;
  const int key = blockIdx.x / kBlocks;
  const int bid = blockIdx.x - key * kBlocks;
  const int tid = bid * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;
  const auto *kcws = cws + key * kDepth;
  int4 *kys = ys + key * (1 << kDepth);
  int4 fcw = fcws[key];

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 n = util::SetLsb(s0s[key], b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (bid >> (b1 - 1 - i)) & 1;
      n = HtPathStep(n, x_bit, kcws[i].s, hash_key, prg);
    }
    s_front[0] = n;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      HtExpandNode(n, kcws[b1 + i].s, hash_key, prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z (levels z..kDepth-1).
  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < kDepth; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 n = nodes[k];
      bool t = util::GetLsb(n);
      int4 h = prg.Gen(util::Xor(hash_key, n))[0];
      int4 left = util::Xor(h, t ? kcws[lev].s : kZero4);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = util::Xor(left, n);
    }
    count *= 2;
  }

  // Leaf conversion: one level-kDepth node per packed output block.
  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 n = nodes[k];
    int4 leaf = prg.Gen(util::Xor(hash_key, n))[0];
    if (util::GetLsb(n)) leaf = util::Xor(leaf, fcw);
    kys[out_base + k] = leaf;
  }
}

// --- DCF ---

// A DCF tree node carries (st, v): st is the seed with t in the LSB, v is the
// running group value of the path, mirroring Dcf::EvalTree.
template <typename Cw, typename Group, typename Prg>
__device__ __forceinline__ void DcfExpandNode(int4 st, Group v, bool b, const Cw &cw, Prg prg,
    int4 &left, Group &v_left, int4 &right, Group &v_right) {
  bool t = util::GetLsb(st);
  int4 s = util::SetLsb(st, false);
  int4 s_cw = cw.s;
  bool tl_cw = util::GetLsb(s_cw);
  s_cw = util::SetLsb(s_cw, false);
  int4 v_cw_buf = cw.v;
  bool tr_cw = util::GetLsb(v_cw_buf);
  v_cw_buf = util::SetLsb(v_cw_buf, false);
  auto v_cw = Group::From(v_cw_buf);

  auto [sl, vl_buf, sr, vr_buf] = prg.Gen(s);
  bool tl = util::GetLsb(sl);
  sl = util::SetLsb(sl, false);
  vl_buf = util::SetLsb(vl_buf, false);
  auto vl = Group::From(vl_buf);
  bool tr = util::GetLsb(sr);
  sr = util::SetLsb(sr, false);
  vr_buf = util::SetLsb(vr_buf, false);
  auto vr = Group::From(vr_buf);

  if (t) {
    sl = util::Xor(sl, s_cw);
    sr = util::Xor(sr, s_cw);
    tl = tl ^ tl_cw;
    tr = tr ^ tr_cw;
    vl = vl + v_cw;
    vr = vr + v_cw;
  }
  if (b) {
    vl = -vl;
    vr = -vr;
  }
  vl = vl + v;
  vr = vr + v;

  left = util::SetLsb(sl, tl);
  right = util::SetLsb(sr, tr);
  v_left = vl;
  v_right = vr;
}

// One step of a DCF path walk: only the chosen child's (st, v) is kept.
template <typename Cw, typename Group, typename Prg>
__device__ __forceinline__ void DcfPathStep(int4 st, Group v, bool b, const Cw &cw, Prg prg, bool x_bit,
    int4 &next_st, Group &next_v) {
  int4 left, right;
  Group v_left, v_right;
  DcfExpandNode(st, v, b, cw, prg, left, v_left, right, v_right);
  if (x_bit) {
    next_st = right;
    next_v = v_right;
  } else {
    next_st = left;
    next_v = v_left;
  }
}

template <int in_bits, int z, int b1, typename Group, typename Prg>
__global__ void DcfEvalAllKernel(
    bool b, int4 s0, const typename Dcf<in_bits, Group, Prg, uint>::Cw *cws, int4 *ys, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  __shared__ Group v_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
    Group v;
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (blockIdx.x >> (b1 - 1 - i)) & 1;
      DcfPathStep(st, v, b, cws[i], prg, x_bit, st, v);
    }
    s_front[0] = st;
    v_front[0] = v;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    Group nv;
    if (active) {
      n = s_front[threadIdx.x >> 1];
      nv = v_front[threadIdx.x >> 1];
    }
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      Group v_left, v_right;
      DcfExpandNode(n, nv, b, cws[b1 + i], prg, left, v_left, right, v_right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
      v_front[threadIdx.x] = (threadIdx.x & 1) ? v_right : v_left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z, then convert to leaves.
  int4 nodes[1 << D];
  Group vs[1 << D];
  nodes[0] = s_front[threadIdx.x];
  vs[0] = v_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      Group v_left, v_right;
      DcfExpandNode(nodes[k], vs[k], b, cws[lev], prg, left, v_left, right, v_right);
      nodes[2 * k] = left;
      vs[2 * k] = v_left;
      nodes[2 * k + 1] = right;
      vs[2 * k + 1] = v_right;
    }
    count *= 2;
  }

  auto v_cw_np1 = Group::From(cws[in_bits].v);
  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 st_cur = nodes[k];
    bool t_cur = util::GetLsb(st_cur);
    int4 s_cur = util::SetLsb(st_cur, false);
    auto term = Group::From(s_cur);
    if (t_cur) term = term + v_cw_np1;
    if (b) term = -term;
    auto y = vs[k] + term;
    ys[out_base + k] = y.Into();
  }
}

// Batched variant: nkeys keys evaluated in one launch. cws is flat, key-major
// with stride (in_bits + 1) per key (the final cw's v part is used); each
// key's 2^in_bits outputs land at ys + k * (1 << in_bits).
template <int in_bits, int z, int b1, typename Group, typename Prg>
__global__ void DcfEvalAllBatchKernel(bool b, const int4 *s0s,
    const typename Dcf<in_bits, Group, Prg, uint>::Cw *cws, int4 *ys, int nkeys, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  constexpr int kBlocks = 1 << b1;
  const int key = blockIdx.x / kBlocks;
  const int bid = blockIdx.x - key * kBlocks;
  const int tid = bid * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;
  const auto *kcws = cws + key * (in_bits + 1);
  int4 *kys = ys + key * (1 << in_bits);

  __shared__ int4 s_front[1 << kSub];
  __shared__ Group v_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0s[key], b);
    Group v;
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (bid >> (b1 - 1 - i)) & 1;
      DcfPathStep(st, v, b, kcws[i], prg, x_bit, st, v);
    }
    s_front[0] = st;
    v_front[0] = v;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    Group nv;
    if (active) {
      n = s_front[threadIdx.x >> 1];
      nv = v_front[threadIdx.x >> 1];
    }
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      Group v_left, v_right;
      DcfExpandNode(n, nv, b, kcws[b1 + i], prg, left, v_left, right, v_right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
      v_front[threadIdx.x] = (threadIdx.x & 1) ? v_right : v_left;
    }
    __syncthreads();
  }

  int4 nodes[1 << D];
  Group vs[1 << D];
  nodes[0] = s_front[threadIdx.x];
  vs[0] = v_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      Group v_left, v_right;
      DcfExpandNode(nodes[k], vs[k], b, kcws[lev], prg, left, v_left, right, v_right);
      nodes[2 * k] = left;
      vs[2 * k] = v_left;
      nodes[2 * k + 1] = right;
      vs[2 * k + 1] = v_right;
    }
    count *= 2;
  }

  auto v_cw_np1 = Group::From(kcws[in_bits].v);
  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 st_cur = nodes[k];
    bool t_cur = util::GetLsb(st_cur);
    int4 s_cur = util::SetLsb(st_cur, false);
    auto term = Group::From(s_cur);
    if (t_cur) term = term + v_cw_np1;
    if (b) term = -term;
    auto y = vs[k] + term;
    kys[out_base + k] = y.Into();
  }
}

// --- Grotto DCF ---

// Frontier expansion of the DPF tree that writes only the per-leaf control
// bit: t[leaf] = GetLsb(leaf node), the rule of GrottoDcf::ExpandTreeRec.
// Same shape as DpfEvalAllKernel: Phase 0 root walk, Phase 1 shared-mem
// breadth expansion, Phase 2 per-thread DFS.
template <int in_bits, int z, int b1, typename Cw, typename Prg>
__global__ void GrottoDcfExpandLeafBitsKernel(bool b, int4 s0, const Cw *cws, bool *t, Prg prg) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (blockIdx.x >> (b1 - 1 - i)) & 1;
      st = DpfPathStep(st, cws[i], prg, x_bit);
    }
    s_front[0] = st;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      DpfExpandNode(n, cws[b1 + i], prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z, then write leaf control bits.
  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      DpfExpandNode(nodes[k], cws[lev], prg, left, right);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = right;
    }
    count *= 2;
  }

  const int out_base = tid << D;
#pragma unroll
  for (int k = 0; k < count; ++k) {
    t[out_base + k] = util::GetLsb(nodes[k]);
  }
}

// One level of the bottom-up parity build of GrottoDcf::Preprocess: thread j
// sets p[idx] = p[2*idx+1] ^ p[2*idx+2] for its node of the level.
// nodes = 2^l is the level width; node j of level l has tree index nodes-1+j.
__global__ void GrottoDcfParityLevelKernel(bool *p, int nodes) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nodes) return;
  int idx = nodes - 1 + j;
  p[idx] = p[2 * idx + 1] ^ p[2 * idx + 2];
}

// Top of the parity tree in one block: all levels of at most bs nodes each,
// built top-down in launch order but computed children-level-first with one
// __syncthreads() between levels, so no cross-block dependency remains.
template <int top_level>
__global__ void GrottoDcfParityTopKernel(bool *p) {
#pragma unroll
  for (int l = top_level; l >= 0; --l) {
    int nodes = 1 << l;
    if (threadIdx.x < nodes) {
      int idx = nodes - 1 + threadIdx.x;
      p[idx] = p[2 * idx + 1] ^ p[2 * idx + 2];
    }
    __syncthreads();
  }
}

// XOR scan functor for cub::DeviceScan over the 1-byte control-bit domain.
struct GrottoXorScanOp {
  __device__ __forceinline__ unsigned char operator()(unsigned char lhs, unsigned char rhs) const {
    return static_cast<unsigned char>(lhs ^ rhs);
  }
};

constexpr int ConstLog2(int val) {
  int r = 0;
  while ((1 << (r + 1)) <= val) ++r;
  return r;
}

// --- VDPF ---
//
// Frontier split as DpfEvalAllKernel: Phase 0 root walk by thread 0, Phase 1
// shared-mem breadth expansion, Phase 2 per-thread DFS. The VDPF Cw has the
// same layout as the DPF Cw (s with tl_cw in the clamped bit + bool tr in the
// 32B slot), so the walk reuses DpfPathStep/DpfExpandNode.
//
// Phase 2 computes each leaf's output share and corrected verification hash
// pi_tilde_j exactly as Vdpf::EvalAllMerkle phase 2, then folds the leaves
// into a binary Merkle tree with the same hash the chain uses. The paper's
// chain accumulation is inherently sequential; the fold is not: each thread
// streams its 2^D leaves into a thread-local subtree (a stack of D partial
// nodes), the block folds its threads' subtree roots in shared memory, and
// VdpfMerkleRootsKernel folds the block roots. No per-point data leaves the
// device. See doc/vdpf_merkle_fold_security.md.
template <int in_bits, int z, int b1, typename Group, typename Prg, typename XorHash, typename Hash, typename In,
    typename Cw>
__global__ void VdpfEvalAllMerkleKernel(bool b, int4 s0, const Cw *cws, cuda::std::array<int4, 4> cs, int4 ocw,
    int4 *ys, cuda::std::array<int4, 2> *roots, Prg prg, XorHash xor_hash, Hash hash) {
  constexpr int D = in_bits - z;
  constexpr int kSub = z - b1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (1 << z)) return;

  // Phase 0: block root walk (top b1 levels) by thread 0.
  __shared__ int4 s_front[1 << kSub];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < b1; ++i) {
      bool x_bit = (blockIdx.x >> (b1 - 1 - i)) & 1;
      st = DpfPathStep(st, cws[i], prg, x_bit);
    }
    s_front[0] = st;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels b1..z.
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = kZero4;
    if (active) n = s_front[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      DpfExpandNode(n, cws[b1 + i], prg, left, right);
      s_front[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth z, convert each leaf to the output
  // share and the corrected verification hash, and fold the leaf hashes into
  // the thread's Merkle subtree on the fly.
  int4 nodes[1 << D];
  nodes[0] = s_front[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = z; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      DpfExpandNode(nodes[k], cws[lev], prg, left, right);
      nodes[2 * k] = left;
      nodes[2 * k + 1] = right;
    }
    count *= 2;
  }

  auto ocw_group = Group::From(ocw);
  const int out_base = tid << D;
  cuda::std::array<int4, 2> stack[D + 1];
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 st_cur = nodes[k];
    bool t_cur = util::GetLsb(st_cur);
    int4 s_cur = util::SetLsb(st_cur, false);

    // Output share
    auto g = Group::From(s_cur);
    if (t_cur) g = g + ocw_group;
    if (b) g = -g;
    ys[out_base + k] = g.Into();

    // Leaf: the corrected verification hash of the point
    int4 x_buf = util::Pack(static_cast<In>(out_base + k));
    auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, s_cur});
    if (t_cur) {
      pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs));
    }
    int4 leaf_msg[4] = {pi_tilde[0], pi_tilde[1], pi_tilde[2], pi_tilde[3]};
    cuda::std::array<int4, 2> node = hash.Hash(cuda::std::span<const int4, 4>(leaf_msg));

    // Stream the leaf into the thread's subtree: combine with the pending
    // sibling stack entries while the leaf index has trailing one bits.
    int rest = k;
    int lvl = 0;
#pragma unroll
    while (rest != 0 && (rest & 1) == 1) {
      int4 fold_msg[4] = {stack[lvl][0], stack[lvl][1], node[0], node[1]};
      node = hash.Hash(cuda::std::span<const int4, 4>(fold_msg));
      rest >>= 1;
      ++lvl;
    }
    stack[lvl] = node;
  }

  // Block fold of the threads' subtree roots, in thread order: adjacent pairs
  // at each stride, slot idx combining with idx + stride.
  __shared__ cuda::std::array<int4, 2> s_node[1 << kSub];
  s_node[threadIdx.x] = stack[D];
  __syncthreads();
#pragma unroll
  for (int stride = 1; stride < (1 << kSub); stride <<= 1) {
    int idx = 2 * stride * threadIdx.x;
    if (idx < (1 << kSub)) {
      int4 fold_msg[4] = {s_node[idx][0], s_node[idx][1], s_node[idx + stride][0], s_node[idx + stride][1]};
      s_node[idx] = hash.Hash(cuda::std::span<const int4, 4>(fold_msg));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) roots[blockIdx.x] = s_node[0];
}

// Folds the per-block subtree roots into the final 64B proof. The root count
// is (1 << z) / bs = 2^b1, at most one block wide; only the first nroots
// shared slots participate, so the rest stay unwritten and unread.
template <int bs, typename Hash>
__global__ void VdpfMerkleRootsKernel(const cuda::std::array<int4, 2> *roots, int nroots, cuda::std::array<int4, 4> cs,
    cuda::std::array<int4, 4> *pi, Hash hash) {
  __shared__ cuda::std::array<int4, 2> s_node[bs];
  if (threadIdx.x < nroots) s_node[threadIdx.x] = roots[threadIdx.x];
  __syncthreads();
  for (int stride = 1; stride < nroots; stride <<= 1) {
    int idx = 2 * stride * threadIdx.x;
    if (idx < nroots) {
      int4 fold_msg[4] = {s_node[idx][0], s_node[idx][1], s_node[idx + stride][0], s_node[idx + stride][1]};
      s_node[idx] = hash.Hash(cuda::std::span<const int4, 4>(fold_msg));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *pi = cuda::std::array<int4, 4>{s_node[0][0], s_node[0][1], cs[2], cs[3]};
  }
}

}  // namespace detail

/**
 * GPU full-domain evaluation of a HalfTree DPF key: ys[x] = Eval(b, s0, cws, ocw, x) for all x.
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(in_bits - z) leaves.
 *   Defaults to min(in_bits - 1, 16).
 * @tparam b1 Block-root depth. Each block walks b1 levels to its own subtree root,
 *   then expands levels b1..z breadth-parallel. Must satisfy 2^(z - b1) == bs.
 *   Defaults to 8.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param ocw Output correction word returned by Gen().
 * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
 * @param dpf A HalfTreeDpf instance (holds the PRG and hash key). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename In>
void HalfTreeDpfEvalAllGpu(bool b, int4 s0, const typename HalfTreeDpf<in_bits, Group, Prg, In>::Cw *cws, int4 ocw,
    int4 *ys, const HalfTreeDpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits - 1 <= 16 ? in_bits - 1 : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z < in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  constexpr int kBlocks = (1 << Z) / bs;
  detail::HalfTreeDpfEvalAllKernel<in_bits, Z, B1, Group, Prg>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, ocw, ys, dpf.hash_key, dpf.prg);
}

/**
 * GPU full-domain evaluation of a DPF key: ys[x] = Eval(b, s0, cws, x) for all x.
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(in_bits - z) leaves.
 *   Defaults to min(in_bits, 16).
 * @tparam b1 Block-root depth. Each block walks b1 levels to its own subtree root,
 *   then expands levels b1..z breadth-parallel. Must satisfy 2^(z - b1) == bs.
 *   Defaults to 8.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
 * @param dpf A Dpf instance (holds the PRG). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename In>
void DpfEvalAllGpu(bool b, int4 s0, const typename Dpf<in_bits, Group, Prg, In>::Cw *cws, int4 *ys,
    const Dpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  constexpr int kBlocks = (1 << Z) / bs;
  detail::DpfEvalAllKernel<in_bits, Z, B1, Group, Prg>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, ys, dpf.prg);
}

/**
 * GPU full-domain evaluation of nkeys DPF keys in one launch: per-key time
 * drops toward the PRG bound as the grid fills the device (tail waves and
 * launch overhead amortized). This matches the workload that evaluates many
 * keys, e.g. a Gen() loop followed by EvalAll per key.
 *
 * Layouts: s0s has nkeys entries (party b's seed each); cws is flat, key-major,
 * with stride (in_bits + 1) per key; ys is flat with stride (1 << in_bits) per
 * key. All sizes must be pre-allocated by the caller.
 *
 * @tparam z Frontier depth, see DpfEvalAllGpu.
 * @tparam b1 Block-root depth, see DpfEvalAllGpu.
 * @tparam bs Block size, see DpfEvalAllGpu.
 * @param b Party index. False for 0 and true for 1.
 * @param s0s nkeys seeds, one per key.
 * @param cws Flat correction words, key-major, stride (in_bits + 1).
 * @param nkeys Number of keys to evaluate.
 * @param ys Output array, flat, stride (1 << in_bits) per key.
 * @param dpf A Dpf instance (holds the PRG). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename In>
void DpfEvalAllGpuBatch(bool b, const int4 *s0s, const typename Dpf<in_bits, Group, Prg, In>::Cw *cws, int nkeys,
    int4 *ys, const Dpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  detail::DpfEvalAllBatchKernel<in_bits, Z, B1, Group, Prg>
      <<<nkeys * (1 << B1), bs, 0, stream>>>(b, s0s, cws, ys, nkeys, dpf.prg);
}

/**
 * GPU full-domain evaluation of nkeys HalfTree DPF keys in one launch, see
 * DpfEvalAllGpuBatch. cws stride is in_bits per key; ocws has nkeys entries.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename In>
void HalfTreeDpfEvalAllGpuBatch(bool b, const int4 *s0s, const typename HalfTreeDpf<in_bits, Group, Prg, In>::Cw *cws,
    const int4 *ocws, int nkeys, int4 *ys, const HalfTreeDpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits - 1 <= 16 ? in_bits - 1 : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z < in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  detail::HalfTreeDpfEvalAllBatchKernel<in_bits, Z, B1, Group, Prg>
      <<<nkeys * (1 << B1), bs, 0, stream>>>(b, s0s, cws, ocws, ys, nkeys, dpf.hash_key, dpf.prg);
}

/**
 * GPU full-domain evaluation of a PackedHalfTree DPF key: block j of ys covers inputs
 * [j * kLanes, (j + 1) * kLanes) for all j, matching EvalAll().
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(kDepth - z) output
 *   blocks, where kDepth = in_bits - log2(kLanes). Defaults to min(kDepth, 16).
 * @tparam b1 Block-root depth. Each block walks b1 levels to its own subtree root,
 *   then expands levels b1..z breadth-parallel. Must satisfy 2^(z - b1) == bs.
 *   Defaults to 8.
 * @tparam bs Block size. Defaults to 256.
 * @param b Party index. False for 0 and true for 1. It only sets the initial control
 *   bit: packed outputs are XOR shares, so unlike HalfTree there is no party negation.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param fcw Final correction word returned by Gen().
 * @param ys Pre-allocated output array. Its size must be at least 2 ** kDepth.
 * @param dpf A PackedHalfTreeDpf instance (holds the PRG and hash key). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, int out_bits, typename Prg, typename In>
void PackedHalfTreeDpfEvalAllGpu(bool b, int4 s0,
    const typename PackedHalfTreeDpf<in_bits, out_bits, Prg, In>::Cw *cws, int4 fcw, int4 *ys,
    const PackedHalfTreeDpf<in_bits, out_bits, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int kDepth = PackedHalfTreeDpf<in_bits, out_bits, Prg, In>::kDepth;
  constexpr int Z = (z < 0 ? (kDepth <= 16 ? kDepth : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= kDepth && kDepth - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  constexpr int kBlocks = (1 << Z) / bs;
  detail::PackedHalfTreeDpfEvalAllKernel<in_bits, out_bits, Z, B1, Prg>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, fcw, ys, dpf.hash_key, dpf.prg);
}

/**
 * GPU full-domain evaluation of nkeys PackedHalfTree DPF keys in one launch, see
 * PackedHalfTreeDpfEvalAllGpu and DpfEvalAllGpuBatch. cws stride is kDepth per key;
 * fcws has nkeys entries; ys stride is (1 << kDepth) per key.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, int out_bits, typename Prg, typename In>
void PackedHalfTreeDpfEvalAllGpuBatch(bool b, const int4 *s0s,
    const typename PackedHalfTreeDpf<in_bits, out_bits, Prg, In>::Cw *cws, const int4 *fcws, int nkeys, int4 *ys,
    const PackedHalfTreeDpf<in_bits, out_bits, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int kDepth = PackedHalfTreeDpf<in_bits, out_bits, Prg, In>::kDepth;
  constexpr int Z = (z < 0 ? (kDepth <= 16 ? kDepth : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= kDepth && kDepth - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  detail::PackedHalfTreeDpfEvalAllBatchKernel<in_bits, out_bits, Z, B1, Prg>
      <<<nkeys * (1 << B1), bs, 0, stream>>>(b, s0s, cws, fcws, ys, nkeys, dpf.hash_key, dpf.prg);
}

/**
 * GPU full-domain evaluation of a DCF key: ys[x] = Eval(b, s0, cws, x) for all x.
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(in_bits - z) leaves.
 *   Defaults to min(in_bits, 16).
 * @tparam b1 Block-root depth. Each block walks b1 levels to its own subtree root,
 *   then expands levels b1..z breadth-parallel. Must satisfy 2^(z - b1) == bs.
 *   Defaults to 8.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
 * @param dcf A Dcf instance (holds the PRG). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename In>
void DcfEvalAllGpu(bool b, int4 s0, const typename Dcf<in_bits, Group, Prg, In>::Cw *cws, int4 *ys,
    const Dcf<in_bits, Group, Prg, In> &dcf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  constexpr int kBlocks = (1 << Z) / bs;
  detail::DcfEvalAllKernel<in_bits, Z, B1, Group, Prg>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, ys, dcf.prg);
}

/**
 * GPU full-domain evaluation of nkeys DCF keys in one launch: per-key time
 * drops toward the PRG bound as the grid fills the device (tail waves and
 * launch overhead amortized). This matches the workload that evaluates many
 * keys, e.g. a Gen() loop followed by EvalAll per key.
 *
 * Layouts: s0s has nkeys entries (party b's seed each); cws is flat, key-major,
 * with stride (in_bits + 1) per key; ys is flat with stride (1 << in_bits) per
 * key. All sizes must be pre-allocated by the caller.
 *
 * @tparam z Frontier depth, see DcfEvalAllGpu.
 * @tparam b1 Block-root depth, see DcfEvalAllGpu.
 * @tparam bs Block size, see DcfEvalAllGpu.
 * @param b Party index. False for 0 and true for 1.
 * @param s0s nkeys seeds, one per key.
 * @param cws Flat correction words, key-major, stride (in_bits + 1).
 * @param nkeys Number of keys to evaluate.
 * @param ys Output array, flat, stride (1 << in_bits) per key.
 * @param dcf A Dcf instance (holds the PRG). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename In>
void DcfEvalAllGpuBatch(bool b, const int4 *s0s, const typename Dcf<in_bits, Group, Prg, In>::Cw *cws, int nkeys,
    int4 *ys, const Dcf<in_bits, Group, Prg, In> &dcf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  detail::DcfEvalAllBatchKernel<in_bits, Z, B1, Group, Prg>
      <<<nkeys * (1 << B1), bs, 0, stream>>>(b, s0s, cws, ys, nkeys, dcf.prg);
}

/**
 * GPU preprocess of a Grotto DCF key: expands the DPF tree on the GPU and
 * builds the parity segment tree of GrottoDcf::Preprocess in device memory.
 * d_p[0..2N-2] then supports GrottoDcf::Eval directly on the GPU, with
 * N = 2 ** in_bits and the leaf control bits at d_p[N-1 .. 2N-2].
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(in_bits - z) leaves.
 *   Defaults to min(in_bits, 16).
 * @tparam b1 Block-root depth. Each block walks b1 levels to its own subtree root,
 *   then expands levels b1..z breadth-parallel. Must satisfy 2^(z - b1) == bs.
 *   Defaults to 8.
 * @tparam bs Block size, see z and b1. Defaults to 256.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param d_p Pre-allocated device output array. Its size must be at least 2 * 2 ** in_bits - 1.
 * @param dcf A GrottoDcf instance (holds the PRG). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Prg, typename In>
void GrottoDcfPreprocessGpu(bool b, int4 s0, const typename GrottoDcf<in_bits, Prg, In>::Cw *cws, bool *d_p,
    const GrottoDcf<in_bits, Prg, In> &dcf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  constexpr int kBlocks = (1 << Z) / bs;
  constexpr size_t N = size_t{1} << in_bits;
  detail::GrottoDcfExpandLeafBitsKernel<in_bits, Z, B1, typename GrottoDcf<in_bits, Prg, In>::Cw, Prg>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, d_p + (N - 1), dcf.prg);
  // Bottom-up parity build. Levels wider than one block get one kernel each;
  // the levels of at most bs nodes at the top go through one single-block
  // kernel, which keeps the small-launch count at in_bits - log2(bs) + 1.
  constexpr int kTopL = detail::ConstLog2(bs);
  constexpr int kTop = kTopL < in_bits - 1 ? kTopL : in_bits - 1;
  for (int l = in_bits - 1; l > kTop; --l) {
    int nodes = 1 << l;
    detail::GrottoDcfParityLevelKernel<<<(nodes + bs - 1) / bs, bs, 0, stream>>>(d_p, nodes);
  }
  detail::GrottoDcfParityTopKernel<kTop><<<1, bs, 0, stream>>>(d_p);
}

/**
 * GPU full-domain evaluation of a Grotto DCF key: d_ys[x] is party b's share
 * of 1[alpha <= x] for all x, as in GrottoDcf::EvalAll. Expands the DPF tree
 * on the GPU, then runs an inclusive prefix-XOR scan over the leaf control
 * bits in place.
 *
 * @tparam z Frontier depth, see GrottoDcfPreprocessGpu.
 * @tparam b1 Block-root depth, see GrottoDcfPreprocessGpu.
 * @tparam bs Block size, see GrottoDcfPreprocessGpu.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param d_ys Pre-allocated device output array. Its size must be at least 2 ** in_bits.
 * @param dcf A GrottoDcf instance (holds the PRG). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Prg, typename In>
void GrottoDcfEvalAllGpu(bool b, int4 s0, const typename GrottoDcf<in_bits, Prg, In>::Cw *cws, bool *d_ys,
    const GrottoDcf<in_bits, Prg, In> &dcf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  constexpr int kBlocks = (1 << Z) / bs;
  constexpr size_t N = size_t{1} << in_bits;
  detail::GrottoDcfExpandLeafBitsKernel<in_bits, Z, B1, typename GrottoDcf<in_bits, Prg, In>::Cw, Prg>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, d_ys, dcf.prg);
  // Host-side plain C++: scan the 1-byte control bits as unsigned char. The
  // CUB scratch is cached per host thread: a per-call cudaMallocAsync costs a
  // driver round trip on some setups when the pool cannot serve the block
  // from cache, which measured ~0.6ms of stream stall per call on a shared
  // A6000. Calls from one thread on different streams are only safe when
  // those streams are not run concurrently.
  auto *bits = reinterpret_cast<unsigned char *>(d_ys);
  detail::GrottoXorScanOp scan_op;
  size_t tmp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, tmp_bytes, bits, bits, scan_op, static_cast<int>(N), stream);
  thread_local void *d_tmp = nullptr;
  thread_local size_t tmp_cap = 0;
  if (tmp_cap < tmp_bytes) {
    if (d_tmp != nullptr) cudaFree(d_tmp);
    cudaMalloc(&d_tmp, tmp_bytes);
    tmp_cap = tmp_bytes;
  }
  cub::DeviceScan::InclusiveScan(d_tmp, tmp_bytes, bits, bits, scan_op, static_cast<int>(N), stream);
}

/**
 * GPU full-domain evaluation of a VDPF key with the Merkle-fold proof:
 * ys[x] is the output share of Eval(b, s0, cws, cs, ocw, x) and d_pi the 64B
 * proof, matching Vdpf::EvalAllMerkle bit for bit. One frontier kernel
 * computes the shares and folds each thread's leaves into a Merkle subtree; a
 * second single-block kernel folds the block roots into d_pi. The proof never
 * leaves the device. See doc/vdpf_merkle_fold_security.md for the security
 * analysis of the fold.
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(in_bits - z) leaves.
 *   Defaults to min(in_bits, 16).
 * @tparam b1 Block-root depth. Each block walks b1 levels to its own subtree root,
 *   then expands levels b1..z breadth-parallel. Must satisfy 2^(z - b1) == bs.
 *   Defaults to 8.
 * @tparam bs Block size. Defaults to 256.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Initial seed of the party.
 * @param cws Correction words returned by Gen().
 * @param cs Check seed returned by Gen(). Passed by value (64B) as a kernel param.
 * @param ocw Output correction word returned by Gen().
 * @param ys Pre-allocated device output array. Its size must be at least 2 ** in_bits.
 * @param d_pi Pre-allocated device output for the 64B proof.
 * @param vdpf A Vdpf instance (holds the PRG and hashes). Used on the host side.
 * @param stream CUDA stream to launch on.
 */
template <int z = -1, int b1 = 8, int bs = 256, int in_bits, typename Group, typename Prg, typename XorHash,
    typename Hash, typename In>
void VdpfEvalAllMerkleGpu(bool b, int4 s0, const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw *cws,
    cuda::std::array<int4, 4> cs, int4 ocw, int4 *ys, cuda::std::array<int4, 4> *d_pi,
    const Vdpf<in_bits, Group, Prg, XorHash, Hash, In> &vdpf, cudaStream_t stream = 0) {
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  // VdpfMerkleRootsKernel folds the block roots in one block, so they must
  // fit: (1 << Z) / bs <= bs.
  static_assert((1 << Z) <= bs * bs);
  constexpr int kBlocks = (1 << Z) / bs;
  // The per-block root buffer is tiny (kBlocks entries), but a per-call
  // cudaMallocAsync costs a driver round trip on some setups, so keep one
  // allocation per host thread. Calls from one thread on different streams
  // must not run concurrently.
  thread_local cuda::std::array<int4, 2> *d_roots = nullptr;
  if (d_roots == nullptr) cudaMalloc(&d_roots, sizeof(cuda::std::array<int4, 2>) * kBlocks);
  detail::VdpfEvalAllMerkleKernel<in_bits, Z, B1, Group, Prg, XorHash, Hash, In,
      typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw>
      <<<kBlocks, bs, 0, stream>>>(b, s0, cws, cs, ocw, ys, d_roots, vdpf.prg, vdpf.xor_hash, vdpf.hash);
  detail::VdpfMerkleRootsKernel<bs, Hash><<<1, bs, 0, stream>>>(d_roots, kBlocks, cs, d_pi, vdpf.hash);
}

}  // namespace fss::gpu
