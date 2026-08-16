// SPDX-License-Identifier: Apache-2.0
/**
 * @file eval_all_gpu.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief GPU full-domain evaluation (EvalAll) for DPF and HalfTree DPF.
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
#include <fss/dpf.cuh>
#include <fss/half_tree_dpf.cuh>

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
    if (threadIdx.x < (1 << (i + 1))) {
      int4 n = s_front[threadIdx.x >> 1];
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
    if (threadIdx.x < (1 << (i + 1))) {
      int4 n = s_front[threadIdx.x >> 1];
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
    if (threadIdx.x < (1 << (i + 1))) {
      int4 n = s_front[threadIdx.x >> 1];
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
    if (threadIdx.x < (1 << (i + 1))) {
      int4 n = s_front[threadIdx.x >> 1];
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

}  // namespace detail

/**
 * GPU full-domain evaluation of a HalfTree DPF key: ys[x] = Eval(b, s0, cws, ocw, x) for all x.
 *
 * @tparam z Frontier depth. 2^z threads each expand a subtree of 2^(in_bits - z) leaves.
 *   Defaults to min(in_bits, 16).
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
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
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
  constexpr int Z = (z < 0 ? (in_bits <= 16 ? in_bits : 16) : z);
  constexpr int B1 = (b1 < 0 ? 8 : b1);
  static_assert(Z <= in_bits && in_bits - Z <= 8);
  static_assert(B1 <= Z && (1 << (Z - B1)) == bs);
  detail::HalfTreeDpfEvalAllBatchKernel<in_bits, Z, B1, Group, Prg>
      <<<nkeys * (1 << B1), bs, 0, stream>>>(b, s0s, cws, ocws, ys, nkeys, dpf.hash_key, dpf.prg);
}

}  // namespace fss::gpu
