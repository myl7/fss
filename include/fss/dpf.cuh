// SPDX-License-Identifier: Apache-2.0
/**
 * @file dpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed point function (DPF).
 *
 * The scheme is from the paper, [_Function Secret Sharing: Improvements and Extensions_](https://eprint.iacr.org/2018/707) (@ref dpf "1: the published version").
 *
 * ## Definitions
 *
 * **Point function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, and $b \in \sG_{out}$, a point function $f_{a, b}$ is a function that for any input $x$, the output $y$ has $y = b$ only when $x = a$, otherwise $y = 0$.
 *
 * **DPF**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, $b \in \sG_{out}$, and a security parameter $\lambda$, 2-party DPF is a scheme consisting of the methods:
 *
 * - Key generation: $Gen(1^\lambda, f_{a, b}) \rightarrow (k_0, k_1)$.
 * - Evaluation: $Eval(k_i, x) \rightarrow y_{i,x}$ for any $i \in \{0, 1\}$ and any $x \in \sG_{in}$.
 *
 * That satisfies:
 *
 * - Correctness: $y_{0, x} + y_{1, x} = b$ only when $x = a$, otherwise $y_{0, x} + y_{1, x} = 0$.
 * - Privacy: Neither $k_0$ nor $k_1$ reveals any information about $a$ or $b$.
 *   Formally speaking, there exists a probabilistic polynomial time (PPT) simulator $Sim$ that can generate output computationally indistinguishable from any strict subset of the keys output by $Gen$.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's LSB to 0, corresponding to $\lambda = 127$.
 * See Groupable for more details.
 *
 * We limit the max input domain bit size to 128.
 * This is enough for most applications and allows us to represent the input as an integer.
 *
 * ## References
 *
 * 1. Elette Boyle, Niv Gilboa, Yuval Ishai: Function Secret Sharing: Improvements and Extensions. CCS 2016: 1292-1303. <https://doi.org/10.1145/2976749.2978429>. @anchor dpf
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <omp.h>
#include <vector>
#include <cuda/std/array>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>

namespace fss {

template <int in_bits, typename Prg, typename In, int par_depth>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 && Prgable<Prg, 2>)
class GrottoDcf;

/**
 * 2-party DPF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint, int par_depth = -1>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Groupable<Group> && Prgable<Prg, 2>)
class Dpf {
public:
  Prg prg;

  /**
     * Correction word.
     *
     * ## Layout
     *
     * According to the paper, there are s, tl, tr to be stored.
     * tl is stored at the clamped bit of s.
     */
  struct __align__(32) Cw {
    int4 s;
    bool tr;
  };
  // For only 1 and aligned memory access on GPU
  static_assert(sizeof(Cw) == 32);

  /**
     * A party's key: the correction words plus the party's initial seed.
     * Both parties share the same correction words and differ in the seed.
     */
  struct Key {
    static constexpr int kParDepth = par_depth;
    Cw cws[in_bits + 1];
    int4 s0;
  };

  /**
     * Key generation method.
     *
     * @param k0, k1 Returned keys of the two parties.
     * @param s0s 2 initial seeds. Users can randomly sample them.
     * @param a $a$.
     * @param b_buf $b$. Will be clamped and converted to the group element.
     */
  __host__ __device__ void Gen(Key &k0, Key &k1, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf) {
    Gen(k0.cws, s0s.data(), a, b_buf);
    for (int i = 0; i <= in_bits; ++i) k1.cws[i] = k0.cws[i];
    k0.s0 = s0s[0];
    k1.s0 = s0s[1];
  }

  /**
     * Evaluation method.
     *
     * @param b Party index. False for 0 and true for 1. $i$.
     * @param key The party's key returned by Gen().
     * @param x Evaluated input. $x$.
     * @return Output share. $y_{i,x}$.
     */
  __host__ __device__ int4 Eval(bool b, const Key &key, In x) {
    return Eval(b, key.s0, key.cws, x);
  }

  /**
     * Full domain evaluation method.
     *
     * Evaluate the key on each input, i.e., 0b00...0 - 0b11...1.
     * Store the outputs sequentially.
     *
     * @param b Party index. False for 0 and true for 1.
     * @param key The party's key returned by Gen().
     * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
     *
     * Support parallel using OpenMP.
     * The task is divided to 2 ** par_depth parallel sub-tasks with the equal workloads.
     * par_depth = -1: use ceil(log(num of threads)).
     * par_depth = 0: no parallelism, i.e., sequential execution.
     */
  void EvalAll(bool b, const Key &key, cuda::std::span<int4> ys) {
    EvalAll(b, key.s0, key.cws, ys.data());
  }

private:
  // EvalAll() shape: a scalar prefix of kBatchPrefix levels grows the frontier to
  // 8 nodes, detail::GenBatch() expands 8 parents per level (pipelined through
  // Gen8() when the PRG offers a batched entry), and a kBatchTail-level tail
  // converts each 8-subtree group of 8 << kBatchTail leaves in local memory.
  static constexpr int kBatchPrefix = 3;
  static constexpr int kBatchTail = 3;

public:
  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a, int4 b_buf) {
    int4 s0 = s0s[0];
    s0 = util::SetLsb(s0, false);
    int4 s1 = s0s[1];
    s1 = util::SetLsb(s1, false);
    bool t0 = false;
    bool t1 = true;
    b_buf = util::SetLsb(b_buf, false);

    for (int i = 0; i < in_bits; ++i) {
      auto [s0l, s0r] = prg.Gen(s0);
      auto [s1l, s1r] = prg.Gen(s1);

      bool t0l = util::GetLsb(s0l);
      s0l = util::SetLsb(s0l, false);
      bool t0r = util::GetLsb(s0r);
      s0r = util::SetLsb(s0r, false);
      bool t1l = util::GetLsb(s1l);
      s1l = util::SetLsb(s1l, false);
      bool t1r = util::GetLsb(s1r);
      s1r = util::SetLsb(s1r, false);

      bool a_bit = (a >> (in_bits - 1 - i)) & 1;

      int4 s_cw;
      if (!a_bit) s_cw = util::Xor(s0r, s1r);
      else s_cw = util::Xor(s0l, s1l);

      bool tl_cw = t0l ^ t1l ^ a_bit ^ 1;
      bool tr_cw = t0r ^ t1r ^ a_bit;

      if (!a_bit) {
        s0 = s0l;
        if (t0) s0 = util::Xor(s0, s_cw);
        s1 = s1l;
        if (t1) s1 = util::Xor(s1, s_cw);

        if (t0) t0 = t0l ^ tl_cw;
        else t0 = t0l;
        if (t1) t1 = t1l ^ tl_cw;
        else t1 = t1l;
      } else {
        s0 = s0r;
        if (t0) s0 = util::Xor(s0, s_cw);
        s1 = s1r;
        if (t1) s1 = util::Xor(s1, s_cw);

        if (t0) t0 = t0r ^ tr_cw;
        else t0 = t0r;
        if (t1) t1 = t1r ^ tr_cw;
        else t1 = t1r;
      }

      s_cw = util::SetLsb(s_cw, tl_cw);
      // Write both 16B halves so the full 32B slot is covered by unmasked stores.
      // {s_cw, tr_cw} alone leaves the padding undefined: NVCC emits a masked
      // 256-bit store, and the partial-sector write forces a write-allocate
      // round trip per level on GPU (measured ~2x Gen time).
      int4 *cw_slot = reinterpret_cast<int4 *>(&cws[i]);
      cw_slot[0] = s_cw;
      cw_slot[1] = make_int4(tr_cw, 0, 0, 0);
    }

    auto v_cw_np1 = Group::From(b_buf) + (-Group::From(s0)) + Group::From(s1);
    if (t1) v_cw_np1 = -v_cw_np1;
    cws[in_bits] = {v_cw_np1.Into(), false};
  }

  // Seed + correction-words form used by the internal recursion and tests.
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], In x) {
    int4 st = Walk(b, s0, cws, x, [](bool, bool) {});
    bool t = util::GetLsb(st);
    int4 s = util::SetLsb(st, false);

    auto y = Group::From(s);
    int4 v_cw_np1 = cws[in_bits].s;
    assert((v_cw_np1.w & 1) == 0);
    if (t) y = y + Group::From(v_cw_np1);
    if (b) y = -y;

    return y.Into();
  }

  void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[]) {
    assert(in_bits < sizeof(size_t) * 8);

    if constexpr (in_bits < kBatchPrefix + kBatchTail) {
      // Domains too small to fill an 8-wide frontier evaluate per input; the
      // point walk is cheaper than the batched machinery at this size.
      constexpr size_t n = size_t{1} << in_bits;
      util::ParallelFor(par_depth, n, [&](size_t x) { ys[x] = Eval(b, s0, cws, static_cast<In>(x)); });
      return;
    }

    int4 st = s0;
    st = util::SetLsb(st, b);

    int par_depth_ = util::ResolveParDepth(par_depth, in_bits);

    int split_depth = par_depth_;
    if (split_depth > in_bits - kBatchPrefix - kBatchTail) split_depth = in_bits - kBatchPrefix - kBatchTail;
    if (split_depth > 0) {
      // Split at a frontier of 2 ** split_depth subtree roots instead of per-node tasks,
      // so the sequential batched walk stays the fast default path.
      size_t num_tasks = size_t{1} << split_depth;
      std::vector<int4> frontier = BuildParallelFrontier(st, cws, split_depth);
      size_t span = (size_t{1} << in_bits) / num_tasks;
      util::ParallelFor(par_depth_, num_tasks, [&](size_t task) {
        EvalBatched8(b, frontier[task], cws, ys, task * span, split_depth);
      });
    } else {
      EvalBatched8(b, st, cws, ys, 0, 0);
    }
  }

private:
  friend class GrottoDcf<in_bits, Prg, In, par_depth>;

  // Walks x's path from the root and returns the leaf seed with its control bit in the LSB.
  // visit(t, x_bit) sees the control bit of every expanded node and the branch x takes from it.
  template <typename Visit>
  __host__ __device__ int4 Walk(bool b, int4 s0, const Cw cws[], In x, Visit &&visit) {
    int4 s = s0;
    s = util::SetLsb(s, false);
    bool t = b;

    for (int i = 0; i < in_bits; ++i) {
      bool x_bit = (x >> (in_bits - 1 - i)) & 1;
      visit(t, x_bit);

      Cw cw = cws[i];
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

      if (!x_bit) {
        s = sl;
        t = tl;
      } else {
        s = sr;
        t = tr;
      }
    }

    return util::SetLsb(s, t);
  }

  // GCC 12.4.0 can discard these writes before the OpenMP walk after IPA
  // effect analysis. Keep this boundary opaque on GCC 12 releases before 12.5.
  // GCC PR115033 describes a related closure escape-analysis defect.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 12 && __GNUC_MINOR__ <= 4
  __attribute__((noipa))
#endif
  std::vector<int4> BuildParallelFrontier(int4 st, const Cw cws[], int split_depth) {
    std::vector<int4> frontier(size_t{1} << split_depth);
    frontier[0] = st;
    for (int lv = 0; lv < split_depth; ++lv) ExpandFrontier(frontier.data(), size_t{1} << lv, cws, lv);
    return frontier;
  }

  // Expands w scalar frontier nodes (t in LSB) by one level with cws[lv].
  // The children overwrite the frontier in place: iteration is in reverse so
  // the writes at 2j, 2j+1 never clobber nodes still to be read.
  void ExpandFrontier(int4 frontier[], size_t w, const Cw cws[], int lv) {
    int4 s_cw = util::SetLsb(cws[lv].s, false);
    bool tl_cw = util::GetLsb(cws[lv].s);
    bool tr_cw = cws[lv].tr;

    for (size_t j = w; j-- > 0;) {
      bool t = util::GetLsb(frontier[j]);
      int4 s = util::SetLsb(frontier[j], false);

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

      frontier[2 * j] = util::SetLsb(sl, tl);
      frontier[2 * j + 1] = util::SetLsb(sr, tr);
    }
  }

  // Applies the level correction to one expanded child branchlessly.
  // m is 0 or -1 from the parent control bit t, and t_cw is the cw control bit
  // masked by t. The cw seed is clamped, so the XOR leaves the child LSB free
  // for the packed control bit.
  static int4 CorrectChild(int4 child, int4 s_cw, int m, int t_cw) {
    int4 res = {child.x ^ (s_cw.x & m), child.y ^ (s_cw.y & m), child.z ^ (s_cw.z & m), child.w ^ (s_cw.w & m)};
    res.w = (res.w & ~1) | ((child.w ^ t_cw) & 1);
    return res;
  }

  // Full-domain evaluation of the subtree rooted at st (t in LSB) at level i through an
  // 8-wide batched frontier: a kBatchPrefix-level scalar prefix, batched expansions, and
  // a kBatchTail-level tail converted in local memory.
  // Requires in_bits - i >= kBatchPrefix + kBatchTail.
  void EvalBatched8(bool b, int4 st, const Cw cws[], int4 ys[], size_t l, int i) {
    int4 frontier[8];
    frontier[0] = st;
    for (int lv = i; lv < i + kBatchPrefix; ++lv) ExpandFrontier(frontier, size_t{1} << (lv - i), cws, lv);
    EvalBatched8Group(b, frontier, cws, ys, l, i + kBatchPrefix);
  }

  // Evaluates a group of 8 consecutive frontier nodes (t in LSB) at level i, covering the
  // leaves [l, l + 8 << (in_bits - i)) in order. Requires in_bits - i >= kBatchTail.
  void EvalBatched8Group(bool b, const int4 (&frontier)[8], const Cw cws[], int4 ys[], size_t l, int i) {
    if (i == in_bits - kBatchTail) {
      // Expand the group to its 8 << kBatchTail leaves in local memory, then convert.
      int4 bufs[2][8 << kBatchTail];
      int4 *cur = bufs[0];
      int4 *next = bufs[1];
      for (int j = 0; j < 8; ++j) cur[j] = frontier[j];
      int w = 8;
      for (int lv = i; lv < in_bits; ++lv) {
        int4 s_cw = util::SetLsb(cws[lv].s, false);
        int tl_cw = util::GetLsb(cws[lv].s);
        int tr_cw = cws[lv].tr;
        for (int g = 0; g < w / 8; ++g) {
          cuda::std::array<int4, 8> seeds;
          int ts[8];
          for (int j = 0; j < 8; ++j) {
            ts[j] = static_cast<int>(util::GetLsb(cur[8 * g + j]));
            seeds[j] = util::SetLsb(cur[8 * g + j], false);
          }
          auto out = detail::GenBatch<2>(prg, seeds);
          for (int j = 0; j < 8; ++j) {
            int m = -ts[j];
            next[16 * g + 2 * j] = CorrectChild(out[j][0], s_cw, m, ts[j] & tl_cw);
            next[16 * g + 2 * j + 1] = CorrectChild(out[j][1], s_cw, m, ts[j] & tr_cw);
          }
        }
        int4 *swap = cur;
        cur = next;
        next = swap;
        w *= 2;
      }

      int4 v_cw_np1 = cws[in_bits].s;
      assert((v_cw_np1.w & 1) == 0);
      Group v_cw = Group::From(v_cw_np1);
      for (int j = 0; j < w; ++j) {
        bool t = util::GetLsb(cur[j]);
        int4 s = util::SetLsb(cur[j], false);
        auto y = Group::From(s);
        if (t) y = y + v_cw;
        if (b) y = -y;
        ys[l + j] = y.Into();
      }
      return;
    }

    int4 s_cw = util::SetLsb(cws[i].s, false);
    int tl_cw = util::GetLsb(cws[i].s);
    int tr_cw = cws[i].tr;

    cuda::std::array<int4, 8> seeds;
    int ts[8];
    for (int j = 0; j < 8; ++j) {
      ts[j] = static_cast<int>(util::GetLsb(frontier[j]));
      seeds[j] = util::SetLsb(frontier[j], false);
    }
    auto out = detail::GenBatch<2>(prg, seeds);
    // The 16 children interleave in leaf order: node j's children land at 2j, 2j+1 of the
    // next level. De-interleave them into two consecutive groups of 8.
    int4 low[8];
    int4 high[8];
    for (int j = 0; j < 4; ++j) {
      int m = -ts[j];
      low[2 * j] = CorrectChild(out[j][0], s_cw, m, ts[j] & tl_cw);
      low[2 * j + 1] = CorrectChild(out[j][1], s_cw, m, ts[j] & tr_cw);
      int m4 = -ts[j + 4];
      high[2 * j] = CorrectChild(out[j + 4][0], s_cw, m4, ts[j + 4] & tl_cw);
      high[2 * j + 1] = CorrectChild(out[j + 4][1], s_cw, m4, ts[j + 4] & tr_cw);
    }

    size_t half = size_t{8} << (in_bits - i - 1);
    EvalBatched8Group(b, low, cws, ys, l, i + 1);
    EvalBatched8Group(b, high, cws, ys, l + half, i + 1);
  }
};

}  // namespace fss
