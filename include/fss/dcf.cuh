// SPDX-License-Identifier: Apache-2.0
/**
 * @file dcf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed comparison function (DCF).
 *
 * The scheme is from the paper, [_Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation_](https://eprint.iacr.org/2020/1392) (@ref dcf "1: the published version").
 *
 * ## Definitions
 *
 * **Comparison function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, and $b \in \sG_{out}$, a comparison function $f^<_{a, b}$ is a function that for any input $x$, the output $y$ has $y = b$ only when $x < a$, otherwise $y = 0$.
 *
 * **DCF**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, $b \in \sG_{out}$, and a security parameter $\lambda$, 2-party DCF is a scheme consisting of the methods:
 *
 * - Key generation: $Gen(1^\lambda, f^<_{a, b}) \rightarrow (k_0, k_1)$.
 * - Evaluation: $Eval(k_i, x) \rightarrow y_{i,x}$ for any $i \in \{0, 1\}$ and any $x \in \sG_{in}$.
 *
 * That satisfies:
 *
 * - Correctness: $y_{0, x} + y_{1, x} = b$ only when $x < a$, otherwise $y_{0, x} + y_{1, x} = 0$.
 * - Privacy: Neither $k_0$ nor $k_1$ reveals any information about $a$ or $b$.
 *   Formally speaking, there exists a probabilistic polynomial time (PPT) simulator $Sim$ that can generate output computationally indistinguishable from any strict subset of the keys output by $Gen$.
 *
 * Similarly, we have $f^>_{a, b}$ and DCF for it.
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
 * 1. Elette Boyle, Nishanth Chandran, Niv Gilboa, Divya Gupta, Yuval Ishai, Nishant Kumar, Mayank Rathee: Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation. EUROCRYPT (2) 2021: 871-900. <https://doi.org/10.1007/978-3-030-77886-6_30>. @anchor dcf
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
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>

namespace fss {

/**
 * Comparison predicate.
 *
 * For the input domain $\sG_{in} = \{0, 1\}^n$ as bits and $x, a \in \sG_{in}$, $x < a$ is defined as that the unsigned integer represented by $x$ is less than that represented by $a$, i.e., comparison starts from the most significant bit (MSB).
 */
enum class DcfPred {
  kLt, /**< $y = b$ when $x < a$ */
  kGt, /**< $y = b$ when $x > a$ */
};

/**
 * 2-party DCF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam pred See DcfPred.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint, DcfPred pred = DcfPred::kLt,
    int par_depth = -1>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Groupable<Group> && Prgable<Prg, 4>)
class Dcf {
public:
  Prg prg;

  /**
     * Correction word.
     *
     * ## Layout
     *
     * According to the paper, there are s, v, tl, tr to be stored.
     * v is converted to the clamped 16B to be stored.
     * tl, tr is stored at the clamped bit of s, v, respectively.
     */
  struct __align__(32) Cw {
    int4 s;
    int4 v;
  };
  // For only 1 and aligned memory access on GPU
  static_assert(sizeof(Cw) == 32);

  /**
     * A party's key: the correction words plus the party's initial seed.
     * Both parties share the same correction words and differ in the seed.
     */
  struct Key {
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
  // EvalAll() shape (mirroring Dpf): a scalar prefix of kBatchPrefix levels grows the
  // frontier to 8 nodes, each carrying the seed (t in LSB) beside the running group
  // value, detail::GenBatch() expands 8 parents per level (pipelined through Gen8()
  // when the PRG offers a batched entry), and a kBatchTail-level tail converts each
  // 8-subtree group of 8 << kBatchTail leaves in local memory. Domains below
  // kBatchPrefix + kBatchTail levels keep the recursive EvalTree() walk.
  static constexpr int kBatchPrefix = 3;
  static constexpr int kBatchTail = 3;

  // The v-channel correction has branchless masked fast paths for the XOR group and
  // the clamped 127-bit integer group; every other group runs the generic operators.
  static constexpr bool kGroupIsBytes = std::is_same_v<Group, group::Bytes>;
  static constexpr bool kGroupIsUint127 =
      std::is_same_v<Group, group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>>;

  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a, int4 b_buf) {
    int4 s0 = s0s[0];
    s0 = util::SetLsb(s0, false);
    int4 s1 = s0s[1];
    s1 = util::SetLsb(s1, false);
    bool t0 = false;
    bool t1 = true;
    Group v;
    b_buf = util::SetLsb(b_buf, false);

    for (int i = 0; i < in_bits; ++i) {
      auto [s0l, v0l_buf, s0r, v0r_buf] = prg.Gen(s0);
      auto [s1l, v1l_buf, s1r, v1r_buf] = prg.Gen(s1);

      bool t0l = util::GetLsb(s0l);
      s0l = util::SetLsb(s0l, false);
      v0l_buf = util::SetLsb(v0l_buf, false);
      auto v0l = Group::From(v0l_buf);
      bool t0r = util::GetLsb(s0r);
      s0r = util::SetLsb(s0r, false);
      v0r_buf = util::SetLsb(v0r_buf, false);
      auto v0r = Group::From(v0r_buf);
      bool t1l = util::GetLsb(s1l);
      s1l = util::SetLsb(s1l, false);
      v1l_buf = util::SetLsb(v1l_buf, false);
      auto v1l = Group::From(v1l_buf);
      bool t1r = util::GetLsb(s1r);
      s1r = util::SetLsb(s1r, false);
      v1r_buf = util::SetLsb(v1r_buf, false);
      auto v1r = Group::From(v1r_buf);

      bool a_bit = (a >> (in_bits - 1 - i)) & 1;

      int4 s_cw;
      if (!a_bit) s_cw = util::Xor(s0r, s1r);
      else s_cw = util::Xor(s0l, s1l);

      Group v_cw = (-v);
      if (!a_bit) {
        v_cw = v_cw + v1r + (-v0r);
        if constexpr (pred == DcfPred::kGt) v_cw = v_cw + Group::From(b_buf);
      } else {
        v_cw = v_cw + v1l + (-v0l);
        if constexpr (pred == DcfPred::kLt) v_cw = v_cw + Group::From(b_buf);
      }
      if (t1) v_cw = -v_cw;

      if (!a_bit) v = v + (-v1l) + v0l;
      else v = v + (-v1r) + v0r;
      if (t1) v = v + (-v_cw);
      else v = v + v_cw;

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
      int4 v_buf = v_cw.Into();
      v_buf = util::SetLsb(v_buf, tr_cw);
      cws[i] = {s_cw, v_buf};
    }

    auto v_cw_np1 = Group::From(s1) + (-Group::From(s0)) + (-v);
    if (t1) v_cw_np1 = -v_cw_np1;
    cws[in_bits] = {{0, 0, 0, 0}, v_cw_np1.Into()};
  }

  // Seed + correction-words form used by the internal recursion and tests.
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], In x) {
    int4 s = s0;
    s = util::SetLsb(s, false);
    Group v;
    bool t = b;

    for (int i = 0; i < in_bits; ++i) {
      auto cw = cws[i];

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
      }

      bool x_bit = (x >> (in_bits - 1 - i)) & 1;

      if (b) {
        if (!x_bit) v = v + (-vl);
        else v = v + (-vr);
        if (t) v = v + (-v_cw);
      } else {
        if (!x_bit) v = v + vl;
        else v = v + vr;
        if (t) v = v + v_cw;
      }

      if (!x_bit) {
        s = sl;
        t = tl;
      } else {
        s = sr;
        t = tr;
      }
    }

    int4 v_cw_np1_buf = cws[in_bits].v;
    assert((v_cw_np1_buf.w & 1) == 0);
    auto v_cw_np1 = Group::From(v_cw_np1_buf);

    if (b) {
      v = v + (-Group::From(s));
      if (t) v = v + (-v_cw_np1);
    } else {
      v = v + Group::From(s);
      if (t) v = v + v_cw_np1;
    }

    return v.Into();
  }

  void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[]) {
    assert(in_bits < sizeof(size_t) * 8);

    int par_depth_ = util::ResolveParDepth(par_depth);

    int4 st = s0;
    st = util::SetLsb(st, b);
    Group v;

    if constexpr (in_bits < kBatchPrefix + kBatchTail) {
      // Domains too small to fill an 8wide frontier keep the recursive tree walk.
#pragma omp parallel
#pragma omp single
      EvalTree(b, st, cws, ys, 0, size_t{1} << in_bits, 0, par_depth_, v);
      return;
    }

    int split_depth = par_depth_;
    if (split_depth > in_bits - kBatchPrefix - kBatchTail) split_depth = in_bits - kBatchPrefix - kBatchTail;
    if (split_depth > 0) {
      // Split at a frontier of 2 ** split_depth subtree roots instead of per-node tasks,
      // so the sequential batched walk stays the fast default path.
      size_t num_tasks = size_t{1} << split_depth;
      std::vector<int4> st_front(num_tasks);
      std::vector<Group> v_front(num_tasks);
      st_front[0] = st;
      v_front[0] = v;
      for (int lv = 0; lv < split_depth; ++lv) ExpandFrontier(b, st_front.data(), v_front.data(), 1 << lv, cws, lv);
      size_t span = (size_t{1} << in_bits) / num_tasks;
#pragma omp parallel for
      for (size_t task = 0; task < num_tasks; ++task) {
        EvalBatched8(b, st_front[task], v_front[task], cws, ys, task * span, split_depth);
      }
    } else {
      EvalBatched8(b, st, v, cws, ys, 0, 0);
    }
  }

  void EvalTree(bool b, int4 st, const Cw cws[], int4 ys[], size_t l, size_t r, int i, int par_depth_, Group v) {
    bool t = util::GetLsb(st);
    int4 s = st;
    s = util::SetLsb(s, false);

    if (i == in_bits) {
      int4 v_cw_np1_buf = cws[in_bits].v;
      assert((v_cw_np1_buf.w & 1) == 0);
      auto term = Group::From(s);
      if (t) term = term + Group::From(v_cw_np1_buf);
      if (b) term = -term;
      v = v + term;
      assert(l + 1 == r);
      ys[l] = v.Into();
      return;
    }

    Cw cw = cws[i];
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

    int4 stl = sl;
    stl = util::SetLsb(stl, tl);
    int4 str = sr;
    str = util::SetLsb(str, tr);

    size_t mid = (l + r) / 2;

    if (i < par_depth_) {
#pragma omp task
      EvalTree(b, stl, cws, ys, l, mid, i + 1, par_depth_, vl);
#pragma omp task
      EvalTree(b, str, cws, ys, mid, r, i + 1, par_depth_, vr);
#pragma omp taskwait
    } else {
      EvalTree(b, stl, cws, ys, l, mid, i + 1, par_depth_, vl);
      EvalTree(b, str, cws, ys, mid, r, i + 1, par_depth_, vr);
    }
  }

  // Expands w scalar frontier nodes (t in LSB, v the running group value) by one level
  // with cws[lv]. The children overwrite the frontier in place: iteration is in reverse
  // so the writes at 2j, 2j+1 never clobber nodes still to be read.
  void ExpandFrontier(bool b, int4 st_front[], Group v_front[], int w, const Cw cws[], int lv) {
    int4 s_cw = util::SetLsb(cws[lv].s, false);
    bool tl_cw = util::GetLsb(cws[lv].s);
    bool tr_cw = util::GetLsb(cws[lv].v);
    auto v_cw = Group::From(util::SetLsb(cws[lv].v, false));

    for (int j = w; j-- > 0;) {
      bool t = util::GetLsb(st_front[j]);
      int4 s = util::SetLsb(st_front[j], false);
      // The parent value must be read before the children overwrite the frontier:
      // for j = 0 the left child lands exactly on the parent's slot.
      Group v_parent = v_front[j];

      auto [sl, vl_buf, sr, vr_buf] = prg.Gen(s);

      int m = -static_cast<int>(t);
      st_front[2 * j] = CorrectChild(sl, s_cw, m, m & static_cast<int>(tl_cw));
      st_front[2 * j + 1] = CorrectChild(sr, s_cw, m, m & static_cast<int>(tr_cw));
      v_front[2 * j] = CorrectVal(vl_buf, v_cw, t, b, v_parent);
      v_front[2 * j + 1] = CorrectVal(vr_buf, v_cw, t, b, v_parent);
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

  // Applies the v-channel correction to one child value: the PRG word becomes the child
  // increment (v_cw added when the parent control bit t is set, negated for party 1),
  // which then adds into the parent's running value v. Bit-equal with the operator
  // sequence in EvalTree().
  static Group CorrectVal(int4 base_buf, Group v_cw, bool t, bool b, Group v) {
    if constexpr (kGroupIsBytes) {
      // The XOR group: negation is the identity, so the correction is three XORs with
      // the cw masked by the parent control bit.
      int m = -static_cast<int>(t);
      int4 base = util::SetLsb(base_buf, false);
      int4 res;
      res.x = v.val.x ^ base.x ^ (v_cw.val.x & m);
      res.y = v.val.y ^ base.y ^ (v_cw.val.y & m);
      res.z = v.val.z ^ base.z ^ (v_cw.val.z & m);
      res.w = v.val.w ^ base.w ^ (v_cw.val.w & m);
      Group out;
      out.val = res;
      return out;
    } else if constexpr (kGroupIsUint127) {
      // The clamped 127-bit integer group: mod-2^127 addition and negation in branchless
      // masked forms. Operands are < 2^127, so a uint128 sum loses at most the carry-out
      // at bit 127 and masking it off is the exact reduction; the party-1 subtraction
      // wraps the borrow the same way. Bit-equal with the generic operator sequence.
      constexpr __uint128_t kMask127 = (static_cast<__uint128_t>(1) << 127) - 1;
      __uint128_t m = t ? ~static_cast<__uint128_t>(0) : 0;
      __uint128_t x = (Group::From(util::SetLsb(base_buf, false)).val + (v_cw.val & m)) & kMask127;
      __uint128_t res;
      if (b) res = (v.val - x) & kMask127;
      else res = (v.val + x) & kMask127;
      Group out;
      out.val = res;
      return out;
    } else {
      auto base = Group::From(util::SetLsb(base_buf, false));
      if (t) base = base + v_cw;
      if (b) base = -base;
      return v + base;
    }
  }

  // Full-domain evaluation of the subtree rooted at (st, v) (t in LSB) at level i through
  // an 8wide batched frontier: a kBatchPrefix-level scalar prefix, batched expansions, and
  // a kBatchTail-level tail carrying (st, v) pairs. Requires
  // in_bits - i >= kBatchPrefix + kBatchTail.
  void EvalBatched8(bool b, int4 st, Group v, const Cw cws[], int4 ys[], size_t l, int i) {
    int4 st_front[8];
    Group v_front[8];
    st_front[0] = st;
    v_front[0] = v;
    for (int lv = i; lv < i + kBatchPrefix; ++lv) ExpandFrontier(b, st_front, v_front, 1 << (lv - i), cws, lv);
    EvalBatched8Group(b, st_front, v_front, cws, ys, l, i + kBatchPrefix);
  }

  // Evaluates a group of 8 consecutive frontier nodes (t in LSB, v the running group
  // value) at level i, covering the leaves [l, l + 8 << (in_bits - i)) in order.
  // Requires in_bits - i >= kBatchTail.
  void EvalBatched8Group(
      bool b, const int4 (&st_front)[8], const Group (&v_front)[8], const Cw cws[], int4 ys[], size_t l, int i) {
    if (i == in_bits - kBatchTail) {
      // Expand the group to its 8 << kBatchTail leaves in local memory, then convert.
      int4 st_bufs[2][8 << kBatchTail];
      Group v_bufs[2][8 << kBatchTail];
      int4 *st_cur = st_bufs[0];
      int4 *st_next = st_bufs[1];
      Group *v_cur = v_bufs[0];
      Group *v_next = v_bufs[1];
      for (int j = 0; j < 8; ++j) {
        st_cur[j] = st_front[j];
        v_cur[j] = v_front[j];
      }
      int w = 8;
      for (int lv = i; lv < in_bits; ++lv) {
        int4 s_cw = util::SetLsb(cws[lv].s, false);
        int tl_cw = util::GetLsb(cws[lv].s);
        int tr_cw = util::GetLsb(cws[lv].v);
        auto v_cw = Group::From(util::SetLsb(cws[lv].v, false));
        for (int g = 0; g < w / 8; ++g) {
          cuda::std::array<int4, 8> seeds;
          int ts[8];
          for (int j = 0; j < 8; ++j) {
            ts[j] = static_cast<int>(util::GetLsb(st_cur[8 * g + j]));
            seeds[j] = util::SetLsb(st_cur[8 * g + j], false);
          }
          auto out = detail::GenBatch<4>(prg, seeds);
          for (int j = 0; j < 8; ++j) {
            int m = -ts[j];
            st_next[16 * g + 2 * j] = CorrectChild(out[j][0], s_cw, m, ts[j] & tl_cw);
            st_next[16 * g + 2 * j + 1] = CorrectChild(out[j][2], s_cw, m, ts[j] & tr_cw);
            v_next[16 * g + 2 * j] = CorrectVal(out[j][1], v_cw, ts[j] != 0, b, v_cur[8 * g + j]);
            v_next[16 * g + 2 * j + 1] = CorrectVal(out[j][3], v_cw, ts[j] != 0, b, v_cur[8 * g + j]);
          }
        }
        int4 *st_swap = st_cur;
        st_cur = st_next;
        st_next = st_swap;
        Group *v_swap = v_cur;
        v_cur = v_next;
        v_next = v_swap;
        w *= 2;
      }

      int4 v_cw_np1_buf = cws[in_bits].v;
      assert((v_cw_np1_buf.w & 1) == 0);
      auto v_cw_np1 = Group::From(v_cw_np1_buf);
      for (int j = 0; j < w; ++j) {
        bool t = util::GetLsb(st_cur[j]);
        int4 s = util::SetLsb(st_cur[j], false);
        auto y = CorrectVal(s, v_cw_np1, t, b, v_cur[j]);
        ys[l + j] = y.Into();
      }
      return;
    }

    int4 s_cw = util::SetLsb(cws[i].s, false);
    int tl_cw = util::GetLsb(cws[i].s);
    int tr_cw = util::GetLsb(cws[i].v);
    auto v_cw = Group::From(util::SetLsb(cws[i].v, false));

    cuda::std::array<int4, 8> seeds;
    int ts[8];
    for (int j = 0; j < 8; ++j) {
      ts[j] = static_cast<int>(util::GetLsb(st_front[j]));
      seeds[j] = util::SetLsb(st_front[j], false);
    }
    auto out = detail::GenBatch<4>(prg, seeds);
    // The 16 children interleave in leaf order: node j's children land at 2j, 2j+1 of
    // the next level. De-interleave them into two consecutive groups of 8.
    int4 st_low[8];
    int4 st_high[8];
    Group v_low[8];
    Group v_high[8];
    for (int j = 0; j < 4; ++j) {
      int m = -ts[j];
      st_low[2 * j] = CorrectChild(out[j][0], s_cw, m, ts[j] & tl_cw);
      st_low[2 * j + 1] = CorrectChild(out[j][2], s_cw, m, ts[j] & tr_cw);
      v_low[2 * j] = CorrectVal(out[j][1], v_cw, ts[j] != 0, b, v_front[j]);
      v_low[2 * j + 1] = CorrectVal(out[j][3], v_cw, ts[j] != 0, b, v_front[j]);
      int m4 = -ts[j + 4];
      st_high[2 * j] = CorrectChild(out[j + 4][0], s_cw, m4, ts[j + 4] & tl_cw);
      st_high[2 * j + 1] = CorrectChild(out[j + 4][2], s_cw, m4, ts[j + 4] & tr_cw);
      v_high[2 * j] = CorrectVal(out[j + 4][1], v_cw, ts[j + 4] != 0, b, v_front[j + 4]);
      v_high[2 * j + 1] = CorrectVal(out[j + 4][3], v_cw, ts[j + 4] != 0, b, v_front[j + 4]);
    }

    size_t half = size_t{8} << (in_bits - i - 1);
    EvalBatched8Group(b, st_low, v_low, cws, ys, l, i + 1);
    EvalBatched8Group(b, st_high, v_high, cws, ys, l + half, i + 1);
  }
};

}  // namespace fss
