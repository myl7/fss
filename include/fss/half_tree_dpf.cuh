// SPDX-License-Identifier: Apache-2.0
/**
 * @file half_tree_dpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed point function (DPF) using the Half-Tree scheme.
 *
 * The scheme is from the paper, [_Half-Tree: Halving the Cost of Tree Expansion in COT and DPF_](https://eprint.iacr.org/2022/1431) (@ref half_tree_dpf "1: the published version").
 * It reduces the number of hash calls from 2N to 1.5N for full-domain evaluation.
 *
 * ## References
 *
 * 1. Xiaojie Guo, Kang Yang, Xiao Wang, Wenhao Zhang, Xiang Xie, Jiang Zhang, Zheli Liu: Half-Tree: Halving the Cost of Tree Expansion in COT and DPF. EUROCRYPT (1) 2023: 330-362. <https://doi.org/10.1007/978-3-031-30545-0_12>. <https://eprint.iacr.org/2022/1431>. @anchor half_tree_dpf
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

/**
 * 2-party DPF scheme using the Half-Tree construction.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=1 (CCR hash, 128->128 bits).
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint, int par_depth = -1>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Groupable<Group> && Prgable<Prg, 1>)
class HalfTreeDpf {
public:
  Prg prg;
  int4 hash_key;

  /**
   * Correction word.
   *
   * For levels 1..n-1, s is the CW and extra is unused (false).
   * For level n, s stores SetLsb(HCW, LCW_0) and extra stores LCW_1.
   */
  struct __align__(32) Cw {
    int4 s;
    bool extra;
  };
  static_assert(sizeof(Cw) == 32);

  /**
   * A party's key: the correction words, the output correction word, and the party's initial
   * seed. Both parties share the same correction words and the output correction word, and
   * differ in the seed.
   */
  struct Key {
    Cw cws[in_bits];
    int4 ocw;
    int4 s0;
  };

  /**
   * Key generation method.
   *
   * @param k0, k1 Returned keys of the two parties.
   * @param s0s 2 initial seeds. Users can randomly sample them.
   * @param a The special point alpha.
   * @param b_buf The nonzero output value beta. Will be clamped and converted to the group element.
   */
  __host__ __device__ void Gen(Key &k0, Key &k1, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf) {
    Gen(k0.cws, k0.ocw, s0s.data(), a, b_buf);
    for (int i = 0; i < in_bits; ++i) k1.cws[i] = k0.cws[i];
    k1.ocw = k0.ocw;
    k0.s0 = s0s[0];
    k1.s0 = s0s[1];
  }

  /**
   * Evaluation method.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param key The party's key returned by Gen().
   * @param x Evaluated input.
   * @return Output share.
   */
  __host__ __device__ int4 Eval(bool b, const Key &key, In x) {
    return Eval(b, key.s0, key.cws, key.ocw, x);
  }

  /**
   * Full domain evaluation method.
   *
   * Evaluate the key on each input, i.e., 0b00...0 - 0b11...1.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param key The party's key returned by Gen().
   * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
   *
   * Support parallel using OpenMP.
   */
  void EvalAll(bool b, const Key &key, cuda::std::span<int4> ys) {
    EvalAll(b, key.s0, key.cws, key.ocw, ys.data());
  }

private:
  // EvalAll() shape: a scalar prefix of kBatchPrefix levels grows the frontier
  // to 8 nodes and detail::GenBatch() expands 8 parents per level (pipelined
  // through Gen8() when the PRG offers a batched entry). The tail fuses the
  // last-level conversion: each group of 8 parents emits its 16 leaves
  // directly, so no backward in-place pass over ys is needed.
  static constexpr int kBatchPrefix = 3;

  __host__ __device__ void Gen(Cw cws[], int4 &ocw, const int4 s0s[2], In a, int4 b_buf) {
    b_buf = util::SetLsb(b_buf, false);

    // Initialize: node0 has t=0, node1 has t=1
    int4 node0 = util::SetLsb(s0s[0], false);
    int4 node1 = util::SetLsb(s0s[1], true);
    int4 delta = util::Xor(node0, node1);  // LSB = 0^1 = 1

    // Levels 1 to n-1 (index i = 0 to in_bits-2)
    for (int i = 0; i < in_bits - 1; ++i) {
      int4 h0 = prg.Gen(util::Xor(hash_key, node0))[0];
      int4 h1 = prg.Gen(util::Xor(hash_key, node1))[0];

      bool a_bit = (a >> (in_bits - 1 - i)) & 1;

      // CW = h0 ^ h1 ^ (!a_bit ? delta : 0)
      // When a_bit=0 (go left): non-alpha is right, CW = h0^h1^delta makes right0=right1
      // When a_bit=1 (go right): non-alpha is left, CW = h0^h1 makes left0=left1
      int4 cw = util::Xor(h0, h1);
      if (!a_bit) cw = util::Xor(cw, delta);

      cws[i] = {cw, false};

      bool t0 = util::GetLsb(node0);
      bool t1 = util::GetLsb(node1);

      // node_b = h_b ^ (a_bit ? node_b : 0) ^ (t_b ? cw : 0)
      int4 zero4 = {0, 0, 0, 0};
      int4 ab_mask0 = a_bit ? node0 : zero4;
      int4 ab_mask1 = a_bit ? node1 : zero4;
      int4 t0_mask = t0 ? cw : zero4;
      int4 t1_mask = t1 ? cw : zero4;

      node0 = util::Xor(util::Xor(h0, ab_mask0), t0_mask);
      node1 = util::Xor(util::Xor(h1, ab_mask1), t1_mask);

      // delta = node0 ^ node1 for next level
      delta = util::Xor(node0, node1);
    }

    // Level n (last level, index i = in_bits-1)
    {
      bool a_n = (a >> 0) & 1;  // last bit of alpha
      bool t0 = util::GetLsb(node0);
      bool t1 = util::GetLsb(node1);

      // Hash with sigma in {0, 1}
      int4 h0_0 = prg.Gen(util::Xor(hash_key, util::SetLsb(node0, false)))[0];
      int4 h0_1 = prg.Gen(util::Xor(hash_key, util::SetLsb(node0, true)))[0];
      int4 h1_0 = prg.Gen(util::Xor(hash_key, util::SetLsb(node1, false)))[0];
      int4 h1_1 = prg.Gen(util::Xor(hash_key, util::SetLsb(node1, true)))[0];

      // Extract high (s) and low (t) parts
      int4 high0_0 = util::SetLsb(h0_0, false);
      bool low0_0 = util::GetLsb(h0_0);
      int4 high0_1 = util::SetLsb(h0_1, false);
      bool low0_1 = util::GetLsb(h0_1);
      int4 high1_0 = util::SetLsb(h1_0, false);
      bool low1_0 = util::GetLsb(h1_0);
      int4 high1_1 = util::SetLsb(h1_1, false);
      bool low1_1 = util::GetLsb(h1_1);

      // HCW corrects the non-alpha direction so both parties converge.
      // HCW = high{!a_n}_0 ^ high{!a_n}_1
      int4 HCW;
      if (a_n) HCW = util::Xor(high0_0, high1_0);
      else HCW = util::Xor(high0_1, high1_1);

      // LCW ensures:
      //   Alpha direction (sigma=a_n): low0 ^ low1 = 1 (exactly one adds ocw)
      //   Non-alpha direction (sigma=!a_n): low0 ^ low1 = 0 (cancels)
      // LCW_0 = low{0}_0 ^ low{0}_1 ^ !a_n
      // LCW_1 = low{1}_0 ^ low{1}_1 ^ a_n
      bool LCW_0 = low0_0 ^ low1_0 ^ !a_n;
      bool LCW_1 = low0_1 ^ low1_1 ^ a_n;

      // Store CW_n. Both 16B halves written so the full 32B slot gets unmasked
      // stores: {s, LCW_1} alone leaves padding undefined, NVCC emits a masked
      // 256-bit store, and the partial-sector write forces a write-allocate
      // round trip on GPU (measured ~2x Gen time for the same pattern).
      int4 *cw_slot = reinterpret_cast<int4 *>(&cws[in_bits - 1]);
      cw_slot[0] = util::SetLsb(HCW, LCW_0);
      cw_slot[1] = make_int4(LCW_1, 0, 0, 0);

      // Compute leaf for each party
      // leaf_b = (a_n ? high{1}_b||low{1}_b : high{0}_b||low{0}_b)
      int4 leaf0, leaf1;
      if (a_n) {
        leaf0 = util::SetLsb(high0_1, low0_1);
        leaf1 = util::SetLsb(high1_1, low1_1);
      } else {
        leaf0 = util::SetLsb(high0_0, low0_0);
        leaf1 = util::SetLsb(high1_0, low1_0);
      }

      // Apply CW correction: if t_b: leaf_b ^= SetLsb(HCW, lcw_an)
      bool lcw_an = a_n ? LCW_1 : LCW_0;
      int4 leaf_cw = util::SetLsb(HCW, lcw_an);
      if (t0) leaf0 = util::Xor(leaf0, leaf_cw);
      if (t1) leaf1 = util::Xor(leaf1, leaf_cw);

      // Output CW: v_cw = Group::From(b_buf) + (-Group::From(SetLsb(leaf0,false))) + Group::From(SetLsb(leaf1,false))
      auto v_cw =
          Group::From(b_buf) + (-Group::From(util::SetLsb(leaf0, false))) + Group::From(util::SetLsb(leaf1, false));
      if (util::GetLsb(leaf1)) v_cw = -v_cw;
      ocw = v_cw.Into();
    }
  }

  // Seed + correction-words form used by the internal recursion and tests.
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], int4 ocw, In x) {
    int4 node = util::SetLsb(s0, b);

    // Levels 1 to n-1 (index i = 0 to in_bits-2)
    for (int i = 0; i < in_bits - 1; ++i) {
      bool x_bit = (x >> (in_bits - 1 - i)) & 1;
      bool t = util::GetLsb(node);

      int4 h = prg.Gen(util::Xor(hash_key, node))[0];

      int4 zero4 = {0, 0, 0, 0};
      int4 xb_mask = x_bit ? node : zero4;
      int4 t_mask = t ? cws[i].s : zero4;

      node = util::Xor(util::Xor(h, xb_mask), t_mask);
    }

    // Level n (last level)
    {
      bool x_n = (x >> 0) & 1;
      bool t = util::GetLsb(node);

      int4 h = prg.Gen(util::Xor(hash_key, util::SetLsb(node, x_n)))[0];

      // Unpack CW_n
      int4 hcw = util::SetLsb(cws[in_bits - 1].s, false);
      bool lcw_xn;
      if (x_n) lcw_xn = cws[in_bits - 1].extra;
      else lcw_xn = util::GetLsb(cws[in_bits - 1].s);

      int4 high = util::SetLsb(h, false);
      bool low = util::GetLsb(h);

      if (t) {
        high = util::Xor(high, hcw);
        low = low ^ lcw_xn;
      }

      auto y = Group::From(high);
      if (low) y = y + Group::From(ocw);
      if (b) y = -y;

      return y.Into();
    }
  }

  void EvalAll(bool b, int4 s0, const Cw cws[], int4 ocw, int4 ys[]) {
    assert(in_bits < sizeof(size_t) * 8);

    if constexpr (in_bits < kBatchPrefix + 1) {
      // Domains too small to fill an 8-wide frontier evaluate per input; the
      // point walk is cheaper than the batched machinery at this size.
      constexpr size_t n = size_t{1} << in_bits;
#pragma omp parallel for if (util::ResolveParDepth(par_depth) > 0)
      for (size_t x = 0; x < n; ++x) ys[x] = Eval(b, s0, cws, ocw, static_cast<In>(x));
      return;
    }

    int4 node = util::SetLsb(s0, b);

    int par_depth_ = util::ResolveParDepth(par_depth);

    int split_depth = par_depth_;
    if (split_depth > in_bits - kBatchPrefix - 1) split_depth = in_bits - kBatchPrefix - 1;
    if (split_depth > 0) {
      // Split at a frontier of 2 ** split_depth subtree roots instead of per-node tasks,
      // so the sequential batched walk stays the fast default path.
      size_t num_tasks = size_t{1} << split_depth;
      std::vector<int4> frontier(num_tasks);
      frontier[0] = node;
      for (int lv = 0; lv < split_depth; ++lv) ExpandFrontier(frontier.data(), 1 << lv, cws, lv);
      size_t span = size_t{1} << (in_bits - split_depth);
#pragma omp parallel for
      for (size_t task = 0; task < num_tasks; ++task) {
        EvalBatched8(b, frontier[task], cws, Group::From(ocw), ys, task * span, split_depth);
      }
    } else {
      EvalBatched8(b, node, cws, Group::From(ocw), ys, 0, 0);
    }
  }

  // Expands w scalar frontier nodes by one level with cws[lv].
  // The children overwrite the frontier in place: iteration is in reverse so
  // the writes at 2j, 2j+1 never clobber nodes still to be read.
  void ExpandFrontier(int4 frontier[], int w, const Cw cws[], int lv) {
    int4 cw = cws[lv].s;
    int4 zero4 = {0, 0, 0, 0};
    for (int j = w; j-- > 0;) {
      int4 node = frontier[j];
      int4 h = prg.Gen(util::Xor(hash_key, node))[0];
      int4 t_mask = util::GetLsb(node) ? cw : zero4;
      int4 left = util::Xor(h, t_mask);
      frontier[2 * j] = left;
      frontier[2 * j + 1] = util::Xor(left, node);
    }
  }

  // Branchless XOR of a correction word into 16B under a 0/-1 mask from the control bit.
  static int4 MaskXor(int4 val, int4 cw, int m) {
    return {val.x ^ (cw.x & m), val.y ^ (cw.y & m), val.z ^ (cw.z & m), val.w ^ (cw.w & m)};
  }

  // Full-domain evaluation of the subtree rooted at node at level i through an
  // 8-wide batched frontier: a kBatchPrefix-level scalar prefix, batched
  // expansions, and a fused tail that converts each group of 8 parents to its
  // 16 leaves. Requires in_bits - i >= kBatchPrefix + 1.
  void EvalBatched8(bool b, int4 node, const Cw cws[], Group ocw_group, int4 ys[], size_t l, int i) {
    int4 frontier[8];
    frontier[0] = node;
    for (int lv = i; lv < i + kBatchPrefix; ++lv) ExpandFrontier(frontier, 1 << (lv - i), cws, lv);
    EvalBatched8Group(b, frontier, cws, ocw_group, ys, l, i + kBatchPrefix);
  }

  // Evaluates a group of 8 consecutive frontier nodes at level i, covering the leaves
  // [l, l + 8 << (in_bits - i)) in order. Requires in_bits - i >= 1.
  void EvalBatched8Group(bool b, const int4 (&frontier)[8], const Cw cws[], Group ocw_group, int4 ys[], size_t l, int i) {
    if (i == in_bits - 1) {
      // 8 parents -> 16 leaves with two Gen8 sets (sigma = 0, 1).
      int4 hcw = util::SetLsb(cws[in_bits - 1].s, false);
      int lcw_0 = util::GetLsb(cws[in_bits - 1].s);
      int lcw_1 = cws[in_bits - 1].extra;

      cuda::std::array<int4, 8> seeds0;
      cuda::std::array<int4, 8> seeds1;
      int ts[8];
      for (int j = 0; j < 8; ++j) {
        ts[j] = static_cast<int>(util::GetLsb(frontier[j]));
        seeds0[j] = util::Xor(hash_key, util::SetLsb(frontier[j], false));
        seeds1[j] = util::Xor(hash_key, util::SetLsb(frontier[j], true));
      }
      auto h0s = detail::GenBatch<1>(prg, seeds0);
      auto h1s = detail::GenBatch<1>(prg, seeds1);
      for (int j = 0; j < 8; ++j) {
        int m = -ts[j];
        int4 h0 = h0s[j][0];
        int4 high0 = MaskXor(h0, hcw, m);
        high0.w &= ~1;
        int low0 = (h0.w & 1) ^ (ts[j] & lcw_0);
        int4 h1 = h1s[j][0];
        int4 high1 = MaskXor(h1, hcw, m);
        high1.w &= ~1;
        int low1 = (h1.w & 1) ^ (ts[j] & lcw_1);

        auto y0 = Group::From(high0);
        if (low0) y0 = y0 + ocw_group;
        if (b) y0 = -y0;
        ys[l + 2 * j] = y0.Into();

        auto y1 = Group::From(high1);
        if (low1) y1 = y1 + ocw_group;
        if (b) y1 = -y1;
        ys[l + 2 * j + 1] = y1.Into();
      }
      return;
    }

    int4 cw = cws[i].s;
    cuda::std::array<int4, 8> seeds;
    int ts[8];
    for (int j = 0; j < 8; ++j) {
      ts[j] = static_cast<int>(util::GetLsb(frontier[j]));
      seeds[j] = util::Xor(hash_key, frontier[j]);
    }
    auto out = detail::GenBatch<1>(prg, seeds);
    // The 16 children interleave in leaf order: node j's children land at 2j, 2j+1 of the
    // next level. De-interleave them into two consecutive groups of 8.
    int4 low[8];
    int4 high[8];
    for (int j = 0; j < 4; ++j) {
      int m = -ts[j];
      int4 left = MaskXor(out[j][0], cw, m);
      low[2 * j] = left;
      low[2 * j + 1] = util::Xor(left, frontier[j]);
      int m4 = -ts[j + 4];
      int4 left4 = MaskXor(out[j + 4][0], cw, m4);
      high[2 * j] = left4;
      high[2 * j + 1] = util::Xor(left4, frontier[j + 4]);
    }

    size_t half = size_t{8} << (in_bits - i - 1);
    EvalBatched8Group(b, low, cws, ocw_group, ys, l, i + 1);
    EvalBatched8Group(b, high, cws, ocw_group, ys, l + half, i + 1);
  }
};

}  // namespace fss
