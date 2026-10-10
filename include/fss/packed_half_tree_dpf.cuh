// SPDX-License-Identifier: Apache-2.0
/**
 * @file packed_half_tree_dpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed point function (DPF) with packed sub-128-bit outputs, based on Half-Tree expansion.
 *
 * @warning Experimental. No published paper describes this construction as a whole: it combines the
 * Half-Tree expansion with truncated-tree packed outputs (the trick popularized by LibDPF) and a
 * domain-separated leaf conversion. The composition has no standalone security proof. The API may
 * change without notice.
 *
 * ## Packed outputs
 *
 * For the output width out_bits < 128, full-domain evaluation can return one 16B block per
 * 128 / out_bits consecutive inputs instead of one element per input: the expansion stops at
 * tree depth in_bits - log2(kLanes), and the bits of each level node cover the kLanes outputs
 * of the inputs under it. This removes the last log2(kLanes) levels of PRG calls and shrinks
 * both the key and the output array by that factor, mirroring what LibDPF does for 1-bit outputs.
 *
 * A block is read as a little-endian 128-bit integer with x the low word and w the high word,
 * matching util::Pack(). Lane j occupies bits [j * out_bits, (j + 1) * out_bits). The output
 * share for input x is lane x & (kLanes - 1) of block x >> log2(kLanes); see Extract().
 *
 * The output group is XOR only: the two parties' blocks XOR-reconstruct to beta at alpha's lane
 * and to the zero block everywhere else. Additive groups cannot be packed into shared lanes,
 * and comparison functions need per-leaf control bits, so no packed counterpart exists for DCF.
 *
 * ## Implementation details
 *
 * The tree levels follow Half-Tree: one mul=1 PRG call per expanded node yields both children
 * (right = left XOR node), and the control bit rides in the node LSB. On the alpha path the two
 * parties' nodes differ by a constant delta equal to the XOR of the two root seeds, and exactly
 * one party has t = 1; off the path the nodes are identical.
 *
 * The leaf conversion is a hash: y = PRG(hash_key XOR node) XOR (t ? fcw : 0), with
 * fcw = packed_beta XOR leaf_0 XOR leaf_1. The hash is required for privacy. Without it the
 * output blocks would be the raw level nodes, and fcw = packed_beta XOR node_0 XOR node_1 would
 * expose the constant path difference — the XOR of the two root seeds — to any key holder
 * whenever beta is known (LibDPF accepts the milder one-bit version of this leak by returning
 * raw blocks; we do not). Its inputs are the level kDepth nodes, which are never expanded, so
 * expansion and conversion read disjoint PRG input sets, as in HalfTreeDpf.
 *
 * ## Complexity
 *
 * Gen makes 2 * (kDepth + 1) PRG calls, Eval kDepth + 1, and EvalAll under 2^(kDepth + 1).
 * A key is kDepth correction words plus fcw, i.e. 16 * (kDepth + 1) bytes.
 *
 * ## References
 *
 * 1. Xiaojie Guo, Kang Yang, Xiao Wang, Wenhao Zhang, Xiang Xie, Jiang Zhang, Zheli Liu: Half-Tree: Halving the Cost of Tree Expansion in COT and DPF. EUROCRYPT (1) 2023: 330-362. <https://doi.org/10.1007/978-3-031-30545-0_12>. @anchor packed_half_tree_dpf_ht
 * 2. Elette Boyle, Niv Gilboa, Yuval Ishai: Function Secret Sharing: Improvements and Extensions. CCS 2016: 1292-1303. <https://doi.org/10.1145/2976749.2978429>. @anchor packed_half_tree_dpf_bgi
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <omp.h>
#include <fss/prg.cuh>
#include <fss/util.cuh>

namespace fss {

/**
 * 2-party DPF scheme with packed sub-128-bit outputs, using the Half-Tree construction.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam out_bits Output width in bits. A power of two within [1, 64]: 128 % out_bits == 0 then
 * holds, so kLanes = 128 / out_bits lanes exactly fill a block. Widths above 64 give a single
 * lane per block and no packing gain; use HalfTreeDpf with its 127-bit group instead.
 * @tparam Prg See Prgable. Requires mul=1 (CCR hash, 128->128 bits).
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, int out_bits, typename Prg, typename In = uint, int par_depth = -1>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Prgable<Prg, 1>)
class PackedHalfTreeDpf {
  static_assert(out_bits >= 1 && out_bits <= 64, "out_bits must be within [1, 64]");
  static_assert((out_bits & (out_bits - 1)) == 0, "out_bits must be a power of two");

public:
  Prg prg;
  int4 hash_key;

  /**
   * Number of consecutive inputs covered by one output block.
   */
  static constexpr int kLanes = 128 / out_bits;

  /**
   * Tree depth that is actually expanded. The remaining log2(kLanes) levels never run.
   */
  static constexpr int kDepth = in_bits - [] {
    int r = 0;
    int v = kLanes;
    while (v > 1) {
      v >>= 1;
      ++r;
    }
    return r;
  }();
  static_assert(kDepth >= 1, "in_bits must exceed log2(kLanes) so that the tree has at least one level");

  /**
   * Largest beta that fits in out_bits bits.
   */
  static constexpr uint64_t kBetaMax = out_bits == 64 ? ~0ULL : (1ULL << out_bits) - 1;

  /**
   * Correction word, one per expanded tree level.
   *
   * ## Layout
   *
   * s is the seed correction, applied by the party whose control bit t is set.
   */
  struct __align__(16) Cw {
    int4 s;
  };
  static_assert(sizeof(Cw) == 16);

  /**
   * Key generation method.
   *
   * @param cws Pre-allocated array of Cw as returns. The array size must be kDepth.
   * @param fcw Final correction word as return. It carries the packed beta.
   * @param s0s 2 initial seeds. Users can randomly sample them.
   * @param a $a$. Its low log2(kLanes) bits select the lane within the output block.
   * @param beta $b$. Must fit in out_bits bits.
   *
   * The key for party i consists of cws + fcw + s0s[i].
   */
  __host__ __device__ void Gen(Cw cws[], int4 &fcw, const int4 s0s[2], In a, uint64_t beta) {
    assert(beta <= kBetaMax);

    int4 packed_beta = PackBeta(a, beta);

    // node1 starts with the control bit set, so the path difference delta keeps LSB 1 and
    // exactly one party has t=1 on the path at every level.
    int4 node0 = util::SetLsb(s0s[0], false);
    int4 node1 = util::SetLsb(s0s[1], true);
    int4 delta = util::Xor(node0, node1);  // LSB = 0^1 = 1

    for (int i = 0; i < kDepth; ++i) {
      int4 h0 = prg.Gen(util::Xor(hash_key, node0))[0];
      int4 h1 = prg.Gen(util::Xor(hash_key, node1))[0];

      bool a_bit = (a >> (in_bits - 1 - i)) & 1;

      // CW = h0 ^ h1 ^ (!a_bit ? delta : 0)
      // When a_bit=0 (go left): non-alpha is right, CW = h0^h1^delta makes right0=right1
      // When a_bit=1 (go right): non-alpha is left, CW = h0^h1 makes left0=left1
      int4 cw = util::Xor(h0, h1);
      if (!a_bit) cw = util::Xor(cw, delta);

      cws[i] = {cw};

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

      // delta = node0 ^ node1 for the next level; stays the root seed XOR
      delta = util::Xor(node0, node1);
    }

    // The leaf hash masks the constant path difference, see the file-level details.
    int4 leaf0 = prg.Gen(util::Xor(hash_key, node0))[0];
    int4 leaf1 = prg.Gen(util::Xor(hash_key, node1))[0];
    fcw = util::Xor(util::Xor(packed_beta, leaf0), leaf1);
  }

  /**
   * Evaluation method.
   *
   * @param b Party index. False for 0 and true for 1. $i$.
   * @param s0 Initial seed of the party.
   * @param cws Returned by Gen().
   * @param fcw Final correction word returned by Gen().
   * @param x Evaluated input. $x$.
   * @return Output share block containing $x$. Extract() reads the lane of $x$ from it.
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], int4 fcw, In x) {
    int4 node = util::SetLsb(s0, b);

    for (int i = 0; i < kDepth; ++i) {
      bool x_bit = (x >> (in_bits - 1 - i)) & 1;
      bool t = util::GetLsb(node);

      int4 h = prg.Gen(util::Xor(hash_key, node))[0];

      int4 zero4 = {0, 0, 0, 0};
      int4 xb_mask = x_bit ? node : zero4;
      int4 t_mask = t ? cws[i].s : zero4;

      node = util::Xor(util::Xor(h, xb_mask), t_mask);
    }

    return ConvertLeaf(node, fcw);
  }

  /**
   * Packs beta into the lane indexed by the low log2(kLanes) bits of a, for checking
   * the XOR-reconstruction of the two parties' blocks in tests and applications.
   */
  static __host__ __device__ int4 PackBeta(In a, uint64_t beta) {
    int lane = static_cast<int>(a) & (kLanes - 1);
    int4 packed = {0, 0, 0, 0};
    if constexpr (out_bits == 64) {
      packed = lane == 0 ? int4{static_cast<int>(beta), static_cast<int>(beta >> 32), 0, 0}
                         : int4{0, 0, static_cast<int>(beta), static_cast<int>(beta >> 32)};
    } else {
      constexpr int lanes_per_word = 32 / out_bits;
      int word = lane / lanes_per_word;
      int off = (lane % lanes_per_word) * out_bits;
      unsigned int v = static_cast<unsigned int>(beta) << off;
      if (word == 0) packed.x = static_cast<int>(v);
      else if (word == 1) packed.y = static_cast<int>(v);
      else if (word == 2) packed.z = static_cast<int>(v);
      else packed.w = static_cast<int>(v);
    }
    return packed;
  }

  /**
   * Extracts the out_bits-bit output share for input x from a packed block.
   *
   * @param block A block returned by Eval() or EvalAll(), or the XOR of the two parties' blocks.
   * @param x The input whose lane is read.
   * @return The out_bits-bit lane value.
   */
  static __host__ __device__ uint64_t Extract(int4 block, In x) {
    int lane = static_cast<int>(x) & (kLanes - 1);
    if constexpr (out_bits == 64) {
      uint64_t lo = lane == 0 ? static_cast<unsigned int>(block.x) : static_cast<unsigned int>(block.z);
      uint64_t hi = lane == 0 ? static_cast<unsigned int>(block.y) : static_cast<unsigned int>(block.w);
      return lo | (hi << 32);
    } else {
      constexpr int lanes_per_word = 32 / out_bits;
      int word = lane / lanes_per_word;
      int off = (lane % lanes_per_word) * out_bits;
      unsigned int w = word == 0 ? static_cast<unsigned int>(block.x)
          : word == 1 ? static_cast<unsigned int>(block.y)
          : word == 2 ? static_cast<unsigned int>(block.z)
          : static_cast<unsigned int>(block.w);
      return (w >> off) & static_cast<unsigned int>((1ULL << out_bits) - 1);
    }
  }

  /**
   * Full domain evaluation method.
   *
   * Evaluate the key on each input, i.e., 0b00...0 - 0b11...1.
   * Store the packed output blocks sequentially: block j covers inputs j * kLanes to
   * (j + 1) * kLanes - 1.
   *
   * @param b Party index. False for 0 and true for 1. $i$.
   * @param s0 Initial seed of the party.
   * @param cws Correction words returned by Gen().
   * @param fcw Final correction word returned by Gen().
   * @param ys Pre-allocated output array. Its size must be at least 2 ** kDepth.
   *
   * Support parallel using OpenMP.
   */
  void EvalAll(bool b, int4 s0, const Cw cws[], int4 fcw, int4 ys[]) {
    int4 node = util::SetLsb(s0, b);

    assert(kDepth < sizeof(size_t) * 8);
    size_t num_blocks = 1ULL << kDepth;

    int par_depth_ = util::ResolveParDepth(par_depth);

    // Phase 1: tree traversal for levels 1..kDepth, stores the level nodes in ys as scratch.
#pragma omp parallel
#pragma omp single
    EvalTree(node, cws, ys, 0, num_blocks, 0, par_depth_);

    // Phase 2: in-place leaf conversion, one block per level node.
#pragma omp parallel for
    for (size_t j = 0; j < num_blocks; ++j) {
      ys[j] = ConvertLeaf(ys[j], fcw);
    }
  }

private:
  __host__ __device__ int4 ConvertLeaf(int4 node, int4 fcw) {
    int4 leaf = prg.Gen(util::Xor(hash_key, node))[0];
    if (util::GetLsb(node)) leaf = util::Xor(leaf, fcw);
    return leaf;
  }

  void EvalTree(int4 node, const Cw cws[], int4 ys[], size_t l, size_t r, int i, int par_depth_) {
    // i is the level index (0-based), we traverse levels 0..kDepth-1.
    // At level kDepth, we store the node.
    if (i == kDepth) {
      assert(l + 1 == r);
      ys[l] = node;
      return;
    }

    bool t = util::GetLsb(node);
    int4 h = prg.Gen(util::Xor(hash_key, node))[0];

    int4 zero4 = {0, 0, 0, 0};
    int4 t_mask = t ? cws[i].s : zero4;

    // Left child: left = H_S(parent) ^ (t ? cw : 0)
    int4 left = util::Xor(h, t_mask);
    // Right child: right = left ^ parent
    int4 right = util::Xor(left, node);

    size_t mid = (l + r) / 2;

    if (i < par_depth_) {
#pragma omp task
      EvalTree(left, cws, ys, l, mid, i + 1, par_depth_);
#pragma omp task
      EvalTree(right, cws, ys, mid, r, i + 1, par_depth_);
#pragma omp taskwait
    } else {
      EvalTree(left, cws, ys, l, mid, i + 1, par_depth_);
      EvalTree(right, cws, ys, mid, r, i + 1, par_depth_);
    }
  }
};

}  // namespace fss
