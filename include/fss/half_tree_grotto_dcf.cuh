// SPDX-License-Identifier: Apache-2.0
/**
 * @file half_tree_grotto_dcf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed comparison function (DCF) over F2 from the packed Half-Tree DPF.
 *
 * @warning Experimental, like PackedHalfTreeDpf, whose keys it evaluates. The API may change
 * without notice.
 *
 * The comparison is the prefix-parity algorithm of Grotto (@ref half_tree_grotto_dcf_grotto "1"),
 * which GrottoDcf runs on the BGI DPF, applied to a PackedHalfTreeDpf key with out_bits = 1 and
 * beta = 1. SIGMA (@ref half_tree_grotto_dcf_sigma "2") runs the same algorithm on the BGI DPF with
 * early termination. A key is the DPF key itself, so PackedHalfTreeDpf::Eval() on the same key
 * still gives the shares of 1[x == alpha].
 *
 * Output shares are in F2 (XOR sharing). Each party holds a single bool per query.
 * For inputs x: share_0 XOR share_1 = 1[alpha <= x], as in GrottoDcf.
 *
 * ## Evaluation
 *
 * The two parties' control bits of a node XOR to 1 exactly when the node is on alpha's path,
 * i.e., to the parity of the 1-bit outputs below it. Eval() walks x's path and XORs the control
 * bit of every node at which x's bit differs from the previous bit, taking the bit before the
 * first as 0. These are the nodes where the path turns. If x leaves alpha's path at tree level d,
 * the XORed bits reconstruct to x's bit at level d, and x's output block reconstructs to zero.
 * If x stays on the path, they reconstruct to x's last tree bit r, and the block holds a single 1
 * at alpha's lane. Eval() then XORs the block lanes up to and including x's lane when r = 0, or
 * the lanes above it when r = 1, which gives 1[alpha <= x] in both cases.
 *
 * ## Complexity
 *
 * Gen makes 2 * (kDepth + 1) PRG calls and Eval kDepth + 1, the costs of PackedHalfTreeDpf.
 * A key is 16 * (kDepth + 1) bytes with kDepth = in_bits - 7.
 *
 * ## References
 *
 * 1. Kyle Storrier, Adithya Vadapalli, Allan Lyons, Ryan Henry: Grotto: Screaming fast (2+1)-PC for Z_{2^n} via (2, 2)-DPFs. CCS 2023: 2143-2157. <https://doi.org/10.1145/3576915.3623147>. <https://eprint.iacr.org/2023/108>. @anchor half_tree_grotto_dcf_grotto
 * 2. Kanav Gupta, Neha Jawalkar, Ananta Mukherjee, Nishanth Chandran, Divya Gupta, Ashish Panwar, Rahul Sharma: SIGMA: Secure GPT Inference with Function Secret Sharing. Cryptology ePrint Archive, Paper 2023/1269. <https://eprint.iacr.org/2023/1269>. @anchor half_tree_grotto_dcf_sigma
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/bit>
#include <fss/grotto_dcf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/util.cuh>

namespace fss {

/**
 * 2-party DCF scheme over F2 from the packed Half-Tree DPF (Grotto construction).
 *
 * @tparam in_bits Input domain bit size. Must exceed 7, the levels a 1-bit output block covers.
 * @tparam Prg See Prgable. Requires mul=1 (CCR hash, 128->128 bits), as PackedHalfTreeDpf.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth Forwarded to PackedHalfTreeDpf, which fixes the Key type. Eval() does not use it.
 */
template <int in_bits, typename Prg, typename In = uint, int par_depth = -1>
class HalfTreeGrottoDcf {
  using DpfType = PackedHalfTreeDpf<in_bits, 1, Prg, In, par_depth>;

public:
  Prg prg;
  int4 hash_key;

  /**
   * Tree depth that is actually expanded, see PackedHalfTreeDpf::kDepth.
   */
  static constexpr int kDepth = DpfType::kDepth;

  using Cw = typename DpfType::Cw;

  /**
   * A party's key, the PackedHalfTreeDpf key with out_bits = 1 and beta = 1.
   */
  using Key = typename DpfType::Key;

  /**
   * Key generation method. Delegates to PackedHalfTreeDpf::Gen() with beta = 1.
   *
   * @param k0, k1 Returned keys of the two parties.
   * @param s0s 2 initial seeds. Users can randomly sample them.
   * @param a The secret comparison threshold.
   */
  __host__ __device__ void Gen(Key &k0, Key &k1, cuda::std::span<const int4, 2> s0s, In a) {
    MakeDpf().Gen(k0, k1, s0s, a, 1);
  }

  /**
   * Evaluation method, see "Evaluation" in the file details.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param key The party's key returned by Gen().
   * @param x Query point.
   * @return bool share such that share_0 XOR share_1 = 1[alpha <= x].
   */
  __host__ __device__ bool Eval(bool b, const Key &key, In x) {
    DpfType dpf = MakeDpf();
    detail::GrottoTurns turns;
    int4 node = dpf.Walk(b, key.s0, key.cws, x, turns);
    return turns.share ^ LeafParity(dpf.ConvertLeaf(node, key.fcw), x);
  }

  /**
   * Leaf step of Eval(), shared with the GPU point launcher.
   *
   * @param block x's output block.
   * @param x The evaluated input.
   * @return The parity of the lanes of block up to and including x's lane when x's last tree bit
   * is 0, or of the lanes above x's lane when it is 1.
   */
  static __host__ __device__ bool LeafParity(int4 block, In x) {
    int lane = static_cast<int>(x) & (DpfType::kLanes - 1);
    bool right = ((x >> (in_bits - kDepth)) & 1) != 0;
    int word = lane >> 5;
    unsigned int partial = 0xffffffffu >> (31 - (lane & 31));
    unsigned int masks[4];
    for (int j = 0; j < 4; ++j) {
      unsigned int upto = j < word ? 0xffffffffu : (j == word ? partial : 0u);
      masks[j] = right ? ~upto : upto;
    }
    unsigned int acc = (static_cast<unsigned int>(block.x) & masks[0]) ^
        (static_cast<unsigned int>(block.y) & masks[1]) ^ (static_cast<unsigned int>(block.z) & masks[2]) ^
        (static_cast<unsigned int>(block.w) & masks[3]);
    return (cuda::std::popcount(acc) & 1) != 0;
  }

private:
  __host__ __device__ DpfType MakeDpf() const {
    return DpfType{prg, hash_key};
  }
};

}  // namespace fss
