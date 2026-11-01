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
 * of the inputs under it. This removes the last log2(kLanes) levels of hash calls and shrinks
 * both the key and the output array by that factor, mirroring what LibDPF does for 1-bit outputs.
 *
 * A block is read as a little-endian 128-bit integer with x the low word and w the high word,
 * matching fss::detail::Pack(). Lane j occupies bits [j * out_bits, (j + 1) * out_bits). The output
 * share for input x is lane x & (kLanes - 1) of block x >> log2(kLanes); see Extract().
 *
 * The output group is XOR only: the two parties' blocks XOR-reconstruct to beta at alpha's lane
 * and to the zero block everywhere else. Additive groups cannot be packed into shared lanes.
 * HalfTreeGrottoDcf evaluates a 1-bit key with beta = 1 as a comparison function.
 *
 * ## Implementation details
 *
 * The tree levels follow Half-Tree: one CCR hash call per expanded node yields both children
 * (right = left XOR node), and the control bit rides in the node LSB. On the alpha path the two
 * parties' nodes differ by a constant delta equal to the XOR of the two root seeds, and exactly
 * one party has t = 1; off the path the nodes are identical.
 *
 * The leaf conversion is a hash: y = H(S XOR node) XOR (t ? fcw : 0), with
 * fcw = packed_beta XOR leaf_0 XOR leaf_1. The hash is required for privacy. Without it the
 * output blocks would be the raw level nodes, and fcw = packed_beta XOR node_0 XOR node_1 would
 * expose the constant path difference — the XOR of the two root seeds — to any key holder
 * whenever beta is known (LibDPF accepts the milder one-bit version of this leak by returning
 * raw blocks; we do not). Its inputs are the level kDepth nodes, which are never expanded, so
 * expansion and conversion read disjoint hash input sets, as in HalfTreeDpf.
 *
 * EvalAll() runs in level order: a scalar top widens the root to an 8-node
 * frontier, then every level advances 8 frontier nodes per batched expansion
 * through detail::HashBatch(). The correction words apply through branchless
 * masks, and the leaf conversion is
 * fused into the final expansion so each output block is written once. The
 * level-order node index equals the in-order leaf index the DFS walk produces,
 * so the ys layout is identical. Domains with kDepth <= kTopDepth never fill a
 * frontier and evaluate one point per block.
 *
 * ## Complexity
 *
 * Gen makes 2 * (kDepth + 1) hash calls, Eval kDepth + 1, and EvalAll under 2^(kDepth + 1).
 * A key contains kDepth + 1 correction words and s0, i.e. 16 * (kDepth + 2) bytes.
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
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fss/hash.cuh>
#include <fss/util.cuh>
#include <fss/constraints.cuh>
#include <cuda/std/span>
#include <limits>

#include <fss/half_tree_dpf.cuh>

namespace fss {

namespace detail {
class SchemeAccess;
}  // namespace detail

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
class HalfTreeGrottoDcf;

/**
 * 2-party DPF scheme with packed sub-128-bit outputs, using the Half-Tree construction.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam out_bits Output width in bits. A power of two within [1, 64]: 128 % out_bits == 0 then
 * holds, so kLanes = 128 / out_bits lanes exactly fill a block. Widths above 64 give a single
 * lane per block and no packing gain; use HalfTreeDpf with its 127-bit group instead.
 * @tparam Hash Unary CCR hash backend. See CcrHashable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @ingroup fss_api
 *
 * Gen() and scalar Eval() are callable from host and device when the backend supports
 * the execution space and all accessed storage is accessible there.
 * EvalAll() is a host operation. EvalAllGpu() and EvalAllGpuBatch() are host
 * launch methods compiled with a CUDA compiler; they do not run on the device.
 */
template <int in_bits, int out_bits, typename Hash, typename In = uint>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
class PackedHalfTreeDpf {
public:
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
   * Correction word for a tree level or the final packed-output correction.
   *
   * ## Layout
   *
   * For tree levels, s is the seed correction applied when the control bit t is set.
   * The final entry cws[kDepth].s holds the packed-output correction.
   * @ingroup fss_api
   */
  struct __align__(16) Cw {
    int4 s;  ///< Tree seed correction, or packed-output correction in the final entry.
  };
  static_assert(sizeof(Cw) == 16);

  /**
   * A party's key: shared corrections, including the final output correction, and
   * the party's initial seed.
   * @ingroup fss_api
   */
  struct Key {
    Cw cws[kDepth + 1];  ///< Tree corrections followed by the packed-output correction.
    int4 s0;  ///< This party's initial seed.
  };

  /**
   * Configure the CCR hash and public instance seed S.
   *
   * @param hash Unary CCR hash backend used throughout generation and evaluation.
   * @param instance_seed Uniformly sampled public seed S, fixed for this instance.
   */
  __host__ __device__ explicit PackedHalfTreeDpf(Hash hash, int4 instance_seed);

  /**
   * Generate shared corrections for the packed point function.
   *
   * @param cws Output array of kDepth + 1 shared corrections, including the output correction.
   * @param s0s Two initial seeds, retained by the caller.
   * @param a The secret point. Its low log2(kLanes) bits select the packed lane.
   * @param beta Output value, which must fit in out_bits bits.
   * @pre a is in [0, 2^in_bits).
   * @post cws[kDepth].s holds the final packed-output correction.
   */
  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a, uint64_t beta);

  /**
   * Evaluate the packed output block containing x.
   *
   * The result contains kLanes consecutive shares, not only x's share.
   * Extract(result, x) reads x's share using the original input x.
   * It is the same packed block as ys[x / kLanes] from EvalAll().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws The kDepth + 1 shared corrections returned by Gen().
   * @param x Input in [0, 2^in_bits).
   * @return This party's block of kLanes output shares. Extract() reads x's lane.
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], In x);

  /**
   * Extracts the out_bits-bit output share for input x from a packed block.
   *
   * For full-domain output ys, use Extract(ys[x / kLanes], x).
   * The lane is x % kLanes. Pass the original input x, not the block index.
   * For in_bits = 8 and out_bits = 8, kLanes = 16 and x = 42 selects
   * block 2 and lane 10: Extract(ys[2], 42).
   *
   * @param block A block returned by Eval() or EvalAll(), or the XOR of the two parties' blocks.
   * @param x The original input in [0, 2^in_bits) whose lane is read.
   * @return The out_bits-bit lane value.
   */
  static __host__ __device__ uint64_t Extract(int4 block, In x);

  /**
   * Evaluate all packed output blocks on the host.
   *
   * The output has exactly B = 2^kDepth = 2^in_bits / kLanes blocks.
   * Its array payload uses B * sizeof(int4) bytes per party.
   * Two parties' output arrays use twice that amount.
   * Corrections, seeds, and any temporary storage are separate.
   * For in_bits = 8 and out_bits = 8, B = 16, so ys has 16 int4 entries
   * rather than 256 entries. Extract(ys[x / kLanes], x) reads input x.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws The kDepth + 1 shared corrections returned by Gen().
   * @param ys Pre-allocated output array of at least 2^kDepth blocks, in input order.
   * Block j covers inputs j * kLanes through (j + 1) * kLanes - 1.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The packed output element count and byte size fit size_t, and ys is allocated.
   */
  void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[], int par_depth = 0);


  /**
   * Launch full-domain CUDA evaluation from this host-accessible instance.
   *
   * Hash values and the instance seed are passed as kernel arguments. The caller
   * prepares device buffers and any device resources referenced by Hash, and
   * keeps them alive until stream completes. Hash and output operations must
   * support concurrent device execution. No allocation, upload or synchronization
   * is performed. The scheme itself must be accessible on the host.
   * @tparam frontier_depth Frontier depth, capped at kDepth. Defaults to 16.
   * @tparam threads_per_block A power of two in [1,1024], capped at the frontier width.
   * Defaults to 256. Block-root depth is derived from the effective frontier and
   * block size. At most eight output levels may remain below the frontier, and
   * the block-root depth must be at most 30. These are fixed correctness caps.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device storage containing at least kDepth + 1 corrections.
   * @param ys Device storage containing at least 2^kDepth packed output blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @return Immediate CUDA launch status. Invalid buffer dimensions, null used
   * buffers, or unaddressable byte sizes return cudaErrorInvalidValue. Execution
   * remains asynchronous; the caller observes completion errors through stream.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const Cw> cws,
      cuda::std::span<int4> ys, cudaStream_t stream = 0) const
    requires(kDepth < sizeof(size_t) * 8);

  /**
   * Launch full-domain evaluation for s0s.size() keys with this configuration.
   *
   * Uses the same execution, resource and tuning rules as EvalAllGpu().
   * Seeds, corrections and outputs are device-accessible buffers. Corrections
   * and outputs are key-major rows, with kDepth + 1 corrections and
   * 2^kDepth packed output blocks per key. The key count comes from s0s.size().
   * @param b Party index shared by the batch.
   * @param s0s Device storage containing one seed per key.
   * @param cws Flat device correction rows.
   * @param ys Flat device output rows.
   * @param stream Stream on the device owning the referenced resources.
   * @return Immediate CUDA status. Empty batches succeed without reading buffers.
   * Invalid dimensions, null used buffers, or unaddressable counts and bytes
   * return cudaErrorInvalidValue. Oversized grids return cudaErrorInvalidConfiguration.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpuBatch(bool b, cuda::std::span<const int4> s0s,
      cuda::std::span<const Cw> cws, cuda::std::span<int4> ys,
      cudaStream_t stream = 0) const
    requires(kDepth < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  template <int frontier_depth, int threads_per_block>
  struct GpuLaunchShape;

  static __host__ __device__ void ExpandNode(Hash &hash, int4 instance_seed,
      int4 node, int4 correction, int4 &left, int4 &right);
  static __host__ __device__ int4 PathStep(Hash &hash, int4 instance_seed,
      int4 node, int4 correction, bool take_right);
  static __host__ __device__ int4 ConvertLeaf(Hash &hash, int4 instance_seed,
      int4 node, int4 output_correction);

  Hash hash_;
  int4 instance_seed_;

  template <int other_bits, typename OtherHash, typename OtherIn>
    requires(ValidForInputBits<OtherIn, other_bits> && CcrHashable<OtherHash>)
  friend class HalfTreeGrottoDcf;

  static constexpr uint64_t kBetaMax = out_bits == 64 ? ~0ULL : (1ULL << out_bits) - 1;
  // Three scalar levels produce the eight parents consumed by Expand8().
  static constexpr int kTopDepth = 3;

  static __host__ __device__ int4 PackBeta(In a, uint64_t beta);

  // Walks x's path to its level-kDepth node. visit(t, x_bit) sees the control bit of every
  // expanded node and the branch x takes from it.
  template <typename Visit>
  __host__ __device__ int4 Walk(bool b, int4 s0, const Cw cws[], In x, Visit &&visit);

  // lhs ^ (rhs & mask) with mask being 0 or -1 in every lane, so the per-node corrections
  // controlled by the seed LSB apply branchlessly.
  static __host__ __device__ int4 MaskXor(int4 lhs, int4 rhs, int mask);

  // Expands one node with a scalar hash call: left = H_S(node) ^ (t ? cw : 0), right = left ^ node.
  __host__ __device__ void ExpandNode(int4 node, int4 cw, int4 &left, int4 &right);

  // Level-order full-domain evaluation over a batched frontier. The frontier of
  // level i lives in bufs[i & 1]; the final expansion fuses the leaf conversion and writes
  // each ys block exactly once, in the same order the DFS walk produces. The level-order
  // node index equals the in-order leaf index, so the ys layout is unchanged.
  void EvalAllFrontier(bool b, int4 s0, const Cw cws[], int4 fcw, int4 ys[], int par_depth);

  // Expands the 8 frontier parents src[8k..8k+8) into their 16 children with one batch.
  // Parent p has children 2p and 2p + 1, so the children of a batch interleave in pairs.
  void Expand8(const int4 *src, int4 *dst, size_t k, int4 cw);

  // Last-level Expand8() with the leaf conversion fused in: the children are the level
  // kDepth nodes, so they are hashed straight from registers into two more batches
  // (even and odd children), fcw applies through the child LSB mask, and each final block
  // is stored exactly once.
  void ExpandConvert8(const int4 *src, int4 *ys, size_t k, int4 cw, int4 fcw);

  __host__ __device__ int4 ConvertLeaf(int4 node, int4 fcw);
};

// -----------------------------------------------------------------------------

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::PackedHalfTreeDpf(Hash hash, int4 instance_seed)
    : hash_(hash), instance_seed_(instance_seed) {}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::PackBeta(
    In a, uint64_t beta) {
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

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline uint64_t PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Extract(
    int4 block, In x) {
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

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Gen(
    Cw cws[], const int4 s0s[2], In a, uint64_t beta) {
  assert(beta <= kBetaMax);

  int4 packed_beta = PackBeta(a, beta);

  // node1 starts with the control bit set, so the path difference delta keeps LSB 1 and
  // exactly one party has t=1 on the path at every level.
  int4 node0 = util::SetLsb(s0s[0], false);
  int4 node1 = util::SetLsb(s0s[1], true);
  int4 delta = util::Xor(node0, node1);  // LSB = 0^1 = 1

  for (int i = 0; i < kDepth; ++i) {
    int4 h0 = hash_.Hash(util::Xor(instance_seed_, node0));
    int4 h1 = hash_.Hash(util::Xor(instance_seed_, node1));

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
  int4 leaf0 = hash_.Hash(util::Xor(instance_seed_, node0));
  int4 leaf1 = hash_.Hash(util::Xor(instance_seed_, node1));
  int4 fcw = util::Xor(util::Xor(packed_beta, leaf0), leaf1);
  cws[kDepth] = {fcw};
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Eval(
    bool b, int4 s0, const Cw cws[], In x) {
  return ConvertLeaf(Walk(b, s0, cws, x, [](bool, bool) {}), cws[kDepth].s);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Visit>
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Walk(
    bool b, int4 s0, const Cw cws[], In x, Visit &&visit) {
  int4 node = util::SetLsb(s0, b);

  for (int i = 0; i < kDepth; ++i) {
    bool x_bit = (x >> (in_bits - 1 - i)) & 1;
    bool t = util::GetLsb(node);
    visit(t, x_bit);

    node = PathStep(hash_, instance_seed_, node, cws[i].s, x_bit);
  }

  return node;
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalAll(
    bool b, int4 s0, const Cw cws[], int4 ys[], int par_depth) {
  assert(par_depth >= 0);
  static_assert(detail::FitsFullDomain<int4, kDepth>(),
      "packed output count and byte size must fit in size_t");
  int4 fcw = cws[kDepth].s;
  if constexpr (kDepth > kTopDepth) {
    EvalAllFrontier(b, s0, cws, fcw, ys, par_depth);
  } else {
    // Domains too shallow to fill an 8-wide frontier evaluate one point per
    // block through the tested point walk.
    constexpr size_t blocks = size_t{1} << kDepth;
    fss::detail::ParallelFor(par_depth, blocks, [&](size_t j) {
      ys[j] = Eval(b, s0, cws, static_cast<In>(j * kLanes));
    });
  }
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::MaskXor(
    int4 lhs, int4 rhs, int mask) {
  return {lhs.x ^ (rhs.x & mask), lhs.y ^ (rhs.y & mask), lhs.z ^ (rhs.z & mask), lhs.w ^ (rhs.w & mask)};
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::ExpandNode(
    int4 node, int4 cw, int4 &left, int4 &right) {
  ExpandNode(hash_, instance_seed_, node, cw, left, right);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalAllFrontier(
    bool b, int4 s0, const Cw cws[], int4 fcw, int4 ys[], int par_depth) {
  constexpr size_t num_blocks = detail::FullDomainSize<int4, kDepth>();

  // Frontier ping-pong scratch, grown on demand and reused across calls of this thread.
  // Two halves of num_blocks / 2 each: no frontier before the last level exceeds that size.
  // thread_local isolates concurrent calls: each thread pays the first-call allocation and
  // retains the peak 16 * 2 ** kDepth bytes, about one packed output buffer, until it exits.
  static thread_local std::vector<int4> scratch;
  if (scratch.size() < num_blocks) scratch.resize(num_blocks);
  int4 *bufs[2] = {scratch.data(), scratch.data() + num_blocks / 2};

  // Scalar top: the root becomes the 2^kTopDepth level-kTopDepth frontier in bufs[kTopDepth & 1].
  bufs[0][0] = util::SetLsb(s0, b);
  for (int i = 0; i < kTopDepth; ++i) {
    size_t count = size_t{1} << i;
    for (size_t j = 0; j < count; ++j) {
      ExpandNode(bufs[i & 1][j], cws[i].s, bufs[(i + 1) & 1][2 * j], bufs[(i + 1) & 1][2 * j + 1]);
    }
  }

  // Each level bounds its independent partitions by 2^par_depth. The
  // sequential branch avoids an inactive OpenMP region as well as workers.
  int depth = fss::detail::ResolveParDepth(par_depth, kDepth);
  if (depth == 0) {
    for (int i = kTopDepth; i < kDepth; ++i) {
      const int4 *src = bufs[i & 1];
      int4 *dst = i + 1 == kDepth ? ys : bufs[(i + 1) & 1];
      size_t batches = (size_t{1} << i) / 8;
      for (size_t k = 0; k < batches; ++k) {
        if (i + 1 == kDepth) ExpandConvert8(src, dst, k, cws[i].s, fcw);
        else Expand8(src, dst, k, cws[i].s);
      }
    }
    return;
  }
  int threads = fss::detail::ResolveParThreads(depth, num_blocks / 16);
#pragma omp parallel num_threads(threads)
  {
    for (int i = kTopDepth; i < kDepth; ++i) {
      const int4 *src = bufs[i & 1];
      int4 *dst = i + 1 == kDepth ? ys : bufs[(i + 1) & 1];
      size_t batches = (size_t{1} << i) / 8;
      size_t partitions = fss::detail::ResolveParPartitions(depth, batches);
#pragma omp for schedule(static)
      for (size_t partition = 0; partition < partitions; ++partition) {
        size_t begin = partition * (batches / partitions) +
            (partition < batches % partitions ? partition : batches % partitions);
        size_t end = begin + batches / partitions + (partition < batches % partitions);
        for (size_t k = begin; k < end; ++k) {
          if (i + 1 == kDepth) ExpandConvert8(src, dst, k, cws[i].s, fcw);
          else Expand8(src, dst, k, cws[i].s);
        }
      }
    }
  }
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Expand8(
    const int4 *src, int4 *dst, size_t k, int4 cw) {
  int4 parents[8];
  cuda::std::array<int4, 8> seeds;
  for (int j = 0; j < 8; ++j) {
    parents[j] = src[8 * k + j];
    seeds[j] = util::Xor(instance_seed_, parents[j]);
  }
  auto hs = detail::HashBatch(hash_, seeds);
  for (int j = 0; j < 8; ++j) {
    int4 left = MaskXor(hs[j], cw, -static_cast<int>(util::GetLsb(parents[j])));
    dst[16 * k + 2 * j] = left;
    dst[16 * k + 2 * j + 1] = util::Xor(left, parents[j]);
  }
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::ExpandConvert8(
    const int4 *src, int4 *ys, size_t k, int4 cw, int4 fcw) {
  int4 parents[8];
  cuda::std::array<int4, 8> seeds;
  for (int j = 0; j < 8; ++j) {
    parents[j] = src[8 * k + j];
    seeds[j] = util::Xor(instance_seed_, parents[j]);
  }
  auto hs = detail::HashBatch(hash_, seeds);

  int4 children[16];
  cuda::std::array<int4, 8> conv_seeds[2];
  for (int j = 0; j < 8; ++j) {
    int4 left = MaskXor(hs[j], cw, -static_cast<int>(util::GetLsb(parents[j])));
    children[2 * j] = left;
    children[2 * j + 1] = util::Xor(left, parents[j]);
    conv_seeds[0][j] = util::Xor(instance_seed_, children[2 * j]);
    conv_seeds[1][j] = util::Xor(instance_seed_, children[2 * j + 1]);
  }
  auto leaves_even = detail::HashBatch(hash_, conv_seeds[0]);
  auto leaves_odd = detail::HashBatch(hash_, conv_seeds[1]);
  for (int j = 0; j < 8; ++j) {
    ys[16 * k + 2 * j] = MaskXor(leaves_even[j], fcw, -static_cast<int>(util::GetLsb(children[2 * j])));
    ys[16 * k + 2 * j + 1] =
        MaskXor(leaves_odd[j], fcw, -static_cast<int>(util::GetLsb(children[2 * j + 1])));
  }
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::ConvertLeaf(
    int4 node, int4 fcw) {
  return ConvertLeaf(hash_, instance_seed_, node, fcw);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::ExpandNode(
    Hash &hash, int4 instance_seed, int4 node, int4 correction, int4 &left, int4 &right) {
  int4 hashed = hash.Hash(util::Xor(instance_seed, node));
  left = MaskXor(hashed, correction, -static_cast<int>(util::GetLsb(node)));
  right = util::Xor(left, node);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::PathStep(
    Hash &hash, int4 instance_seed, int4 node, int4 correction, bool take_right) {
  const int4 zero = {0, 0, 0, 0};
  int4 hashed = hash.Hash(util::Xor(instance_seed, node));
  return util::Xor(util::Xor(hashed, take_right ? node : zero),
      util::GetLsb(node) ? correction : zero);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::ConvertLeaf(
    Hash &hash, int4 instance_seed, int4 node, int4 output_correction) {
  int4 leaf = hash.Hash(util::Xor(instance_seed, node));
  if (util::GetLsb(node)) leaf = util::Xor(leaf, output_correction);
  return leaf;
}

namespace gpu::detail {

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
struct HalfTreeGpuTraits<PackedHalfTreeDpf<in_bits, out_bits, Hash, In>> {
  using Backend = Hash;
  static constexpr int kExpansionDepth = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::kDepth;
  static constexpr int kOutputBits = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::kDepth;
  static constexpr bool kPacked = true;
};

}  // namespace gpu::detail

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <int frontier_depth, int threads_per_block>
struct PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::GpuLaunchShape {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static_assert(kDepth >= 0, "gpu tree must have a nonnegative expansion depth");
  static constexpr int kFrontierDepth = frontier_depth < kDepth ? frontier_depth : kDepth;
  static_assert(kDepth - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  static constexpr int kThreadsPerBlockLog2 = [] {
    int depth = 0;
    for (int count = threads_per_block; count > 1; count >>= 1) ++depth;
    return depth < kFrontierDepth ? depth : kFrontierDepth;
  }();
  static constexpr int kThreadsPerBlock = 1 << kThreadsPerBlockLog2;
  static constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  static_assert(kBlockRootDepth <= 30, "gpu block index exceeds the signed-int range");
  static constexpr size_t kBlocksPerKey = size_t{1} << (kBlockRootDepth <= 30 ? kBlockRootDepth : 0);
};

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const
  requires(kDepth < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  constexpr size_t kOutputsPerKey = size_t{1} << kDepth;
  constexpr size_t kCorrectionsPerKey = kDepth + 1;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  if (cws.size() < kCorrectionsPerKey || ys.size() < kOutputsPerKey ||
      cws.data() == nullptr || ys.data() == nullptr ||
      kOutputsPerKey > kMaxAddressableSize / sizeof(int4)) return cudaErrorInvalidValue;
  gpu::detail::HalfTreeEvalAllKernel<PackedHalfTreeDpf<in_bits, out_bits, Hash, In>, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalAllGpuBatch(
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const
  requires(kDepth < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  constexpr size_t kOutputsPerKey = size_t{1} << kDepth;
  constexpr size_t kCorrectionsPerKey = kDepth + 1;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  const size_t key_count = s0s.size();
  if (key_count == 0) return cudaSuccess;
  if (key_count > kMaxAddressableSize / kCorrectionsPerKey / sizeof(Cw) ||
      key_count > kMaxAddressableSize / kOutputsPerKey / sizeof(int4) ||
      cws.size() < key_count * kCorrectionsPerKey || ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::HalfTreeEvalAllBatchKernel<PackedHalfTreeDpf<in_bits, out_bits, Hash, In>, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
