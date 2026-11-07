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
#include <stdexcept>

namespace fss {

namespace detail {
class SchemeAccess;

// Split Cws view shared by the host and device split paths. The packed
// construction carries no control bits, so layer l is just cws[l*stride] and the
// terminal entry holds the packed-output correction word at cws[kDepth*stride].
// operator[] recovers the Cw the shared algorithms consume, so their bodies stay
// identical on the Cw and split paths.
template <typename Cw>
struct PackedHalfTreeCwsView {
  const int4 *cws;
  size_t stride;

  __host__ __device__ inline Cw operator[](int l) const {
    return Cw{cws[static_cast<size_t>(l) * stride]};
  }
};

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
 * EvalAll() and EvalRange() are host operations. EvalAllGpu() and EvalAllGpuBatch() are host
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
   * Correction word for a tree level or the final packed-output correction word.
   *
   * ## Layout
   *
   * For tree levels, s is the seed correction word applied when the control bit t is set.
   * The final entry cws[kDepth].s holds the packed-output correction word.
   * @ingroup fss_api
   */
  struct __align__(16) Cw {
    int4 s;  ///< Tree seed correction word, or packed-output correction word in the final entry.
  };
  static_assert(sizeof(Cw) == 16);

  /// Seed-component count Gen() writes at stride 1: kDepth levels plus the packed-output terminal.
  static constexpr int kCwCount = kDepth + 1;

  using Input = In;  ///< The caller-facing input-domain type.

  /**
   * Configure the CCR hash and public instance seed S.
   *
   * @param hash Unary CCR hash backend used throughout generation and evaluation.
   * @param instance_seed Uniformly sampled public seed S, fixed for this instance.
   */
  __host__ __device__ explicit PackedHalfTreeDpf(Hash hash, int4 instance_seed);

  /**
   * Generate split Cws for the packed point function.
   *
   * Layer l is stored at cws[l*stride] and the packed-output correction word at
   * cws[kDepth*stride]. Tree levels hold the seed correction words and the terminal
   * holds the packed-output correction word exactly as the final Cw entry does. The
   * construction carries no control bits, so there is no packed-control
   * parameter.
   *
   * @param cws Output storage; its capacity includes kDepth*stride+1 elements.
   * @param s0s Two initial seeds, retained by the caller.
   * @param a The secret point. Its low log2(kLanes) bits select the packed lane.
   * @param beta Output value, which must fit in out_bits bits.
   * @param stride Positive element stride between layer entries. The default
   * stride of 1 stores the components contiguously.
   * @pre a is in [0, 2^in_bits).
   * @pre beta <= the out_bits-bit maximum.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre Inputs and outputs do not overlap.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A beta, stride, or capacity violation aborts with a diagnostic in
   * Debug builds on both the host and the device, and is undefined behavior in
   * Release builds.
   * @post cws and s0s[party] form the corresponding party's key.
   * @post Only the described prefix is written, and storage beyond it is untouched.
   */
  __host__ __device__ void Gen(cuda::std::span<int4> cws,
      cuda::std::span<const int4, 2> s0s, In a, uint64_t beta, size_t stride = 1);

  /**
   * Evaluate the packed output block containing x from split Cws.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter. The result contains kLanes consecutive shares;
   * Extract(result, x) reads x's share using the original input x.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws Correction words, layer l at index l*stride and the
   * packed-output correction word at index kDepth*stride.
   * @param x Input in [0, 2^in_bits).
   * @param stride Positive stride between Cw entries.
   * @return This party's block of kLanes output shares. Extract() reads x's lane.
   * @pre x is in [0, 2^in_bits).
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A stride or capacity violation aborts with a diagnostic in Debug
   * builds on both the host and the device, and is undefined behavior in
   * Release builds.
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, cuda::std::span<const int4> cws,
      In x, size_t stride = 1);

  /**
   * Extracts the out_bits-bit output share for input x from a packed block.
   *
   * For full-domain output ys, use Extract(ys[x / kLanes], x).
   * The lane is x % kLanes. Pass the original input x, not the block index.
   * For in_bits = 8 and out_bits = 8, kLanes = 16 and x = 42 selects
   * block 2 and lane 10: Extract(ys[2], 42).
   * For EvalRange() output, x is the relative input index j = input - first.
   * Use Extract(ys[j / kLanes], j).
   *
   * @param block A packed output block, or the XOR of the two parties' blocks.
   * @param x The original input for Eval()/EvalAll(), or relative input index for EvalRange().
   * @return The out_bits-bit lane value.
   */
  static __host__ __device__ uint64_t Extract(int4 block, In x);

  /**
   * Evaluate all packed output blocks from split Cws on the host.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter. The output has exactly B = 2^kDepth blocks and
   * matches the Cw form block for block.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws Correction words, layer l at index l*stride and the
   * packed-output correction word at index kDepth*stride.
   * @param ys Pre-allocated output array of at least 2^kDepth blocks, in input order.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre The packed output element count and byte size fit size_t, and ys is allocated.
   * @throws std::invalid_argument if stride is zero or too large or cws is
   * shorter than kDepth*stride+1.
   * @post Block j covers inputs j * kLanes through (j + 1) * kLanes - 1.
   */
  void EvalAll(bool b, int4 s0, cuda::std::span<const int4> cws, int4 ys[],
      int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate the closed input interval [first, last] from split Cws.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter. The block layout matches the Cw form exactly.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval, inclusive.
   * @param last Last input in the interval, inclusive.
   * @param s0 This party's initial seed.
   * @param cws Correction words, layer l at index l*stride and the
   * packed-output correction word at index kDepth*stride.
   * @param ys Output storage for ceil((last - first + 1) / kLanes) blocks.
   * @param par_depth Nonnegative OpenMP split depth. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @pre first <= last and both endpoints belong to the input domain.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre The output block count and byte size fit size_t, and used pointers are nonnull.
   * @throws std::invalid_argument if stride is zero or too large, cws is shorter
   * than kDepth*stride+1, ys is null, first > last, an endpoint is outside the
   * input domain, the interval does not fit size_t, or ys is too short.
   * @post Extract(ys[j / kLanes], j) is this party's share at first + j.
   * Unused lanes in the last output block are zero. Extra capacity is untouched.
   */
  void EvalRange(bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<int4> ys, int par_depth = 0, size_t stride = 1);

  /**
   * Launch full-domain CUDA evaluation from split device Cws.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter. Hash values, resource rules and tuning rules match
   * EvalAllGpu().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device correction words, layer l at index l*stride.
   * @param ys Device storage containing at least 2^kDepth packed output blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive stride between Cw entries.
   * @return Immediate CUDA launch status. Invalid buffer dimensions, strides,
   * null used buffers, or unaddressable byte sizes return cudaErrorInvalidValue.
   * Execution remains asynchronous; the caller observes completion errors
   * through stream.
   * @pre stride > 0 and cws holds at least kDepth*stride+1 elements.
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<int4> ys, cudaStream_t stream = 0, size_t stride = 1) const
    requires(kDepth < sizeof(size_t) * 8);

  /**
   * Launch full-domain CUDA evaluation for s0s.size() keys from split device Cws.
   *
   * The batch uses the level-major layout: key k layer l is read from
   * cws[l*stride + k] and key k's packed-output correction word from
   * cws[kDepth*stride + k]. The construction carries no control bits, so there is
   * no packed-control parameter. stride selects the layout. 0 selects the dense
   * level-major layout, in which the stride is the key count of this call. Any
   * other value is a positive stride of at least s0s.size().
   * Resource and tuning rules match EvalAllGpu(). The batch shares one party index
   * and hash configuration. Every key produces the complete packed domain.
   *
   * @param b Party index shared by the batch.
   * @param s0s Device storage containing one seed per key.
   * @param cws Device level-major correction words.
   * @param ys Device key-major rows of 2^kDepth packed output blocks per key.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride 0 for the dense level-major layout, where the stride is the
   * key count of this call, or a positive element stride >= s0s.size().
   * @return Immediate CUDA status. Empty batches succeed without reading buffers.
   * Invalid dimensions, null used buffers, strides, or unaddressable counts and
   * bytes return cudaErrorInvalidValue. Oversized grids return
   * cudaErrorInvalidConfiguration.
   * @pre stride is 0 or at least s0s.size().
   * @pre cws holds at least kDepth*stride + s0s.size() elements, where a
   * stride of 0 counts as s0s.size().
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpuBatch(bool b, cuda::std::span<const int4> s0s,
      cuda::std::span<const int4> cws, cuda::std::span<int4> ys,
      cudaStream_t stream = 0, size_t stride = 0) const
    requires(kDepth < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  template <int frontier_depth, int threads_per_block>
  struct GpuLaunchShape;

  static __host__ __device__ void ExpandNode(Hash &hash, int4 instance_seed,
      int4 node, int4 cw, int4 &left, int4 &right);
  static __host__ __device__ int4 PathStep(Hash &hash, int4 instance_seed,
      int4 node, int4 cw, bool take_right);
  static __host__ __device__ int4 ConvertLeaf(Hash &hash, int4 instance_seed,
      int4 node, int4 output_cw);

  Hash hash_;
  int4 instance_seed_;

  // Split Cws view over the caller's array. The host methods and the
  // split GPU kernels share this one view; see detail::PackedHalfTreeCwsView.
  using CwsView = detail::PackedHalfTreeCwsView<Cw>;

  // Writes the strided Cw slots.
  struct CwsWriter {
    int4 *cws;
    size_t stride;

    __host__ __device__ void Write(int i, int4 s_cw) {
      cws[static_cast<size_t>(i) * stride] = s_cw;
    }

    __host__ __device__ void Finish(int4 terminal) {
      cws[static_cast<size_t>(kDepth) * stride] = terminal;
    }
  };

  // Shared Cw generation loop. The writer is passed by value and owns
  // only output access, so the existing algorithm and backend use stay unchanged.
  template <typename Writer>
  __host__ __device__ void Generate(Writer writer, const int4 *s0s, In a, uint64_t beta);

  // Shared point evaluation over Cw arrays and split Cws views.
  template <typename Cws>
  __host__ __device__ int4 EvalCws(bool b, int4 s0, Cws cws, In x);

  // Shared full-domain evaluation over Cw arrays and split Cws views.
  template <typename Cws>
  void EvalAllCws(bool b, int4 s0, Cws cws, int4 ys[], int par_depth);

  // Shared interval evaluation. Cws is a Cw pointer on the array form
  // and a CwsView on the split form.
  template <typename Cws>
  void EvalRangeCws(bool b, In first, In last, int4 s0, Cws cws,
      cuda::std::span<int4> ys, int par_depth);

  template <typename Cws>
  void EvalRangeTree(int4 node, Cws cws, int4 ys[], In first,
      In last, In output_first, In low, In high, int level);

  template <int other_bits, typename OtherHash, typename OtherIn>
    requires(ValidForInputBits<OtherIn, other_bits> && CcrHashable<OtherHash>)
  friend class HalfTreeGrottoDcf;

  static constexpr uint64_t kBetaMax = out_bits == 64 ? ~0ULL : (1ULL << out_bits) - 1;
  // Three scalar levels produce the eight parents consumed by Expand8().
  static constexpr int kTopDepth = 3;

  static __host__ __device__ int4 PackBeta(In a, uint64_t beta);

  // Walks x's path to its level-kDepth node. visit(t, x_bit) sees the control bit of every
  // expanded node and the branch x takes from it.
  template <typename Cws, typename Visit>
  __host__ __device__ int4 Walk(bool b, int4 s0, Cws cws, In x, Visit &&visit);

  // lhs ^ (rhs & mask) with mask being 0 or -1 in every lane, so the per-node correction words
  // controlled by the seed LSB apply branchlessly.
  static __host__ __device__ int4 MaskXor(int4 lhs, int4 rhs, int mask);

  // Expands one node with a scalar hash call: left = H_S(node) ^ (t ? cw : 0), right = left ^ node.
  __host__ __device__ void ExpandNode(int4 node, int4 cw, int4 &left, int4 &right);

  // Level-order full-domain evaluation over a batched frontier. The frontier of
  // level i lives in bufs[i & 1]; the final expansion fuses the leaf conversion and writes
  // each ys block exactly once, in the same order the DFS walk produces. The level-order
  // node index equals the in-order leaf index, so the ys layout is unchanged.
  template <typename Cws>
  void EvalAllFrontier(bool b, int4 s0, Cws cws, int4 fcw, int4 ys[], int par_depth);

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
    cuda::std::span<int4> cws, cuda::std::span<const int4, 2> s0s, In a, uint64_t beta,
    size_t stride) {
  util::RequireArg(stride > 0, "PackedHalfTreeDpf::Gen: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(kDepth),
      "PackedHalfTreeDpf::Gen: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  util::RequireArg(cws.size() >= required,
      "PackedHalfTreeDpf::Gen: cws must hold kDepth*stride+1 elements");
  Generate(CwsWriter{cws.data(), stride}, s0s.data(), a, beta);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Writer>
__host__ __device__ inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Generate(
    Writer writer, const int4 *s0s, In a, uint64_t beta) {
  util::RequireArg(beta <= kBetaMax, "PackedHalfTreeDpf::Gen: beta exceeds kBetaMax");

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

    writer.Write(i, cw);

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
  writer.Finish(fcw);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Eval(
    bool b, int4 s0, cuda::std::span<const int4> cws, In x, size_t stride) {
  util::RequireArg(stride > 0, "PackedHalfTreeDpf::Eval: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(kDepth),
      "PackedHalfTreeDpf::Eval: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  util::RequireArg(cws.size() >= required,
      "PackedHalfTreeDpf::Eval: cws must hold kDepth*stride+1 elements");
  return EvalCws(b, s0, CwsView{cws.data(), stride}, x);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Cws>
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalCws(
    bool b, int4 s0, Cws cws, In x) {
  return ConvertLeaf(Walk(b, s0, cws, x, [](bool, bool) {}), cws[kDepth].s);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Cws, typename Visit>
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Walk(
    bool b, int4 s0, Cws cws, In x, Visit &&visit) {
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
    bool b, int4 s0, cuda::std::span<const int4> cws, int4 ys[], int par_depth,
    size_t stride) {
  if (stride == 0) throw std::invalid_argument("PackedHalfTreeDpf::EvalAll: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(kDepth))
    throw std::invalid_argument("PackedHalfTreeDpf::EvalAll: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  if (cws.size() < required)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalAll: cws must hold kDepth*stride+1 elements");
  EvalAllCws(b, s0, CwsView{cws.data(), stride}, ys, par_depth);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Cws>
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalAllCws(
    bool b, int4 s0, Cws cws, int4 ys[], int par_depth) {
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
      ys[j] = EvalCws(b, s0, cws, static_cast<In>(j * kLanes));
    });
  }
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalRange(bool b, In first,
    In last, int4 s0, cuda::std::span<const int4> cws, cuda::std::span<int4> ys,
    int par_depth, size_t stride) {
  if (stride == 0)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(kDepth))
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  if (cws.size() < required || cws.data() == nullptr)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: cws must hold kDepth*stride+1 elements");
  if (ys.data() == nullptr)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: ys must not be null");
  EvalRangeCws(b, first, last, s0, CwsView{cws.data(), stride}, ys, par_depth);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Cws>
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalRangeCws(
    bool b, In first, In last, int4 s0, Cws cws,
    cuda::std::span<int4> ys, int par_depth) {
  assert(par_depth >= 0);
  In domain_last = static_cast<In>(~In{0});
  if constexpr (in_bits < sizeof(In) * 8) {
    domain_last = static_cast<In>((In{1} << in_bits) - 1);
  }
  if (first > last)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: first must not exceed last");
  if (last > domain_last)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: last must stay inside the input domain");
  const __uint128_t distance = static_cast<__uint128_t>(last) - first;
  const __uint128_t block_distance = distance / kLanes;
  // The caller controls the interval, so this is an argument check, not an
  // invariant: with a 128-bit input domain the block distance can exceed
  // size_t, and narrowing it first would shrink the output-size check.
  if (block_distance >= std::numeric_limits<size_t>::max() / sizeof(int4))
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: the interval is too large for size_t");
  const size_t blocks = static_cast<size_t>(block_distance) + 1;
  if (ys.size() < blocks || ys.data() == nullptr)
    throw std::invalid_argument("PackedHalfTreeDpf::EvalRange: ys must hold the covered blocks");

  const size_t partitions = detail::ResolveParPartitions(par_depth, blocks);
  detail::ParallelFor(par_depth, partitions, [&](size_t partition) {
    const size_t begin = partition * (blocks / partitions) +
        (partition < blocks % partitions ? partition : blocks % partitions);
    const size_t end = begin + blocks / partitions + (partition < blocks % partitions);
    for (size_t index = begin; index < end; ++index) ys[index] = {0, 0, 0, 0};
    const __uint128_t begin_offset = static_cast<__uint128_t>(begin) * kLanes;
    const __uint128_t end_offset = end == blocks ? distance :
        static_cast<__uint128_t>(end) * kLanes - 1;
    EvalRangeTree(util::SetLsb(s0, b), cws, ys.data(),
        static_cast<In>(first + begin_offset), static_cast<In>(first + end_offset),
        first, In{0}, domain_last, 0);
  });
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
template <typename Cws>
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalRangeTree(
    int4 node, Cws cws, int4 ys[], In first, In last,
    In output_first, In low, In high, int level) {
  if (level == kDepth) {
    const int4 block = ConvertLeaf(node, cws[kDepth].s);
    const In begin = first > low ? first : low;
    const In end = last < high ? last : high;
    const size_t lanes = static_cast<size_t>(end - begin) + 1;
    for (size_t lane = 0; lane < lanes; ++lane) {
      const In x = static_cast<In>(begin + lane);
      const In offset = static_cast<In>(x - output_first);
      const size_t index = static_cast<size_t>(offset / kLanes);
      ys[index] = util::Xor(ys[index], PackBeta(offset, Extract(block, x)));
    }
    return;
  }
  const In middle = static_cast<In>(low + ((high - low) >> 1));
  int4 left, right;
  ExpandNode(node, cws[level].s, left, right);
  if (first <= middle) {
    EvalRangeTree(left, cws, ys, first, last, output_first, low, middle, level + 1);
  }
  if (last > middle) {
    EvalRangeTree(right, cws, ys, first, last, output_first,
        static_cast<In>(middle + 1), high, level + 1);
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
template <typename Cws>
inline void PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::EvalAllFrontier(
    bool b, int4 s0, Cws cws, int4 fcw, int4 ys[], int par_depth) {
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
    Hash &hash, int4 instance_seed, int4 node, int4 cw, int4 &left, int4 &right) {
  int4 hashed = hash.Hash(util::Xor(instance_seed, node));
  left = MaskXor(hashed, cw, -static_cast<int>(util::GetLsb(node)));
  right = util::Xor(left, node);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::PathStep(
    Hash &hash, int4 instance_seed, int4 node, int4 cw, bool take_right) {
  const int4 zero = {0, 0, 0, 0};
  int4 hashed = hash.Hash(util::Xor(instance_seed, node));
  return util::Xor(util::Xor(hashed, take_right ? node : zero),
      util::GetLsb(node) ? cw : zero);
}

template <int in_bits, int out_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      CcrHashable<Hash> && out_bits >= 1 && out_bits <= 64 && (out_bits & (out_bits - 1)) == 0)
__host__ __device__ inline int4 PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::ConvertLeaf(
    Hash &hash, int4 instance_seed, int4 node, int4 output_cw) {
  int4 leaf = hash.Hash(util::Xor(instance_seed, node));
  if (util::GetLsb(node)) leaf = util::Xor(leaf, output_cw);
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

// Hold the packed scheme behind a tag so the shared HalfTree kernels do not
// spell it in their host stub signatures.
template <int in_bits, int out_bits, typename Hash, typename In>
struct PackedHalfTreeGpuKernelDescriptor {
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>;
  using Backend = Hash;
};

#if defined(__CUDACC__)

// Split-layout counterparts of the shared HalfTree kernels. The Cw
// buffer arrives type-erased and the boundary view reconstructs the per-layer
// Cw entries, so the shared traversal stays identical on both paths.
template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void PackedHalfTreeSplitEvalAllKernel(bool b, int4 s0, const void *cws_buffer,
    size_t stride, int4 *ys, int4 instance_seed, typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::PackedHalfTreeCwsView<typename Scheme::Cw>;
  const auto *cws = static_cast<const int4 *>(cws_buffer);
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  HalfTreeEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0,
      Cws{cws, stride}, ys, instance_seed, hash, blockIdx.x, thread_index,
      shared_frontier);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void PackedHalfTreeSplitEvalAllBatchKernel(bool b, const int4 *s0s,
    const void *cws_buffer, size_t stride, int4 *ys, int4 instance_seed,
    typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::PackedHalfTreeCwsView<typename Scheme::Cw>;
  const auto *cws = static_cast<const int4 *>(cws_buffer);
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  constexpr size_t kOutputsPerKey = size_t{1} << HalfTreeGpuTraits<Scheme>::kOutputBits;
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * kOutputsPerKey;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  HalfTreeEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      Cws{cws + key_index, stride}, key_outputs, instance_seed, hash,
      subtree_block_index, thread_index, shared_frontier);
}

// Key-major batch counterpart: key k's components occupy its whole block of
// records with a unit stride.
template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void PackedHalfTreeSplitEvalAllBatchDeviceKeyKernel(bool b, const int4 *s0s,
    const void *cws_buffer, size_t key_records, size_t key_count, int4 *ys, int4 instance_seed,
    typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::PackedHalfTreeCwsView<typename Scheme::Cw>;
  const auto *cws = static_cast<const int4 *>(cws_buffer);
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  constexpr size_t kOutputsPerKey = size_t{1} << HalfTreeGpuTraits<Scheme>::kOutputBits;
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  const int4 *key = cws + static_cast<size_t>(key_index) * key_records;
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * kOutputsPerKey;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  HalfTreeEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      Cws{key, 1}, key_outputs, instance_seed, hash,
      subtree_block_index, thread_index, shared_frontier);
}

// Host launcher for the key-major batch kernel; the hash backend and instance
// seed arrive by value from a host mirror of the device scheme.
template <int in_bits, int out_bits, typename Hash, typename In, int frontier_depth = 16,
    int threads_per_block = 256>
__host__ inline cudaError_t PackedHalfTreeDpfEvalAllGpuBatchDeviceKey(bool b, const int4 *s0s,
    size_t key_count, const int4 *cws, size_t cws_count, size_t key_records, int4 *ys,
    size_t ys_count, int4 instance_seed, const Hash &hash, cudaStream_t stream) {
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>;
  constexpr int kDepth = Scheme::kDepth;
  using Descriptor = PackedHalfTreeGpuKernelDescriptor<in_bits, out_bits, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << kDepth;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  constexpr int kFrontierDepth = frontier_depth < kDepth ? frontier_depth : kDepth;
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static_assert(kDepth - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  constexpr int kThreadsPerBlockLog2 = [] {
    int requested_log2 = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++requested_log2;
    return requested_log2 < kFrontierDepth ? requested_log2 : kFrontierDepth;
  }();
  constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  constexpr int kBlocksPerKey = 1 << kBlockRootDepth;
  if (key_count == 0) return cudaSuccess;
  if (key_count > kMaxElements ||
      key_count > kMaxAddressableSize / kOutputsPerKey / sizeof(int4) ||
      key_count > kMaxAddressableSize / key_records / sizeof(int4) ||
      cws_count < key_count * key_records || ys_count < key_count * kOutputsPerKey ||
      s0s == nullptr || cws == nullptr || ys == nullptr) return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  PackedHalfTreeSplitEvalAllBatchDeviceKeyKernel<Descriptor, kFrontierDepth, kBlockRootDepth>
      <<<key_count * kBlocksPerKey, 1 << kThreadsPerBlockLog2, 0, stream>>>(
          b, s0s, cws, key_records, key_count, ys, instance_seed, hash);
  return cudaGetLastError();
}

#endif  // defined(__CUDACC__)

#if defined(__CUDACC__)

// Point-evaluate packed HalfTree keys, one thread per key: ys[tid] is x's output
// block, or with cmp the 1[alpha <= x] share of HalfTreeGrottoDcf::Eval(). cws
// holds dense level-major correction words, the layout the split form of Gen()
// writes: layer i of key k is cws[i * nkeys + k] and the final correction word
// is the tail row, cws[kDepth * nkeys + k].
template <int in_bits, int unroll, bool b, bool cmp, typename Descriptor, typename In, typename Query, typename Y>
__global__ void PackedHalfTreeDpfEvalPointKernel(Y *ys, const int4 *seeds, const int4 *cws, const Query *xs,
    int nkeys, typename Descriptor::Backend hash, int4 instance_seed) {
  using Scheme = typename Descriptor::Scheme;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = static_cast<In>(xs[tid]);
  fss::detail::GrottoTurns turns;

#pragma unroll unroll
  for (int i = 0; i < Scheme::kDepth; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    bool t = fss::util::GetLsb(node);
    if constexpr (cmp) turns(t, x_bit);
    int4 h = hash.Hash(fss::util::Xor(instance_seed, node));
    int4 cw = cws[(size_t)i * nkeys + tid];
    node = fss::util::Xor(fss::util::Xor(h, x_bit ? node : Zero4()), t ? cw : Zero4());
  }

  int4 leaf = hash.Hash(fss::util::Xor(instance_seed, node));
  if (fss::util::GetLsb(node)) leaf = fss::util::Xor(leaf, cws[(size_t)Scheme::kDepth * nkeys + tid]);
  if constexpr (cmp) {
    ys[tid] = turns.share ^ fss::detail::SchemeAccess::HalfTreeGrottoLeafParity<Scheme>(leaf, x);
  } else {
    ys[tid] = leaf;
  }
}

template <int unroll, bool cmp, int in_bits, typename Descriptor, typename In, typename Query, typename Y>
void PackedHalfTreeDpfEvalPointLaunch(bool b, const int4 *seeds, const int4 *cws, const Query *xs, Y *ys,
    int nkeys, typename Descriptor::Backend hash, int4 instance_seed, cudaStream_t stream) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    PackedHalfTreeDpfEvalPointKernel<in_bits, unroll, true, cmp, Descriptor, In, Query, Y>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cws, xs, nkeys, hash, instance_seed);
  } else {
    PackedHalfTreeDpfEvalPointKernel<in_bits, unroll, false, cmp, Descriptor, In, Query, Y>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cws, xs, nkeys, hash, instance_seed);
  }
}

/**
 * GPU point evaluation of nkeys Packed HalfTree DPF keys: ys[k] = Eval(b,
 * seeds[k], cws, xs[k]), the output block containing xs[k]. Extract() reads the
 * lane. One thread per key. cws holds dense level-major correction words, as the
 * split form of Gen() writes them, including the final correction word row.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds (device memory), one per key.
 * @param cws Level-major correction words (device memory), (kDepth + 1) * nkeys entries.
 * @param xs Evaluation points (device memory), one per key.
 * @param ys Output array (device memory), one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A PackedHalfTreeDpf instance (holds the CCR hash and instance seed).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, int out_bits, typename Hash, typename In, typename Query = In>
  requires QueryFor<Query, in_bits>
void PackedHalfTreeDpfEvalPointGpu(bool b, cuda::std::span<const int4> seeds, cuda::std::span<const int4> cws,
    cuda::std::span<const Query> xs, int4 *ys, int nkeys, const PackedHalfTreeDpf<in_bits, out_bits, Hash, In> &dpf,
    cudaStream_t stream = 0) {
  if (nkeys < 0)
    throw std::invalid_argument("PackedHalfTreeDpfEvalPointGpu: nkeys must be nonnegative");
  if (nkeys == 0) return;
  const size_t count = static_cast<size_t>(nkeys);
  constexpr size_t kLevels = static_cast<size_t>(PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::kDepth) + 1;
  if (count > SIZE_MAX / kLevels || cws.size() < count * kLevels)
    throw std::invalid_argument("PackedHalfTreeDpfEvalPointGpu: cws must hold (kDepth+1)*nkeys elements");
  if (seeds.size() < count || xs.size() < count || ys == nullptr)
    throw std::invalid_argument("PackedHalfTreeDpfEvalPointGpu: seeds, xs, and ys must cover nkeys entries");
  using Descriptor = PackedHalfTreeGpuKernelDescriptor<in_bits, out_bits, Hash, In>;
  PackedHalfTreeDpfEvalPointLaunch<unroll, false, in_bits, Descriptor, In, Query, int4>(b, seeds.data(), cws.data(),
      xs.data(), ys, nkeys, fss::detail::SchemeAccess::GetHash(dpf),
      fss::detail::SchemeAccess::GetInstanceSeed(dpf), stream);
}

#endif  // defined(__CUDACC__)

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
    bool b, int4 s0, cuda::std::span<const int4> cws, cuda::std::span<int4> ys,
    cudaStream_t stream, size_t stride) const
  requires(kDepth < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using Descriptor = gpu::detail::PackedHalfTreeGpuKernelDescriptor<in_bits, out_bits, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << kDepth;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(kDepth))
    return cudaErrorInvalidValue;
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  if (cws.size() < required || ys.size() < kOutputsPerKey ||
      cws.data() == nullptr || ys.data() == nullptr ||
      kOutputsPerKey > std::numeric_limits<size_t>::max() / sizeof(int4))
    return cudaErrorInvalidValue;
  gpu::detail::PackedHalfTreeSplitEvalAllKernel<Descriptor, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth>
      <<<LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), stride, ys.data(), instance_seed_, hash_);
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
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const int4> cws,
    cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const
  requires(kDepth < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using Descriptor = gpu::detail::PackedHalfTreeGpuKernelDescriptor<in_bits, out_bits, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << kDepth;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  const size_t key_count = s0s.size();
  if (key_count == 0) return cudaSuccess;
  // The output guard bounds key_count * kOutputsPerKey * sizeof(int4) so the
  // product in the size check below cannot wrap.
  if (key_count > kMaxElements ||
      key_count > kMaxAddressableSize / kOutputsPerKey / sizeof(int4) ||
      (stride != 0 && (stride < key_count || stride > kMaxElements)))
    return cudaErrorInvalidValue;
  // A stride of 0 resolves to the key count, which is the dense level-major layout.
  const size_t resolved_stride = stride == 0 ? key_count : stride;
  if (resolved_stride > (kMaxElements - key_count) / static_cast<size_t>(kDepth))
    return cudaErrorInvalidValue;
  const size_t required = static_cast<size_t>(kDepth) * resolved_stride + key_count;
  if (cws.size() < required || ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::PackedHalfTreeSplitEvalAllBatchKernel<Descriptor, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), resolved_stride, ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
