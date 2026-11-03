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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cuda/std/array>
#include <fss/group.cuh>
#include <fss/hash.cuh>
#include <fss/util.cuh>
#include <fss/constraints.cuh>
#include <cuda/std/span>
#include <limits>

#include <cstdint>

namespace fss {

namespace detail {
class SchemeAccess;

// Split Cws view shared by the host and device split paths. Layer l is
// read from cws[l*stride]. Only the final tree level (l == in_bits - 1) carries a
// control component, and it is bit in_bits - 1 of packed_tr; every other bit is
// ignored. Each other Cw entry is rebuilt with extra = false, so the terminal
// entry holds the plain output correction word. operator[] recovers the Cw the shared
// algorithms consume, so their bodies stay identical on the Cw and split paths.
template <typename Cw, typename In, int in_bits>
struct HalfTreeDpfCwsView {
  const int4 *cws;
  In packed_tr;
  size_t stride;

  __host__ __device__ inline Cw operator[](int l) const {
    const int4 s_cw = cws[static_cast<size_t>(l) * stride];
    if (l == in_bits - 1) return Cw{s_cw, ((packed_tr >> (in_bits - 1)) & 1) != 0};
    return Cw{s_cw, false};
  }
};

}  // namespace detail

/**
 * 2-party DPF scheme using the Half-Tree construction.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Hash Unary CCR hash backend. See CcrHashable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @ingroup fss_api
 *
 * Gen() and scalar Eval() are callable from host and device when the backend supports
 * the execution space and all accessed storage is accessible there.
 * EvalAll() and EvalRange() are host operations. EvalAllGpu() and EvalAllGpuBatch() are host
 * launch methods compiled with a CUDA compiler; they do not run on the device.
 */
template <int in_bits, typename Group, typename Hash, typename In = uint>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
class HalfTreeDpf {
public:
  /**
   * Correction word.
   *
   * For levels 1..n-1, s is the CW and extra is unused (false).
   * For level n, s stores SetLsb(HCW, LCW_0) and extra stores LCW_1.
   * The final entry cws[in_bits] stores the output correction word in s. Its extra is false
   * after Gen() and is ignored by evaluation.
   * @ingroup fss_api
   */
  struct __align__(32) Cw {
    int4 s;  ///< Seed correction word, or final HCW with LCW_0 in its least significant bit.
    bool extra;  ///< Final tree-level LCW_1. Ignored in earlier levels and the output correction word.
  };
  static_assert(sizeof(Cw) == 32);

  /**
   * A party's key: the shared correction words and the party's initial seed.
   * @ingroup fss_api
   */
  struct Key {
    Cw cws[in_bits + 1];  ///< Tree correction words followed by the output correction word in the final s field.
    int4 s0;  ///< This party's initial seed.
  };

  /**
   * Configure the CCR hash and public instance seed S.
   *
   * @param hash Unary CCR hash backend used throughout generation and evaluation.
   * @param instance_seed Uniformly sampled public seed S, fixed for this instance.
   */
  __host__ __device__ explicit HalfTreeDpf(Hash hash, int4 instance_seed);

  /**
   * Generate shared correction words in caller-owned storage.
   *
   * @param cws Output array of in_bits + 1 correction words, including the output correction word.
   * @param s0s Two initial seeds, retained by the caller.
   * @param a The secret point.
   * @param b_buf Output value, clamped before conversion to Group.
   * @pre a is in [0, 2^in_bits).
   * @post cws[in_bits].s holds the output correction word and cws[in_bits].extra is false.
   */
  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a, int4 b_buf);

  /**
   * Generate split Cws from caller-owned initial seeds.
   *
   * Layer l is stored at cws[l*stride] and the terminal at cws[in_bits*stride].
   * Levels 0..in_bits-2 store a plain seed correction word. The final tree level
   * (in_bits - 1) stores SetLsb(HCW, LCW_0) exactly as Cw::s does, and LCW_1
   * moves out of Cw::extra into bit in_bits-1 of tr. The control component exists
   * only at that level, so every other bit of tr is zero on output. The terminal
   * cws[in_bits*stride] holds the plain output correction word in Cw::s.
   *
   * @param cws Output storage; its capacity includes in_bits*stride+1 elements.
   * @param tr Output packed control bits; only bit in_bits-1 is written.
   * @param s0s Two initial seeds, retained by the caller.
   * @param a The secret point.
   * @param b_buf Output value, clamped before conversion to Group.
   * @param stride Positive element stride between layer entries. The default
   * stride of 1 stores the components contiguously.
   * @pre a is in [0, 2^in_bits).
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre Inputs and outputs do not overlap.
   * @pre All accessed storage is accessible from the calling execution space.
   * @post cws, tr and s0s[party] form the corresponding party's key.
   * @post Only the described prefix is written, and storage beyond it is untouched.
   */
  __host__ __device__ void Gen(cuda::std::span<int4> cws, In &tr,
      cuda::std::span<const int4, 2> s0s, In a, int4 b_buf, size_t stride = 1);

  /**
   * Evaluate one point using a seed and shared correction words.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared correction words returned by Gen().
   * @param x Query point.
   * @return This party's output share.
   * @pre x is in [0, 2^in_bits).
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], In x);

  /**
   * Evaluate one point from split Cws.
   *
   * Layer l is read from cws[l*stride] and the terminal from cws[in_bits*stride].
   * Levels 0..in_bits-2 read only their seed correction word. The final tree level
   * (in_bits - 1) reads SetLsb(HCW, LCW_0) from its layer entry and LCW_1 from bit
   * in_bits-1 of tr; every other bit of tr is ignored. The terminal
   * cws[in_bits*stride] is the output correction word as Cw::s stores it.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer l at index l*stride and the
   * terminal at index in_bits*stride.
   * @param tr Packed control bits; only bit in_bits-1 is read.
   * @param x Query point.
   * @param stride Positive stride between Cw entries.
   * @return This party's output share.
   * @pre x is in [0, 2^in_bits).
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre cws[in_bits*stride].w & 1 == 0.
   * @pre All accessed storage is accessible from the calling execution space.
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, cuda::std::span<const int4> cws,
      In tr, In x, size_t stride = 1);

  /**
   * Evaluate the full domain on the host using caller-owned storage.
   *
   * par_depth controls OpenMP partitioning for this call.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared correction words returned by Gen().
   * @param ys Output array of at least 2^in_bits elements.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The output element count and byte size fit size_t.
   * @post ys[x] contains this party's output share for x.
   */
  void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[], int par_depth = 0);

  /**
   * Evaluate the full domain from split Cws.
   *
   * Layer l is read from cws[l*stride] and the terminal from cws[in_bits*stride].
   * Levels 0..in_bits-2 read only their seed correction word. The final tree level
   * (in_bits - 1) reads SetLsb(HCW, LCW_0) from its layer entry and LCW_1 from bit
   * in_bits-1 of tr; every other bit of tr is ignored. The terminal
   * cws[in_bits*stride] is the output correction word as Cw::s stores it.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer l at index l*stride and the
   * terminal at index in_bits*stride.
   * @param tr Packed control bits; only bit in_bits-1 is read.
   * @param ys Output array of at least 2^in_bits elements.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre cws[in_bits*stride].w & 1 == 0.
   * @pre The output element count and byte size fit size_t.
   * @post ys[x] contains this party's output share for x.
   */
  void EvalAll(bool b, int4 s0, cuda::std::span<const int4> cws, In tr,
      int4 ys[], int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate the closed input interval [first, last] on the host.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval, inclusive.
   * @param last Last input in the interval, inclusive.
   * @param s0 This party's initial seed.
   * @param cws The in_bits + 1 shared correction words returned by Gen().
   * @param ys Output storage for at least last - first + 1 shares.
   * @param par_depth Nonnegative OpenMP split depth. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre first <= last and both endpoints belong to the input domain.
   * @pre The output count and byte size fit size_t, and used pointers are nonnull.
   * @post ys[j] is this party's scalar share at first + j.
   * Extra output capacity is untouched.
   */
  void EvalRange(bool b, In first, In last, int4 s0, const Cw cws[],
      cuda::std::span<int4> ys, int par_depth = 0);

  /**
   * Evaluate the closed input interval [first, last] from split Cws.
   *
   * Layer l is read from cws[l*stride] and the terminal from cws[in_bits*stride].
   * Levels 0..in_bits-2 read only their seed correction word. The final tree level
   * (in_bits - 1) reads SetLsb(HCW, LCW_0) from its layer entry and LCW_1 from bit
   * in_bits-1 of tr; every other bit of tr is ignored. The terminal
   * cws[in_bits*stride] is the output correction word as Cw::s stores it.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval, inclusive.
   * @param last Last input in the interval, inclusive.
   * @param s0 This party's initial seed.
   * @param cws Correction words, layer l at index l*stride and the
   * terminal at index in_bits*stride.
   * @param tr Packed control bits; only bit in_bits-1 is read.
   * @param ys Output storage for at least last - first + 1 shares.
   * @param par_depth Nonnegative OpenMP split depth. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @pre first <= last and both endpoints belong to the input domain.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre cws[in_bits*stride].w & 1 == 0.
   * @pre The output count and byte size fit size_t, and used pointers are nonnull.
   * @post ys[j] is this party's scalar share at first + j.
   * Extra output capacity is untouched.
   */
  void EvalRange(bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws,
      In tr, cuda::std::span<int4> ys, int par_depth = 0, size_t stride = 1);


  /**
   * Evaluate the closed interval [first, last] on the given CUDA stream.
   * Output ys[j] corresponds to first+j. Extra output capacity is unchanged.
   * Resources and asynchronous error handling follow EvalAllGpu().
   * frontier_depth is relative to each aligned evaluation subtree and is capped
   * to its depth (before HalfTreeDpf's special final layer).
   * Private subdivision keeps the tail at most eight output levels and the
   * block-root depth at most 30, without backend or hardware tuning.
   * Invalid endpoints, buffers, counts or byte sizes return cudaErrorInvalidValue.
   * Grid limits return cudaErrorInvalidConfiguration.
   * The input domain itself need not fit in size_t.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalRangeGpu(bool b, In first, In last, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<int4> ys,
      cudaStream_t stream = 0) const;

  /**
   * Evaluate the closed interval [first, last] on the given CUDA stream from
   * split device arrays. Output ys[j] corresponds to first+j.
   *
   * Layer l is read from cws[l*stride] and the terminal from cws[in_bits*stride].
   * Levels 0..in_bits-2 read only their seed correction word. The final tree level
   * (in_bits - 1) reads SetLsb(HCW, LCW_0) from its layer entry and LCW_1 from bit
   * in_bits-1 of tr; every other bit of tr is ignored. The terminal
   * cws[in_bits*stride] is the output correction word as Cw::s stores it.
   * Resources and asynchronous error handling follow EvalAllGpu().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval, inclusive.
   * @param last Last input in the interval, inclusive.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device correction words, layer l at index l*stride.
   * @param tr Packed control bits; only bit in_bits-1 is read.
   * @param ys Device output storage. Entry x - first receives this party's share at x.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive stride between Cw entries.
   * @return Immediate CUDA status. Invalid endpoints, buffers, strides, counts
   * or byte sizes return cudaErrorInvalidValue. Grid limits return
   * cudaErrorInvalidConfiguration.
   * @pre stride > 0 and cws holds at least in_bits*stride+1 elements.
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalRangeGpu(bool b, In first, In last, int4 s0,
      cuda::std::span<const int4> cws, In tr, cuda::std::span<int4> ys,
      cudaStream_t stream = 0, size_t stride = 1) const;

  /**
   * Launch full-domain CUDA evaluation from this host-accessible instance.
   *
   * Hash values and the instance seed are passed as kernel arguments. The caller
   * prepares device buffers and any device resources referenced by Hash, and
   * keeps them alive until stream completes. Hash and output operations must
   * support concurrent device execution. No allocation, upload or synchronization
   * is performed. The scheme itself must be accessible on the host.
   * @tparam frontier_depth Frontier depth, capped at in_bits - 1, before the special final tree level. Defaults to 16.
   * @tparam threads_per_block A power of two in [1,1024], capped at the frontier width.
   * Defaults to 256. Block-root depth is derived from the effective frontier and
   * block size. At most eight output levels may remain below the frontier, and
   * the block-root depth must be at most 30. These are fixed correctness caps.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device storage containing at least in_bits + 1 correction words.
   * @param ys Device storage containing at least 2^in_bits output blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @return Immediate CUDA launch status. Invalid buffer dimensions, null used
   * buffers, or unaddressable byte sizes return cudaErrorInvalidValue. Execution
   * remains asynchronous; the caller observes completion errors through stream.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const Cw> cws,
      cuda::std::span<int4> ys, cudaStream_t stream = 0) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Launch full-domain CUDA evaluation from split device arrays.
   *
   * Layer l is read from cws[l*stride] and the terminal from cws[in_bits*stride].
   * Levels 0..in_bits-2 read only their seed correction word. The final tree level
   * (in_bits - 1) reads SetLsb(HCW, LCW_0) from its layer entry and LCW_1 from bit
   * in_bits-1 of tr; every other bit of tr is ignored. The terminal
   * cws[in_bits*stride] is the output correction word as Cw::s stores it.
   * Hash values, resource rules and tuning rules match EvalAllGpu().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device correction words, layer l at index l*stride.
   * @param tr Packed control bits; only bit in_bits-1 is read.
   * @param ys Device storage containing at least 2^in_bits output blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive stride between Cw entries.
   * @return Immediate CUDA launch status. Invalid buffer dimensions, strides,
   * null used buffers, or unaddressable byte sizes return cudaErrorInvalidValue.
   * Execution remains asynchronous; the caller observes completion errors
   * through stream.
   * @pre stride > 0 and cws holds at least in_bits*stride+1 elements.
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const int4> cws,
      In tr, cuda::std::span<int4> ys, cudaStream_t stream = 0, size_t stride = 1) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Launch full-domain evaluation for s0s.size() keys with this configuration.
   *
   * Uses the same execution, resource and tuning rules as EvalAllGpu().
   * Seeds, correction words and outputs are device-accessible buffers. Cws
   * and outputs are key-major rows, with in_bits + 1 correction words and
   * 2^in_bits output blocks per key. The key count comes from s0s.size().
   * @param b Party index shared by the batch.
   * @param s0s Device storage containing one seed per key.
   * @param cws Flat device rows of Cws.
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
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Launch full-domain CUDA evaluation for s0s.size() keys from split device arrays.
   *
   * The batch uses the level-major layout: key k layer l is read from
   * cws[l*stride + k], key k's packed control is tr[k], and key k's terminal is
   * cws[in_bits*stride + k]. Levels 0..in_bits-2 read only their seed correction word,
   * the final tree level (in_bits - 1) reads SetLsb(HCW, LCW_0) from its layer
   * entry and LCW_1 from bit in_bits-1 of tr[k], and every other bit of tr[k] is
   * ignored.
   * stride selects the layout. 0 selects the dense level-major layout, in
   * which the stride is the key count of this call. Any other value is a
   * positive stride of at least s0s.size(). Resource and tuning rules match
   * EvalAllGpu(). The batch shares one party index and hash configuration.
   * Every key produces the complete domain.
   *
   * @param b Party index shared by the batch.
   * @param s0s Device storage containing one seed per key.
   * @param cws Device level-major correction words.
   * @param tr Device packed control bits, one value per key.
   * @param ys Device key-major rows of 2^in_bits output blocks per key.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride 0 for the dense level-major layout, where the stride is the
   * key count of this call, or a positive element stride >= s0s.size().
   * @return Immediate CUDA status. Empty batches succeed without reading buffers.
   * Invalid dimensions, null used buffers, strides, or unaddressable counts and
   * bytes return cudaErrorInvalidValue. Oversized grids return
   * cudaErrorInvalidConfiguration.
   * @pre stride is 0 or at least s0s.size().
   * @pre cws holds at least in_bits*stride + s0s.size() elements, where a
   * stride of 0 counts as s0s.size().
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpuBatch(bool b, cuda::std::span<const int4> s0s,
      cuda::std::span<const int4> cws, cuda::std::span<const In> tr,
      cuda::std::span<int4> ys, cudaStream_t stream = 0,
      size_t stride = 0) const
    requires(in_bits < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  template <int frontier_depth, int threads_per_block>
  struct GpuLaunchShape;

  static __host__ __device__ void ExpandNode(Hash &hash, int4 instance_seed,
      int4 node, int4 cw, int4 &left, int4 &right);
  static __host__ __device__ int4 PathStep(Hash &hash, int4 instance_seed,
      int4 node, int4 cw, bool take_right);
  template <typename OutputCw>
  static __host__ __device__ int4 ConvertLeaf(bool party, Hash &hash,
      int4 instance_seed, int4 node, bool take_right, const Cw &last_cw,
      OutputCw &output_cw);
  template <typename OutputCw>
  static __host__ __device__ int4 ConvertHashedLeaf(bool party, int4 hashed,
      bool node_control, bool take_right, const Cw &last_cw,
      OutputCw &output_cw);
  template <typename OutputCw>
  static __host__ __device__ void ConvertLeafPair(bool party, Hash &hash,
      int4 instance_seed, int4 node, const Cw &last_cw,
      OutputCw &output_cw, int4 &left, int4 &right);

  // Split Cws view over the caller's array. The host methods and the
  // split GPU kernels share this one view; see detail::HalfTreeDpfCwsView.
  using CwsView = detail::HalfTreeDpfCwsView<Cw, In, in_bits>;

  // Writes Cw slots in the key array layout.
  struct AosWriter {
    Cw *cws;

    __host__ __device__ void Write(int i, int4 s_cw) const {
      cws[i] = {s_cw, false};
    }

    __host__ __device__ void WriteLast(int4 s_cw, bool extra_cw) const {
      // Both 16B halves of the final 32B slot are written so NVCC emits unmasked
      // stores: {s_cw, extra_cw} alone leaves the padding undefined, and the
      // partial-sector write forces a write-allocate round trip on GPU
      // (measured ~2x Gen time for the same pattern).
      int4 *cw_slot = reinterpret_cast<int4 *>(&cws[in_bits - 1]);
      cw_slot[0] = s_cw;
      cw_slot[1] = make_int4(extra_cw, 0, 0, 0);
    }

    __host__ __device__ void Finish(int4 terminal) const {
      cws[in_bits] = {terminal, false};
    }
  };

  // Writes the strided components and accumulates the packed control bit locally.
  struct CwsWriter {
    int4 *cws;
    size_t stride;
    In &tr;
    In packed_tr = 0;

    __host__ __device__ void Write(int i, int4 s_cw) {
      cws[static_cast<size_t>(i) * stride] = s_cw;
    }

    __host__ __device__ void WriteLast(int4 s_cw, bool extra_cw) {
      cws[static_cast<size_t>(in_bits - 1) * stride] = s_cw;
      packed_tr |= static_cast<In>(extra_cw) << (in_bits - 1);
    }

    __host__ __device__ void Finish(int4 terminal) {
      cws[static_cast<size_t>(in_bits) * stride] = terminal;
      tr = packed_tr;
    }
  };

  // Shared Cw generation loop. The writer is passed by value and owns
  // only output access, so the existing algorithm and backend use stay unchanged.
  template <typename Writer>
  __host__ __device__ void Generate(Writer writer, const int4 *s0s, In a, int4 b_buf);

  // Shared point evaluation over Cw arrays and split Cws views.
  template <typename Cws>
  __host__ __device__ int4 EvalWalk(bool b, int4 s0, Cws cws, In x);

  // Shared full-domain evaluation over Cw arrays and split Cws views.
  template <typename Cws>
  void EvalAllCws(bool b, int4 s0, Cws cws, int4 ys[], int par_depth);

  // Shared interval evaluation. Cws is a Cw pointer on the array form
  // and a CwsView on the split form.
  template <typename Cws>
  void EvalRangeCws(bool b, In first, In last, int4 s0, Cws cws,
      cuda::std::span<int4> ys, int par_depth);

  Hash hash_;
  int4 instance_seed_;

  template <typename Cws>
  void EvalRangeTree(bool b, int4 node, Cws cws, int4 ys[],
      In first, In last, In output_first, In low, In high, int level);

  // EvalAll() shape: a scalar prefix of kBatchPrefix levels grows the frontier
  // to 8 nodes and detail::HashBatch() hashes 8 parents per level. The tail fuses the
  // last-level conversion: each group of 8 parents emits its 16 leaves
  // directly, so no backward in-place pass over ys is needed.
  static constexpr int kBatchPrefix = 3;

  // Expands w scalar frontier nodes by one level with cws[lv].
  // The children overwrite the frontier in place: iteration is in reverse so
  // the writes at 2j, 2j+1 never clobber nodes still to be read.
  template <typename Cws>
  void ExpandFrontier(int4 frontier[], size_t w, Cws cws, int lv);

  // Branchless XOR of a correction word into 16B under a 0/-1 mask from the control bit.
  static int4 MaskXor(int4 val, int4 cw, int m);

  // Full-domain evaluation of the subtree rooted at node at level i through an
  // 8-wide batched frontier: a kBatchPrefix-level scalar prefix, batched
  // expansions, and a fused tail that converts each group of 8 parents to its
  // 16 leaves. Requires in_bits - i >= kBatchPrefix + 1.
  template <typename Cws>
  void EvalBatched8(bool b, int4 node, Cws cws, Group ocw_group, int4 ys[],
      size_t l, int i);

  // Evaluates a group of 8 consecutive frontier nodes at level i, covering the leaves
  // [l, l + 8 << (in_bits - i)) in order. Requires in_bits - i >= 1.
  template <typename Cws>
  void EvalBatched8Group(bool b, const int4 (&frontier)[8], Cws cws,
      Group ocw_group, int4 ys[], size_t l, int i);
};

// -----------------------------------------------------------------------------

namespace gpu::detail {

// Hold the scheme behind a tag so wide Group non-type template parameters stay
// out of CUDA host stub signatures.
template <int n, typename GroupTag, typename Hash, typename In>
struct HalfTreeGpuKernelDescriptor {
  using Scheme = HalfTreeDpf<n, typename GroupTag::type, Hash, In>;
  using Backend = Hash;
};

template <typename Scheme>
struct HalfTreeGpuTraits;

#if defined(__CUDACC__)

// One block-local traversal serves the Cw and split kernels. Cws is a Cw
// pointer on the array path and a kernel-boundary split view on the split path,
// so the traversal body does not depend on the layout.
template <typename Scheme, int frontier_depth, int block_root_depth, bool range = false,
    typename Cws>
__device__ __forceinline__ void HalfTreeEvalAllBlock(bool b, int4 s0,
    Cws cws, int4 *ys, int4 instance_seed,
    typename HalfTreeGpuTraits<Scheme>::Backend &hash, int subtree_block_index,
    int64_t thread_index, int4 *shared_frontier,
    __uint128_t first = 0, int subtree_depth = HalfTreeGpuTraits<Scheme>::kOutputBits,
    int range_frontier = frontier_depth, int range_block_root = block_root_depth) {
  using Traits = HalfTreeGpuTraits<Scheme>;
  constexpr int kTailCapacity = range ? 8 : Traits::kOutputBits - frontier_depth;
  const int start_depth = range ? Traits::kOutputBits - subtree_depth : 0;
  const int frontier = range ? start_depth + range_frontier : frontier_depth;
  const int root_depth = range ? start_depth + range_block_root : block_root_depth;
  const int tail_depth = Traits::kOutputBits - frontier;
  const int block_subtree_depth = frontier - root_depth;
  if (threadIdx.x == 0) {
    int4 node = util::SetLsb(s0, b);
#pragma unroll
    for (int level = 0; level < root_depth; ++level) {
      bool take_right = level < start_depth
          ? ((first >> (Traits::kOutputBits - 1 - level)) & 1) != 0
          : ((subtree_block_index >> (root_depth - 1 - level)) & 1) != 0;
      // Preserve the former by-value helper's backend lifetime per path step.
      auto node_hash = hash;
      node = fss::detail::SchemeAccess::HalfTreePathStep<Scheme>(
          node_hash, instance_seed, node, cws[level].s, take_right);
    }
    shared_frontier[0] = node;
  }
  __syncthreads();

#pragma unroll
  for (int level = 0; level < block_subtree_depth; ++level) {
    const bool active = threadIdx.x < (1 << (level + 1));
    int4 node = {0, 0, 0, 0};
    if (active) node = shared_frontier[threadIdx.x >> 1];
    // Finish all parent reads before children overwrite the same frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      auto node_hash = hash;
      fss::detail::SchemeAccess::HalfTreeExpandNode<Scheme>(node_hash, instance_seed,
          node, cws[root_depth + level].s, left, right);
      shared_frontier[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  if constexpr (range && !Traits::kPacked) {
    int4 pending[8];
    int4 node = shared_frontier[threadIdx.x];
    int level = frontier;
    const int count = 1 << (Traits::kExpansionDepth - frontier);
    const size_t offset = static_cast<size_t>(thread_index) << tail_depth;
    using OutputGroup = typename Traits::OutputGroup;
    auto cw = OutputGroup::From(cws[Traits::kOutputBits].s);
    for (int index = 0; index < count; ++index) {
      while (level < Traits::kExpansionDepth) {
        int4 left, right;
        fss::detail::SchemeAccess::HalfTreeExpandNode<Scheme>(hash, instance_seed,
            node, cws[level].s, left, right);
        pending[level - frontier] = right;
        node = left;
        ++level;
      }
      int4 left, right;
      fss::detail::SchemeAccess::HalfTreeConvertLeafPair<Scheme>(b, hash,
          instance_seed, node, cws[Traits::kExpansionDepth], cw, left, right);
      ys[offset + 2 * index] = left;
      ys[offset + 2 * index + 1] = right;
      if (index + 1 < count) {
        int next = index + 1;
        level = Traits::kExpansionDepth - 1;
        while ((next & 1) == 0) {
          --level;
          next >>= 1;
        }
        node = pending[level - frontier];
        ++level;
      }
    }
  } else {
    int4 nodes[1 << kTailCapacity];
    nodes[0] = shared_frontier[threadIdx.x];
    int count = 1;
#pragma unroll
    for (int level = frontier; level < Traits::kExpansionDepth; ++level) {
#pragma unroll
      for (int index = count - 1; index >= 0; --index) {
        int4 left, right;
        fss::detail::SchemeAccess::HalfTreeExpandNode<Scheme>(
            hash, instance_seed, nodes[index], cws[level].s, left, right);
        nodes[2 * index] = left;
        nodes[2 * index + 1] = right;
      }
      count *= 2;
    }

    const size_t output_offset = static_cast<size_t>(thread_index) << tail_depth;
    if constexpr (Traits::kPacked) {
#pragma unroll
      for (int index = 0; index < count; ++index) {
        ys[output_offset + index] = fss::detail::SchemeAccess::PackedConvertLeaf<Scheme>(
            hash, instance_seed, nodes[index], cws[Traits::kExpansionDepth].s);
      }
    } else {
      using OutputGroup = typename Traits::OutputGroup;
      auto output_cw = OutputGroup::From(cws[Traits::kOutputBits].s);
#pragma unroll
      for (int index = 0; index < count; ++index) {
        int4 left, right;
        fss::detail::SchemeAccess::HalfTreeConvertLeafPair<Scheme>(b, hash, instance_seed,
            nodes[index], cws[Traits::kExpansionDepth], output_cw, left, right);
        ys[output_offset + 2 * index] = left;
        ys[output_offset + 2 * index + 1] = right;
      }
    }
  }
}

template <typename Descriptor, int threads_per_block>
__global__ void HalfTreeDpfEvalRangeKernel(bool b, int4 s0,
    const void *cws_buffer, int4 *ys, __uint128_t first,
    int depth, int frontier, int root, int4 instance_seed, typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  const auto *cws = static_cast<const typename Scheme::Cw *>(cws_buffer);
  __shared__ int4 states[threads_per_block];
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (depth == 0) {
    int4 node = util::SetLsb(s0, b);
    for (int level = 0; level < HalfTreeGpuTraits<Scheme>::kExpansionDepth; ++level) {
      auto node_hash = hash;
      node = fss::detail::SchemeAccess::HalfTreePathStep<Scheme>(node_hash,
          instance_seed, node, cws[level].s,
          ((first >> (HalfTreeGpuTraits<Scheme>::kOutputBits - 1 - level)) & 1) != 0);
    }
    int4 left, right;
    using OutputGroup = typename HalfTreeGpuTraits<Scheme>::OutputGroup;
    auto cw = OutputGroup::From(cws[HalfTreeGpuTraits<Scheme>::kOutputBits].s);
    fss::detail::SchemeAccess::HalfTreeConvertLeafPair<Scheme>(b, hash, instance_seed,
        node, cws[HalfTreeGpuTraits<Scheme>::kExpansionDepth], cw, left, right);
    ys[0] = (first & 1) ? right : left;
    return;
  }
  HalfTreeEvalAllBlock<Scheme, 0, 0, true>(b, s0, cws, ys, instance_seed, hash,
      blockIdx.x, thread_index, states, first, depth, frontier, root);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void HalfTreeEvalAllKernel(bool b, int4 s0, const void *cws_buffer,
    int4 *ys, int4 instance_seed, typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  const auto *cws = static_cast<const typename Scheme::Cw *>(cws_buffer);
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  HalfTreeEvalAllBlock<Scheme, frontier_depth, block_root_depth>(
      b, s0, cws, ys, instance_seed, hash, blockIdx.x, thread_index, shared_frontier);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void HalfTreeEvalAllBatchKernel(bool b, const int4 *s0s,
    const void *cws_buffer, int4 *ys, int4 instance_seed,
    typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  const auto *cws = static_cast<const typename Scheme::Cw *>(cws_buffer);
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  constexpr int kCwCount = HalfTreeGpuTraits<Scheme>::kExpansionDepth +
      (HalfTreeGpuTraits<Scheme>::kPacked ? 1 : 2);
  constexpr size_t kOutputsPerKey = size_t{1} << HalfTreeGpuTraits<Scheme>::kOutputBits;
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  const auto *key_cws = cws + static_cast<size_t>(key_index) * kCwCount;
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * kOutputsPerKey;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  HalfTreeEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      key_cws, key_outputs, instance_seed, hash, subtree_block_index,
      thread_index, shared_frontier);
}

// Split-layout counterparts of the three kernels above. The Cw buffer
// arrives type-erased and the boundary view reconstructs the per-layer Cw
// entries, so the shared traversal stays identical on both paths.
template <typename Descriptor, int threads_per_block>
__global__ void HalfTreeSplitDpfEvalRangeKernel(bool b, int4 s0,
    const void *cws_buffer,
    typename HalfTreeGpuTraits<typename Descriptor::Scheme>::Input packed_tr, size_t stride,
    int4 *ys, __uint128_t first, int depth, int frontier, int root,
    int4 instance_seed, typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  using Traits = HalfTreeGpuTraits<Scheme>;
  using Cws = fss::detail::HalfTreeDpfCwsView<typename Scheme::Cw,
      typename Traits::Input, Traits::kOutputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buffer);
  const Cws view{cws, packed_tr, stride};
  __shared__ int4 states[threads_per_block];
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (depth == 0) {
    int4 node = util::SetLsb(s0, b);
    for (int level = 0; level < HalfTreeGpuTraits<Scheme>::kExpansionDepth; ++level) {
      auto node_hash = hash;
      node = fss::detail::SchemeAccess::HalfTreePathStep<Scheme>(node_hash,
          instance_seed, node, view[level].s,
          ((first >> (HalfTreeGpuTraits<Scheme>::kOutputBits - 1 - level)) & 1) != 0);
    }
    int4 left, right;
    using OutputGroup = typename HalfTreeGpuTraits<Scheme>::OutputGroup;
    auto cw = OutputGroup::From(view[HalfTreeGpuTraits<Scheme>::kOutputBits].s);
    fss::detail::SchemeAccess::HalfTreeConvertLeafPair<Scheme>(b, hash, instance_seed,
        node, view[HalfTreeGpuTraits<Scheme>::kExpansionDepth], cw, left, right);
    ys[0] = (first & 1) ? right : left;
    return;
  }
  HalfTreeEvalAllBlock<Scheme, 0, 0, true>(b, s0, view, ys, instance_seed, hash,
      blockIdx.x, thread_index, states, first, depth, frontier, root);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void HalfTreeSplitEvalAllKernel(bool b, int4 s0, const void *cws_buffer,
    typename HalfTreeGpuTraits<typename Descriptor::Scheme>::Input packed_tr, size_t stride,
    int4 *ys, int4 instance_seed, typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  using Traits = HalfTreeGpuTraits<Scheme>;
  using Cws = fss::detail::HalfTreeDpfCwsView<typename Scheme::Cw,
      typename Traits::Input, Traits::kOutputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buffer);
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  HalfTreeEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0,
      Cws{cws, packed_tr, stride}, ys, instance_seed, hash, blockIdx.x,
      thread_index, shared_frontier);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void HalfTreeSplitEvalAllBatchKernel(bool b, const int4 *s0s,
    const void *cws_buffer,
    const typename HalfTreeGpuTraits<typename Descriptor::Scheme>::Input *trs, size_t stride,
    int4 *ys, int4 instance_seed, typename Descriptor::Backend hash) {
  using Scheme = typename Descriptor::Scheme;
  using Traits = HalfTreeGpuTraits<Scheme>;
  using Cws = fss::detail::HalfTreeDpfCwsView<typename Scheme::Cw,
      typename Traits::Input, Traits::kOutputBits>;
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
      Cws{cws + key_index, trs[key_index], stride}, key_outputs, instance_seed,
      hash, subtree_block_index, thread_index, shared_frontier);
}

#endif  // defined(__CUDACC__)

}  // namespace gpu::detail

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline HalfTreeDpf<in_bits, Group, Hash, In>::HalfTreeDpf(Hash hash, int4 instance_seed)
    : hash_(hash), instance_seed_(instance_seed) {}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline void HalfTreeDpf<in_bits, Group, Hash, In>::Gen(
    Cw cws[], const int4 s0s[2], In a, int4 b_buf) {
  Generate(AosWriter{cws}, s0s, a, b_buf);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline void HalfTreeDpf<in_bits, Group, Hash, In>::Gen(
    cuda::std::span<int4> cws, In &tr, cuda::std::span<const int4, 2> s0s,
    In a, int4 b_buf, size_t stride) {
  assert(stride > 0);
  [[maybe_unused]] constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  assert(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits));
  [[maybe_unused]] const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  assert(cws.size() >= required);
  Generate(CwsWriter{cws.data(), stride, tr}, s0s.data(), a, b_buf);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Writer>
__host__ __device__ inline void HalfTreeDpf<in_bits, Group, Hash, In>::Generate(
    Writer writer, const int4 *s0s, In a, int4 b_buf) {
  b_buf = util::SetLsb(b_buf, false);

  // Initialize: node0 has t=0, node1 has t=1
  int4 node0 = util::SetLsb(s0s[0], false);
  int4 node1 = util::SetLsb(s0s[1], true);
  int4 delta = util::Xor(node0, node1);  // LSB = 0^1 = 1

  // Levels 1 to n-1 (index i = 0 to in_bits-2)
  for (int i = 0; i < in_bits - 1; ++i) {
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

    // delta = node0 ^ node1 for next level
    delta = util::Xor(node0, node1);
  }

  // Level n (last level, index i = in_bits-1)
  {
    bool a_n = (a >> 0) & 1;  // last bit of alpha
    bool t0 = util::GetLsb(node0);
    bool t1 = util::GetLsb(node1);

    // Hash with sigma in {0, 1}
    int4 h0_0 = hash_.Hash(util::Xor(instance_seed_, util::SetLsb(node0, false)));
    int4 h0_1 = hash_.Hash(util::Xor(instance_seed_, util::SetLsb(node0, true)));
    int4 h1_0 = hash_.Hash(util::Xor(instance_seed_, util::SetLsb(node1, false)));
    int4 h1_1 = hash_.Hash(util::Xor(instance_seed_, util::SetLsb(node1, true)));

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

    // Store CW_n. The writer covers the whole final slot, see AosWriter::WriteLast().
    writer.WriteLast(util::SetLsb(HCW, LCW_0), LCW_1);

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

    // Apply the CW: if t_b: leaf_b ^= SetLsb(HCW, lcw_an)
    bool lcw_an = a_n ? LCW_1 : LCW_0;
    int4 leaf_cw = util::SetLsb(HCW, lcw_an);
    if (t0) leaf0 = util::Xor(leaf0, leaf_cw);
    if (t1) leaf1 = util::Xor(leaf1, leaf_cw);

    // Output CW: v_cw = Group::From(b_buf) + (-Group::From(SetLsb(leaf0,false))) + Group::From(SetLsb(leaf1,false))
    auto v_cw =
        Group::From(b_buf) + (-Group::From(util::SetLsb(leaf0, false))) + Group::From(util::SetLsb(leaf1, false));
    if (util::GetLsb(leaf1)) v_cw = -v_cw;
    writer.Finish(v_cw.Into());
  }
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::Eval(
    bool b, int4 s0, const Cw cws[], In x) {
  return EvalWalk(b, s0, cws, x);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::Eval(
    bool b, int4 s0, cuda::std::span<const int4> cws, In tr, In x, size_t stride) {
  assert(stride > 0);
  [[maybe_unused]] constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  assert(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits));
  [[maybe_unused]] const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  assert(cws.size() >= required);
  const int4 terminal = cws[static_cast<size_t>(in_bits) * stride];
  assert((terminal.w & 1) == 0);
  return EvalWalk(b, s0, CwsView{cws.data(), tr, stride}, x);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
__host__ __device__ inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::EvalWalk(
    bool b, int4 s0, Cws cws, In x) {
  int4 node = util::SetLsb(s0, b);

  for (int i = 0; i < in_bits - 1; ++i) {
    bool take_right = (x >> (in_bits - 1 - i)) & 1;
    node = PathStep(hash_, instance_seed_, node, cws[i].s, take_right);
  }
  // The split view returns the final entries by value, so the shared leaf
  // conversion takes them as locals.
  const Cw last_cw = cws[in_bits - 1];
  const int4 terminal = cws[in_bits].s;
  return ConvertLeaf(b, hash_, instance_seed_, node, (x & 1) != 0,
      last_cw, terminal);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalAll(
    bool b, int4 s0, const Cw cws[], int4 ys[], int par_depth) {
  EvalAllCws(b, s0, cws, ys, par_depth);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalAll(bool b, int4 s0,
    cuda::std::span<const int4> cws, In tr, int4 ys[], int par_depth, size_t stride) {
  assert(stride > 0);
  [[maybe_unused]] constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  assert(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits));
  [[maybe_unused]] const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  assert(cws.size() >= required);
  const int4 terminal = cws[static_cast<size_t>(in_bits) * stride];
  assert((terminal.w & 1) == 0);
  EvalAllCws(b, s0, CwsView{cws.data(), tr, stride}, ys, par_depth);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalAllCws(
    bool b, int4 s0, Cws cws, int4 ys[], int par_depth) {
  assert(par_depth >= 0);
  static_assert(detail::FitsFullDomain<int4, in_bits>(),
      "full-domain output count and byte size must fit in size_t");

  if constexpr (in_bits < kBatchPrefix + 1) {
    // Domains too small to fill an 8-wide frontier evaluate per input; the
    // point walk is cheaper than the batched machinery at this size.
    constexpr size_t n = size_t{1} << in_bits;
    fss::detail::ParallelFor(par_depth, n,
        [&](size_t x) { ys[x] = EvalWalk(b, s0, cws, static_cast<In>(x)); });
    return;
  }

  int4 node = util::SetLsb(s0, b);

  int par_depth_ = fss::detail::ResolveParDepth(par_depth, in_bits);

  int split_depth = par_depth_;
  if (split_depth > in_bits - kBatchPrefix - 1) split_depth = in_bits - kBatchPrefix - 1;
  if (split_depth > 0) {
    // Split at a frontier of 2 ** split_depth subtree roots instead of per-node tasks,
    // so the sequential batched walk stays the fast default path.
    size_t num_tasks = size_t{1} << split_depth;
    std::vector<int4> frontier(num_tasks);
    frontier[0] = node;
    for (int lv = 0; lv < split_depth; ++lv) ExpandFrontier(frontier.data(), size_t{1} << lv, cws, lv);
    size_t span = size_t{1} << (in_bits - split_depth);
    fss::detail::ParallelFor(par_depth_, num_tasks, [&](size_t task) {
      EvalBatched8(b, frontier[task], cws, Group::From(cws[in_bits].s), ys, task * span, split_depth);
    });
  } else {
    EvalBatched8(b, node, cws, Group::From(cws[in_bits].s), ys, 0, 0);
  }
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalRange(
    bool b, In first, In last, int4 s0, const Cw cws[],
    cuda::std::span<int4> ys, int par_depth) {
  assert(cws != nullptr);
  EvalRangeCws(b, first, last, s0, cws, ys, par_depth);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalRange(bool b, In first, In last,
    int4 s0, cuda::std::span<const int4> cws, In tr, cuda::std::span<int4> ys,
    int par_depth, size_t stride) {
  assert(stride > 0);
  [[maybe_unused]] constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  assert(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits));
  [[maybe_unused]] const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  assert(cws.size() >= required);
  assert(cws.data() != nullptr);
  assert(ys.data() != nullptr);
  const int4 terminal = cws[static_cast<size_t>(in_bits) * stride];
  assert((terminal.w & 1) == 0);
  EvalRangeCws(b, first, last, s0,
      CwsView{cws.data(), tr, stride}, ys, par_depth);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalRangeCws(
    bool b, In first, In last, int4 s0, Cws cws,
    cuda::std::span<int4> ys, int par_depth) {
  assert(par_depth >= 0);
  In domain_last = static_cast<In>(~In{0});
  if constexpr (in_bits < sizeof(In) * 8) {
    domain_last = static_cast<In>((In{1} << in_bits) - 1);
  }
  assert(first <= last && last <= domain_last);
  const __uint128_t distance = static_cast<__uint128_t>(last) - first;
  assert(distance < std::numeric_limits<size_t>::max() / sizeof(int4));
  const size_t count = static_cast<size_t>(distance) + 1;
  assert(ys.size() >= count && ys.data() != nullptr);

  const size_t partitions = detail::ResolveParPartitions(par_depth, count);
  detail::ParallelFor(par_depth, partitions, [&](size_t partition) {
    const size_t begin = partition * (count / partitions) +
        (partition < count % partitions ? partition : count % partitions);
    const size_t end = begin + count / partitions + (partition < count % partitions);
    EvalRangeTree(b, util::SetLsb(s0, b), cws, ys.data(),
        static_cast<In>(first + begin), static_cast<In>(first + (end - 1)),
        first, In{0}, domain_last, 0);
  });
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalRangeTree(
    bool b, int4 node, Cws cws, int4 ys[], In first, In last,
    In output_first, In low, In high, int level) {
  if (level == in_bits - 1) {
    // The split view returns the final entries by value, so the shared leaf
    // conversion takes them as locals.
    const Cw last_cw = cws[level];
    const int4 terminal = cws[in_bits].s;
    if (first <= low) {
      ys[static_cast<size_t>(low - output_first)] = ConvertLeaf(
          b, hash_, instance_seed_, node, false, last_cw, terminal);
    }
    if (last >= high) {
      ys[static_cast<size_t>(high - output_first)] = ConvertLeaf(
          b, hash_, instance_seed_, node, true, last_cw, terminal);
    }
    return;
  }
  const In middle = static_cast<In>(low + ((high - low) >> 1));
  int4 left, right;
  ExpandNode(hash_, instance_seed_, node, cws[level].s, left, right);
  if (first <= middle) {
    EvalRangeTree(b, left, cws, ys, first, last, output_first, low, middle, level + 1);
  }
  if (last > middle) {
    EvalRangeTree(b, right, cws, ys, first, last, output_first,
        static_cast<In>(middle + 1), high, level + 1);
  }
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeDpf<in_bits, Group, Hash, In>::ExpandFrontier(
    int4 frontier[], size_t w, Cws cws, int lv) {
  int4 cw = cws[lv].s;
  for (size_t j = w; j-- > 0;) {
    int4 node = frontier[j];
    int4 left, right;
    ExpandNode(hash_, instance_seed_, node, cw, left, right);
    frontier[2 * j] = left;
    frontier[2 * j + 1] = right;
  }
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::MaskXor(int4 val, int4 cw, int m) {
  return {val.x ^ (cw.x & m), val.y ^ (cw.y & m), val.z ^ (cw.z & m), val.w ^ (cw.w & m)};
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalBatched8(
    bool b, int4 node, Cws cws, Group ocw_group, int4 ys[], size_t l, int i) {
  int4 frontier[8];
  frontier[0] = node;
  for (int lv = i; lv < i + kBatchPrefix; ++lv) ExpandFrontier(frontier, size_t{1} << (lv - i), cws, lv);
  EvalBatched8Group(b, frontier, cws, ocw_group, ys, l, i + kBatchPrefix);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeDpf<in_bits, Group, Hash, In>::EvalBatched8Group(
    bool b, const int4 (&frontier)[8], Cws cws, Group ocw_group, int4 ys[],
    size_t l, int i) {
  if (i == in_bits - 1) {
    // 8 parents -> 16 leaves with two batches of eight hashes (sigma = 0, 1).
    int4 hcw = util::SetLsb(cws[in_bits - 1].s, false);
    int lcw_0 = util::GetLsb(cws[in_bits - 1].s);
    int lcw_1 = cws[in_bits - 1].extra;

    cuda::std::array<int4, 8> seeds0;
    cuda::std::array<int4, 8> seeds1;
    int ts[8];
    for (int j = 0; j < 8; ++j) {
      ts[j] = static_cast<int>(util::GetLsb(frontier[j]));
      seeds0[j] = util::Xor(instance_seed_, util::SetLsb(frontier[j], false));
      seeds1[j] = util::Xor(instance_seed_, util::SetLsb(frontier[j], true));
    }
    auto h0s = detail::HashBatch(hash_, seeds0);
    auto h1s = detail::HashBatch(hash_, seeds1);
    for (int j = 0; j < 8; ++j) {
      int m = -ts[j];
      int4 h0 = h0s[j];
      int4 high0 = MaskXor(h0, hcw, m);
      high0.w &= ~1;
      int low0 = (h0.w & 1) ^ (ts[j] & lcw_0);
      int4 h1 = h1s[j];
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
    seeds[j] = util::Xor(instance_seed_, frontier[j]);
  }
  auto out = detail::HashBatch(hash_, seeds);
  // The 16 children interleave in leaf order: node j's children land at 2j, 2j+1 of the
  // next level. De-interleave them into two consecutive groups of 8.
  int4 low[8];
  int4 high[8];
  for (int j = 0; j < 4; ++j) {
    int m = -ts[j];
    int4 left = MaskXor(out[j], cw, m);
    low[2 * j] = left;
    low[2 * j + 1] = util::Xor(left, frontier[j]);
    int m4 = -ts[j + 4];
    int4 left4 = MaskXor(out[j + 4], cw, m4);
    high[2 * j] = left4;
    high[2 * j + 1] = util::Xor(left4, frontier[j + 4]);
  }

  size_t half = size_t{8} << (in_bits - i - 1);
  EvalBatched8Group(b, low, cws, ocw_group, ys, l, i + 1);
  EvalBatched8Group(b, high, cws, ocw_group, ys, l + half, i + 1);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline void HalfTreeDpf<in_bits, Group, Hash, In>::ExpandNode(
    Hash &hash, int4 instance_seed, int4 node, int4 cw, int4 &left, int4 &right) {
  int4 hashed = hash.Hash(util::Xor(instance_seed, node));
  const int4 zero = {0, 0, 0, 0};
  left = util::Xor(hashed, util::GetLsb(node) ? cw : zero);
  right = util::Xor(left, node);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
__host__ __device__ inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::PathStep(
    Hash &hash, int4 instance_seed, int4 node, int4 cw, bool take_right) {
  const int4 zero = {0, 0, 0, 0};
  int4 hashed = hash.Hash(util::Xor(instance_seed, node));
  return util::Xor(util::Xor(hashed, take_right ? node : zero),
      util::GetLsb(node) ? cw : zero);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename OutputCw>
__host__ __device__ inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::ConvertLeaf(bool party, Hash &hash,
    int4 instance_seed, int4 node, bool take_right, const Cw &last_cw,
    OutputCw &output_cw) {
  int4 hashed = hash.Hash(util::Xor(instance_seed, util::SetLsb(node, take_right)));
  return ConvertHashedLeaf(party, hashed, util::GetLsb(node), take_right,
      last_cw, output_cw);
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename OutputCw>
__host__ __device__ inline int4 HalfTreeDpf<in_bits, Group, Hash, In>::ConvertHashedLeaf(bool party, int4 hashed,
    bool node_control, bool take_right, const Cw &last_cw,
    OutputCw &output_cw) {
  int4 high = util::SetLsb(hashed, false);
  bool low = util::GetLsb(hashed);
  if (node_control) {
    high = util::Xor(high, util::SetLsb(last_cw.s, false));
    low ^= take_right ? last_cw.extra : util::GetLsb(last_cw.s);
  }
  auto value = Group::From(high);
  if (low) {
    if constexpr (std::is_same_v<std::remove_cv_t<OutputCw>, int4>) {
      value = value + Group::From(output_cw);
    } else {
      value = value + output_cw;
    }
  }
  if (party) value = -value;
  return value.Into();
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <typename OutputCw>
__host__ __device__ inline void HalfTreeDpf<in_bits, Group, Hash, In>::ConvertLeafPair(bool party, Hash &hash,
    int4 instance_seed, int4 node, const Cw &last_cw,
    OutputCw &output_cw, int4 &left, int4 &right) {
  int4 left_hash = hash.Hash(util::Xor(instance_seed, util::SetLsb(node, false)));
  int4 right_hash = hash.Hash(util::Xor(instance_seed, util::SetLsb(node, true)));
  bool control = util::GetLsb(node);
  left = ConvertHashedLeaf(party, left_hash, control, false, last_cw, output_cw);
  right = ConvertHashedLeaf(party, right_hash, control, true, last_cw, output_cw);
}

namespace gpu::detail {

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
struct HalfTreeGpuTraits<HalfTreeDpf<in_bits, Group, Hash, In>> {
  using Backend = Hash;
  using OutputGroup = Group;
  using Input = In;
  static constexpr int kExpansionDepth = in_bits - 1;
  static constexpr int kOutputBits = in_bits;
  static constexpr bool kPacked = false;
};

#if defined(__CUDACC__)

// Point-evaluate in_bits-level HalfTree DPF keys: one thread per key, one output
// value per key at xs[tid]. cws holds dense level-major correction words, the
// layout the split form of Gen() writes: layer i of key k is cws[i * nkeys + k]
// and the terminal layer is cws[in_bits * nkeys + k]. tr[k] carries key k's
// packed control bits; its bit in_bits - 1 is the last level's extra bit.
template <int in_bits, int unroll, bool b, typename Descriptor, typename In, typename Query = In>
__global__ void HalfTreeDpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cws, const In *tr,
    const Query *xs, int nkeys, typename Descriptor::Backend hash, int4 instance_seed) {
  using Scheme = typename Descriptor::Scheme;
  using Group = typename HalfTreeGpuTraits<Scheme>::OutputGroup;
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = static_cast<In>(xs[tid]);
  In tr_k = tr[tid];

#pragma unroll unroll
  for (int i = 0; i < in_bits - 1; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    bool t = fss::util::GetLsb(node);
    int4 h = hash.Hash(fss::util::Xor(instance_seed, node));
    int4 cw = cws[(size_t)i * nkeys + tid];
    node = fss::util::Xor(fss::util::Xor(h, x_bit ? node : Zero4()), t ? cw : Zero4());
  }

  bool x_n = GetBit(x, 0);
  bool t = fss::util::GetLsb(node);
  int4 h = hash.Hash(fss::util::Xor(instance_seed, fss::util::SetLsb(node, x_n)));
  int4 cw = cws[(size_t)(in_bits - 1) * nkeys + tid];
  int4 hcw = fss::util::SetLsb(cw, false);
  bool lcw_xn = x_n ? GetBit(tr_k, in_bits - 1) : fss::util::GetLsb(cw);
  int4 high = fss::util::SetLsb(h, false);
  bool low = fss::util::GetLsb(h);
  if (t) {
    high = fss::util::Xor(high, hcw);
    low = low ^ lcw_xn;
  }
  auto y = Group::From(high);
  if (low) y = y + Group::From(cws[static_cast<size_t>(in_bits) * nkeys + tid]);
  if constexpr (b) y = -y;
  ys[tid] = y.Into();
}

/**
 * GPU point evaluation of nkeys HalfTree DPF keys: ys[k] = Eval(b, seeds[k],
 * cws, tr[k], xs[k]). One thread per key. The inputs are split-form key
 * components: dense level-major correction words and packed control bits, where
 * bit in_bits - 1 of tr[k] is key k's last-level extra bit.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds (device memory), one per key.
 * @param cws Level-major correction words (device memory), (in_bits + 1) * nkeys entries.
 * @param tr Packed control bits (device memory), one per key.
 * @param xs Evaluation points (device memory), one per key.
 * @param ys Output array (device memory), one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A HalfTreeDpf instance (holds the CCR hash and instance seed).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Hash, typename In, typename Query = In>
  requires QueryFor<Query, in_bits>
void HalfTreeDpfEvalPointGpu(bool b, cuda::std::span<const int4> seeds, cuda::std::span<const int4> cws,
    cuda::std::span<const In> tr, cuda::std::span<const Query> xs, int4 *ys, int nkeys,
    const HalfTreeDpf<in_bits, Group, Hash, In> &dpf, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    HalfTreeDpfEvalPointKernel<in_bits, unroll, true, Descriptor, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds.data(), cws.data(), tr.data(), xs.data(), nkeys,
        fss::detail::SchemeAccess::GetHash(dpf), fss::detail::SchemeAccess::GetInstanceSeed(dpf));
  } else {
    HalfTreeDpfEvalPointKernel<in_bits, unroll, false, Descriptor, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds.data(), cws.data(), tr.data(), xs.data(), nkeys,
        fss::detail::SchemeAccess::GetHash(dpf), fss::detail::SchemeAccess::GetInstanceSeed(dpf));
  }
}

#endif  // defined(__CUDACC__)

}  // namespace gpu::detail

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
struct HalfTreeDpf<in_bits, Group, Hash, In>::GpuLaunchShape {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static constexpr int kFrontierDepth = frontier_depth < in_bits - 1 ? frontier_depth : in_bits - 1;
  static_assert(in_bits - kFrontierDepth <= 8,
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

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeDpf<in_bits, Group, Hash, In>::EvalRangeGpu(
    bool b, In first, In last, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const {
#if defined(__CUDACC__)
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  size_t count;
  if (!detail::RangeOutputCount<in_bits>(first, last, ys.size(), count) ||
      cws.size() < in_bits + 1 || cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  constexpr int kThreadBits = [] {
    int bits = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++bits;
    return bits;
  }();
  constexpr int kMaxDepth = [] {
    int depth = 30 + kThreadBits + 1;
    if (depth >= sizeof(size_t) * CHAR_BIT) depth = sizeof(size_t) * CHAR_BIT - 1;
    if (frontier_depth < depth && depth - frontier_depth > 8) depth = frontier_depth + 8;
    return depth;
  }();
  return detail::ForEachAlignedRange<in_bits>(first, last, kMaxDepth,
      [&](In chunk_first, int depth, size_t offset, size_t) {
        int frontier = frontier_depth < depth - 1 ? frontier_depth : depth - 1;
        if (frontier < 0) frontier = 0;
        int thread_bits = kThreadBits < frontier ? kThreadBits : frontier;
        int root = frontier - thread_bits;
        size_t blocks = size_t{1} << root;
        if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
          return cudaErrorInvalidConfiguration;
        gpu::detail::HalfTreeDpfEvalRangeKernel<Descriptor, threads_per_block>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, s0, cws.data(), ys.data() + offset,
                chunk_first, depth, frontier, root, instance_seed_, hash_);
        return cudaGetLastError();
      });
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeDpf<in_bits, Group, Hash, In>::EvalRangeGpu(
    bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws, In tr,
    cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const {
#if defined(__CUDACC__)
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  size_t count;
  if (!detail::RangeOutputCount<in_bits>(first, last, ys.size(), count) || stride == 0 ||
      (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits)) ||
      cws.size() < static_cast<size_t>(in_bits) * stride + 1 ||
      cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  constexpr int kThreadBits = [] {
    int bits = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++bits;
    return bits;
  }();
  constexpr int kMaxDepth = [] {
    int depth = 30 + kThreadBits + 1;
    if (depth >= sizeof(size_t) * CHAR_BIT) depth = sizeof(size_t) * CHAR_BIT - 1;
    if (frontier_depth < depth && depth - frontier_depth > 8) depth = frontier_depth + 8;
    return depth;
  }();
  return detail::ForEachAlignedRange<in_bits>(first, last, kMaxDepth,
      [&](In chunk_first, int depth, size_t offset, size_t) {
        int frontier = frontier_depth < depth - 1 ? frontier_depth : depth - 1;
        if (frontier < 0) frontier = 0;
        int thread_bits = kThreadBits < frontier ? kThreadBits : frontier;
        int root = frontier - thread_bits;
        size_t blocks = size_t{1} << root;
        if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
          return cudaErrorInvalidConfiguration;
        gpu::detail::HalfTreeSplitDpfEvalRangeKernel<Descriptor, threads_per_block>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, s0, cws.data(), tr, stride,
                ys.data() + offset, chunk_first, depth, frontier, root, instance_seed_, hash_);
        return cudaGetLastError();
      });
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeDpf<in_bits, Group, Hash, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kCwPerKey = in_bits + 1;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  if (cws.size() < kCwPerKey || ys.size() < kOutputsPerKey ||
      cws.data() == nullptr || ys.data() == nullptr ||
      kOutputsPerKey > kMaxAddressableSize / sizeof(int4)) return cudaErrorInvalidValue;
  gpu::detail::HalfTreeEvalAllKernel<Descriptor, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeDpf<in_bits, Group, Hash, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const int4> cws, In tr,
    cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 ||
      (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits)))
    return cudaErrorInvalidValue;
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required || ys.size() < kOutputsPerKey ||
      cws.data() == nullptr || ys.data() == nullptr ||
      kOutputsPerKey > std::numeric_limits<size_t>::max() / sizeof(int4))
    return cudaErrorInvalidValue;
  gpu::detail::HalfTreeSplitEvalAllKernel<Descriptor, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth>
      <<<LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), tr, stride, ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeDpf<in_bits, Group, Hash, In>::EvalAllGpuBatch(
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kCwPerKey = in_bits + 1;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  const size_t key_count = s0s.size();
  if (key_count == 0) return cudaSuccess;
  if (key_count > kMaxAddressableSize / kCwPerKey / sizeof(Cw) ||
      key_count > kMaxAddressableSize / kOutputsPerKey / sizeof(int4) ||
      cws.size() < key_count * kCwPerKey || ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::HalfTreeEvalAllBatchKernel<Descriptor, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeDpf<in_bits, Group, Hash, In>::EvalAllGpuBatch(
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const int4> cws,
    cuda::std::span<const In> tr, cuda::std::span<int4> ys,
    cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
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
  if (resolved_stride > (kMaxElements - key_count) / static_cast<size_t>(in_bits))
    return cudaErrorInvalidValue;
  const size_t required = static_cast<size_t>(in_bits) * resolved_stride + key_count;
  if (cws.size() < required || tr.size() < key_count || ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || tr.data() == nullptr ||
      ys.data() == nullptr)
    return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::HalfTreeSplitEvalAllBatchKernel<Descriptor, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), tr.data(), resolved_stride, ys.data(), instance_seed_, hash_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
