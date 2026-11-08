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
 * Formally speaking, there exists a probabilistic polynomial time (PPT) simulator $Sim$ that can generate output computationally indistinguishable from any strict subset of the keys output by $Gen$.
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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cuda/std/array>
#include <fss/group.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>
#include <fss/constraints.cuh>
#include <fss/compat.cuh>
#include <cuda/std/span>
#include <limits>

#include <cstdint>
#include <stdexcept>

namespace fss {

namespace detail {
class SchemeAccess;
}  // namespace detail

/**
 * Comparison predicate.
 *
 * For the input domain $\sG_{in} = \{0, 1\}^n$ as bits and $x, a \in \sG_{in}$, $x < a$ is defined as that the unsigned integer represented by $x$ is less than that represented by $a$, i.e., comparison starts from the most significant bit (MSB).
 *
 * @ingroup fss_api
 */
enum class DcfPred {
  kLt, /**< $y = b$ when $x < a$ */
  kGt, /**< $y = b$ when $x > a$ */
};

namespace detail {

// Split correction-word view shared by the host and device split paths. Layer l is
// read from cws[l*stride] and vws[l*stride], and the right-control correction is
// bit l of packed_tr, so the layer LSBs of vws are ignored. The terminal is
// rebuilt as a plain Cw with no control bit, so bits of packed_tr at or above
// in_bits are ignored. operator[] recovers the Cw the shared algorithms consume,
// so their bodies stay identical on the Cw and split paths.
template <typename Cw, typename In, int in_bits>
struct DcfCwsView {
  const int4 *cws;
  const int4 *vws;
  In packed_tr;
  size_t stride;

  __host__ __device__ inline Cw operator[](int l) const {
    const int4 s_cw = cws[static_cast<size_t>(l) * stride];
    const int4 v_cw = vws[static_cast<size_t>(l) * stride];
    if (l >= in_bits) return Cw{s_cw, v_cw};
    return Cw{s_cw, util::SetLsb(v_cw, ((packed_tr >> l) & 1) != 0)};
  }
};

}  // namespace detail

/**
 * 2-party DCF scheme.
 *
 * Gen() and scalar Eval() support host and device calls when Prg does.
 * EvalRange() and EvalAll() are host operations and support both predicates.
 * EvalAllGpu() and EvalAllGpuBatch() are host CUDA launch methods.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam pred See DcfPred.
 *
 * @ingroup fss_api
 */
template <int in_bits, typename Group, typename Prg, typename In = uint, DcfPred pred = DcfPred::kLt>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
class Dcf {
public:
  /**
   * Correction word.
   *
   * ## Layout
   *
   * According to the paper, there are s, v, tl, tr to be stored.
   * v is converted to the clamped 16B to be stored.
   * tl, tr is stored at the clamped bit of s, v, respectively.
   *
   * @ingroup fss_api
   */
  struct __align__(32) Cw {
    int4 s;  ///< Seed correction with the left control correction in its reserved bit.
    int4 v;  ///< Group-value correction with the right control correction in its reserved bit.
  };
  // For only 1 and aligned memory access on GPU
  static_assert(sizeof(Cw) == 32);

  /// Seed- and value-component count Gen() writes at stride 1: in_bits layers plus the terminal.
  static constexpr int kCwCount = in_bits + 1;

  using Input = In;  ///< The caller-facing input-domain type.

  /**
   * Construct a scheme with configured backends.
   *
   * @param prg Configured Prg backend.
   */
  __host__ __device__ explicit Dcf(Prg prg);

  /**
   * Generate split correction arrays from caller-owned initial seeds.
   *
   * cws receives the seed components and vws the group-value components. Layer l
   * is stored at cws[l*stride] and vws[l*stride]. cws keeps the left-control
   * correction in its LSB exactly as Cw::s does, and vws carries the pure group
   * value with a zero LSB. The right-control bits move out of the v LSBs
   * into tr, bit l for layer l, and unused high bits of tr are zero.
   * The terminal is cws[in_bits*stride] = {0,0,0,0} and
   * vws[in_bits*stride] = v_cw_np1.
   *
   * @param cws Output seed components; its capacity includes in_bits*stride+1 elements.
   * @param vws Output group-value components, with the same layout as cws.
   * @param tr Output packed right-control bits, bit l for layer l.
   * @param s0s Two initial seeds, preserved by this call.
   * @param a Comparison threshold in the input domain.
   * @param b_buf Output value, clamped before conversion to Group.
   * @param stride Positive element stride between layer entries. The default
   * stride of 1 stores the correction words contiguously.
   * @pre a is in the input domain.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and both arrays have at least in_bits*stride+1 elements.
   * @pre Inputs and outputs do not overlap.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A stride or capacity violation aborts with a diagnostic in Debug
   * builds on both the host and the device, and is undefined behavior in
   * Release builds.
   * @post cws, vws, tr and s0s[party] form the corresponding party's key.
   * @post Only the described prefix is written, and storage beyond it is untouched.
   */
  __host__ __device__ void Gen(cuda::std::span<int4> cws, cuda::std::span<int4> vws,
      In &tr, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf, size_t stride = 1);

  /**
   * Evaluate one input from a seed and split correction arrays.
   *
   * Layer l is read from cws[l*stride] and vws[l*stride]. cws carries the
   * left-control correction in its LSB exactly as Cw::s does, and vws holds the
   * group value. The right-control correction is bit l of tr, and the layer LSBs
   * of vws are ignored because the correction is read from tr; bits of tr at or
   * above in_bits are ignored.
   * The terminal is cws[in_bits*stride] = {0,0,0,0} and
   * vws[in_bits*stride] = v_cw_np1, as Gen() writes them.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws Seed components, with layer l at index l*stride.
   * @param vws Group-value components, with layer l at index l*stride.
   * @param tr Packed right-control bits, bit l for layer l.
   * @param x Input in the scheme's domain.
   * @param stride Positive element stride between layer entries.
   * @return This party's output share for the selected predicate.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and both arrays have at least in_bits*stride+1 elements.
   * @pre vws[in_bits*stride].w & 1 == 0.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A stride, capacity, or terminal violation aborts with a diagnostic in
   * Debug builds on both the host and the device, and is undefined behavior in
   * Release builds.
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<const int4> vws, In tr, In x, size_t stride = 1);

  /**
   * Evaluate the closed input interval [first, last] in input order from split arrays.
   *
   * Layer l is read from cws[l*stride] and vws[l*stride]. cws carries the
   * left-control correction in its LSB exactly as Cw::s does, and vws holds the
   * group value. The right-control correction is bit l of tr, and the layer LSBs
   * of vws are ignored because the correction is read from tr; bits of tr at or
   * above in_bits are ignored.
   * The terminal is cws[in_bits*stride] = {0,0,0,0} and
   * vws[in_bits*stride] = v_cw_np1, as Gen() writes them.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval.
   * @param last Last input in the interval, inclusive.
   * @param s0 This party's initial seed.
   * @param cws Seed components, with layer l at index l*stride.
   * @param vws Group-value components, with layer l at index l*stride.
   * @param tr Packed right-control bits, bit l for layer l.
   * @param ys Output storage. Entry x - first receives this party's share at x.
   * Extra capacity beyond last - first + 1 entries is unchanged.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs
   * sequentially on the calling thread. Positive depths require backends that
   * support concurrent calls.
   * @param stride Positive element stride between layer entries.
   * @pre first <= last and both endpoints are in the input domain.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and both arrays have at least in_bits*stride+1 elements.
   * @pre vws[in_bits*stride].w & 1 == 0. A violation aborts with a diagnostic
   * in Debug builds and is undefined behavior in Release builds.
   * @pre The interval's output count and byte size fit size_t, and ys has at
   * least that many entries. The full input domain need not fit size_t.
   * @throws std::invalid_argument if first > last, either endpoint is outside
   * the input domain, the interval does not fit size_t, ys is too short, stride
   * is zero or too large, or the arrays are shorter than in_bits*stride+1.
   */
  void EvalRange(bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<const int4> vws, In tr, cuda::std::span<int4> ys,
      int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate the full domain from a seed and split correction arrays.
   *
   * Layer l is read from cws[l*stride] and vws[l*stride]. cws carries the
   * left-control correction in its LSB exactly as Cw::s does, and vws holds the
   * group value. The right-control correction is bit l of tr, and the layer LSBs
   * of vws are ignored because the correction is read from tr; bits of tr at or
   * above in_bits are ignored.
   * The terminal is cws[in_bits*stride] = {0,0,0,0} and
   * vws[in_bits*stride] = v_cw_np1, as Gen() writes them.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws Seed components, with layer l at index l*stride.
   * @param vws Group-value components, with layer l at index l*stride.
   * @param tr Packed right-control bits, bit l for layer l.
   * @param ys Output array with at least 2^in_bits entries, in input order.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive element stride between layer entries.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and both arrays have at least in_bits*stride+1 elements.
   * @pre vws[in_bits*stride].w & 1 == 0. A violation aborts with a diagnostic
   * in Debug builds and is undefined behavior in Release builds.
   * @pre The output element count and byte size fit size_t, and ys is allocated.
   * @throws std::invalid_argument if stride is zero or too large or the arrays
   * are shorter than in_bits*stride+1.
   */
  void EvalAll(bool b, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<const int4> vws, In tr, int4 ys[], int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate the closed interval [first, last] on the given CUDA stream from
   * split device arrays. Output ys[j] corresponds to first+j.
   *
   * Layer l is read from cws[l*stride] and vws[l*stride]. cws carries the
   * left-control correction in its LSB exactly as Cw::s does, and vws holds the
   * group value. The right-control correction is bit l of tr, and the layer LSBs
   * of vws are ignored because the correction is read from tr; bits of tr at or
   * above in_bits are ignored.
   * The terminal is cws[in_bits*stride] = {0,0,0,0} and
   * vws[in_bits*stride] = v_cw_np1, as Gen() writes them.
   * Resources and asynchronous error handling follow EvalAllGpu().
   * frontier_depth is relative to each aligned evaluation subtree and is capped
   * to its depth.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval.
   * @param last Last input in the interval, inclusive.
   * @param s0 This party's seed, passed by value to the kernel.
   * @param cws Device seed components, with layer l at index l*stride.
   * @param vws Device group-value components, with layer l at index l*stride.
   * @param tr Packed right-control bits, bit l for layer l.
   * @param ys Device output storage. Entry x - first receives this party's share at x.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive element stride between layer entries.
   * @return Immediate CUDA status. Invalid endpoints, buffers, strides, counts
   * or byte sizes return cudaErrorInvalidValue. Grid limits return
   * cudaErrorInvalidConfiguration.
   * @pre stride > 0 and both arrays hold at least in_bits*stride+1 elements.
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalRangeGpu(bool b, In first, In last, int4 s0,
      cuda::std::span<const int4> cws, cuda::std::span<const int4> vws, In tr,
      cuda::std::span<int4> ys, cudaStream_t stream = 0, size_t stride = 1) const;

  /**
   * Launch full-domain CUDA evaluation from split device arrays.
   *
   * Layer l is read from cws[l*stride] and vws[l*stride]. cws carries the
   * left-control correction in its LSB exactly as Cw::s does, and vws holds the
   * group value. The right-control correction is bit l of tr, and the layer LSBs
   * of vws are ignored because the correction is read from tr; bits of tr at or
   * above in_bits are ignored.
   * The terminal is cws[in_bits*stride] = {0,0,0,0} and
   * vws[in_bits*stride] = v_cw_np1, as Gen() writes them.
   * Backend values, resource rules and tuning rules match EvalAllGpu().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed, passed by value to the kernel.
   * @param cws Device seed components, with layer l at index l*stride.
   * @param vws Device group-value components, with layer l at index l*stride.
   * @param tr Packed right-control bits, bit l for layer l.
   * @param ys Device output storage containing at least 2^in_bits int4 blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive element stride between layer entries.
   * @return Immediate CUDA status. Invalid dimensions, buffers, strides, null
   * used buffers, and unaddressable byte sizes return cudaErrorInvalidValue.
   * @pre stride > 0 and both arrays hold at least in_bits*stride+1 elements.
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<const int4> vws, In tr, cuda::std::span<int4> ys,
      cudaStream_t stream = 0, size_t stride = 1) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Launch full-domain CUDA evaluation for s0s.size() keys from split device arrays.
   *
   * The batch uses the level-major layout: key k layer l is read from
   * cws[l*stride + k] and vws[l*stride + k], key k's packed control is tr[k],
   * and key k's terminal is cws[in_bits*stride + k] = {0,0,0,0} and
   * vws[in_bits*stride + k] = v_cw_np1. cws carries the left-control correction
   * in its LSB exactly as Cw::s does, and the layer LSBs of vws are ignored
   * because the right-control correction is read from tr[k]; bits of tr[k] at or
   * above in_bits are ignored.
   * stride selects the layout. 0 selects the dense level-major layout, in
   * which the stride is the key count of this call. Any other value is a
   * positive stride of at least s0s.size(). Resource and tuning rules match
   * EvalAllGpu(). The batch shares one party index and backend configuration.
   * Every key produces the complete domain.
   *
   * @param b Party index shared by the batch.
   * @param s0s Device storage containing this party's seed for each key.
   * @param cws Device level-major seed components.
   * @param vws Device level-major group-value components.
   * @param tr Device packed right-control bits, one value per key.
   * @param ys Device key-major rows of 2^in_bits output blocks per key.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride 0 for the dense level-major layout, where the stride is the
   * key count of this call, or a positive element stride >= s0s.size().
   * @return Immediate CUDA status. An empty batch succeeds without reading
   * buffers. Invalid dimensions, buffers, strides or byte sizes return
   * cudaErrorInvalidValue; an oversized grid returns cudaErrorInvalidConfiguration.
   * Work is asynchronous.
   * @pre stride is 0 or at least s0s.size().
   * @pre Both arrays hold at least in_bits*stride + s0s.size() elements, where a
   * stride of 0 counts as s0s.size().
   * @pre All referenced storage remains alive until the stream work completes.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpuBatch(bool b, cuda::std::span<const int4> s0s,
      cuda::std::span<const int4> cws, cuda::std::span<const int4> vws,
      cuda::std::span<const In> tr, cuda::std::span<int4> ys,
      cudaStream_t stream = 0, size_t stride = 0) const
    requires(in_bits < sizeof(size_t) * 8);

#if FSS_ENABLE_V1_COMPAT
  // Deprecated v1.3.0 aggregate-Cw compatibility entries; see fss/compat.cuh.
  // Each converts through a compile-time-sized scratch buffer and the current
  // split-component method above at stride 1. cws has in_bits + 1 entries.

  /** Generate correction words in the deprecated aggregate Cw layout. */
  [[deprecated("use the split-component Gen(cuda::std::span<int4>, cuda::std::span<int4>, In&, cuda::std::span<const int4, 2>, In, int4, size_t) overload")]]
  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a, int4 b_buf);

  /** Evaluate one point from the deprecated aggregate Cw layout. */
  [[deprecated("use the split-component Eval(bool, int4, cuda::std::span<const int4>, cuda::std::span<const int4>, In, In, size_t) overload")]]
  __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], In x);

  /** Evaluate the full domain from the deprecated aggregate Cw layout. */
  [[deprecated("use the split-component EvalAll(bool, int4, cuda::std::span<const int4>, cuda::std::span<const int4>, In, int4[], int, size_t) overload")]]
  void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[]);
#endif  // FSS_ENABLE_V1_COMPAT

private:
  friend class detail::SchemeAccess;

  Prg prg_;

  template <int frontier_depth, int threads_per_block>
  struct GpuLaunchShape;

  static __host__ __device__ void ExpandNode(Prg &prg, int4 st, Group v,
      bool b, const Cw &cw, int4 &left, Group &v_left, int4 &right, Group &v_right);
  static __host__ __device__ void PathStep(Prg &prg, int4 st, Group v,
      bool b, const Cw &cw, bool take_right, int4 &next_st, Group &next_v);

  // Split correction-word view over the caller's arrays. The host methods and the
  // split GPU kernels share this one view; see detail::DcfCwsView.
  using CwsView = detail::DcfCwsView<Cw, In, in_bits>;

  // Writes both split arrays and accumulates packed right-control bits locally.
  struct CwsWriter {
    int4 *cws;
    int4 *vws;
    size_t stride;
    In &tr;
    In packed_tr = 0;

    __host__ __device__ void Write(int i, int4 s_cw, int4 v_buf, bool tr_cw) {
      cws[static_cast<size_t>(i) * stride] = s_cw;
      vws[static_cast<size_t>(i) * stride] = util::SetLsb(v_buf, false);
      packed_tr |= static_cast<In>(tr_cw) << i;
    }

    __host__ __device__ void Finish(int4 terminal) {
      cws[static_cast<size_t>(in_bits) * stride] = {0, 0, 0, 0};
      vws[static_cast<size_t>(in_bits) * stride] = terminal;
      tr = packed_tr;
    }
  };

  // Shared correction generation loop. The writer is passed by value and owns
  // only output access, so the existing algorithm and backend use stay unchanged.
  template <typename Writer>
  __host__ __device__ void Generate(Writer writer, const int4 *s0s, In a, int4 b_buf);

  // Shared point evaluation over the Cw array or the split view.
  template <typename Cws>
  __host__ __device__ int4 EvalWalk(bool b, int4 s0, Cws cws, In x);

  // Shared interval evaluation. Cws is a Cw pointer for the array form
  // and a CwsView value for the split form.
  template <typename Cws>
  void EvalRangeCws(bool b, In first, In last, int4 s0, Cws cws,
      cuda::std::span<int4> ys, int par_depth);

  // Shared full-domain evaluation over the Cw array or the split view.
  template <typename Cws>
  void EvalAllCws(bool b, int4 s0, Cws cws, int4 ys[], int par_depth);

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
  static constexpr bool kGroupIsUint128 = std::is_same_v<Group, group::Uint128<>>;

  template <typename Cws>
  void EvalRangeSerial(bool b, int4 st, Group value, Cws cws, int4 ys[],
      In first, In last, In node_first, In node_last, int level);

  template <typename Cws>
  void EvalTree(bool b, int4 st, Cws cws, int4 ys[], size_t l, size_t r, int i, int par_depth_, Group v);

  // Expands w scalar frontier nodes (t in LSB, v the running group value) by one level
  // with cws[lv]. The children overwrite the frontier in place: iteration is in reverse
  // so the writes at 2j, 2j+1 never clobber nodes still to be read.
  template <typename Cws>
  void ExpandFrontier(bool b, int4 st_front[], Group v_front[], size_t w, Cws cws, int lv);

  // Applies the level correction to one expanded child branchlessly.
  // m is 0 or -1 from the parent control bit t, and t_cw is the cw control bit
  // masked by t. The cw seed is clamped, so the XOR leaves the child LSB free
  // for the packed control bit.
  static int4 CorrectChild(int4 child, int4 s_cw, int m, int t_cw);

  // Applies the v-channel correction to one child value: the PRG word becomes the child
  // increment (v_cw added when the parent control bit t is set, negated for party 1),
  // which then adds into the parent's running value v. Bit-equal with the operator
  // sequence in EvalTree().
  static Group CorrectVal(int4 base_buf, Group v_cw, bool t, bool b, Group v);

  // Full-domain evaluation of the subtree rooted at (st, v) (t in LSB) at level i through
  // an 8wide batched frontier: a kBatchPrefix-level scalar prefix, batched expansions, and
  // a kBatchTail-level tail carrying (st, v) pairs. Requires
  // in_bits - i >= kBatchPrefix + kBatchTail.
  template <typename Cws>
  void EvalBatched8(bool b, int4 st, Group v, Cws cws, int4 ys[], size_t l, int i);

  // Evaluates a group of 8 consecutive frontier nodes (t in LSB, v the running group
  // value) at level i, covering the leaves [l, l + 8 << (in_bits - i)) in order.
  // Requires in_bits - i >= kBatchTail.
  template <typename Cws>
  void EvalBatched8Group(
      bool b, const int4 (&st_front)[8], const Group (&v_front)[8], Cws cws, int4 ys[], size_t l, int i);
};

// -----------------------------------------------------------------------------

namespace gpu::detail {

template <typename Scheme>
struct DcfGpuSchemeTraits;

#if defined(__CUDACC__)

// One block-local traversal serves the single-key and key-major batch kernels.
// Cws is a Cw pointer on the array path and a kernel-boundary split view
// on the split path, so the traversal body does not depend on the layout.
template <typename Scheme, int frontier_depth, int block_root_depth, bool range = false, typename Cws>
__device__ __forceinline__ void DcfMemberEvalAllBlock(bool b, int4 s0,
    Cws cws, int4 *ys,
    typename DcfGpuSchemeTraits<Scheme>::Backend &prg, int subtree_block_index,
    int64_t thread_index, int4 *shared_states,
    typename DcfGpuSchemeTraits<Scheme>::GroupType *shared_values,
    __uint128_t first = 0, int subtree_depth = DcfGpuSchemeTraits<Scheme>::kInputBits,
    int range_frontier = frontier_depth, int range_block_root = block_root_depth) {
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  constexpr int kInputBits = DcfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kTailCapacity = range ? 8 : kInputBits - frontier_depth;
  const int start_depth = range ? kInputBits - subtree_depth : 0;
  const int frontier = range ? start_depth + range_frontier : frontier_depth;
  const int root_depth = range ? start_depth + range_block_root : block_root_depth;
  const int tail_depth = kInputBits - frontier;
  const int block_subtree_depth = frontier - root_depth;
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
    Group v;
#pragma unroll
    for (int i = 0; i < root_depth; ++i) {
      bool x_bit = i < start_depth
          ? ((first >> (kInputBits - 1 - i)) & 1) != 0
          : ((subtree_block_index >> (root_depth - 1 - i)) & 1) != 0;
      {
        auto path_prg = prg;
        fss::detail::SchemeAccess::DcfPathStep<Scheme>(path_prg, st, v, b, cws[i], x_bit, st, v);
      }
    }
    shared_states[0] = st;
    shared_values[0] = v;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels block_root_depth..frontier_depth.
#pragma unroll
  for (int i = 0; i < block_subtree_depth; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = {0, 0, 0, 0};
    Group nv;
    if (active) {
      n = shared_states[threadIdx.x >> 1];
      nv = shared_values[threadIdx.x >> 1];
    }
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      Group v_left, v_right;
      {
        auto node_prg = prg;
        fss::detail::SchemeAccess::DcfExpandNode<Scheme>(node_prg, n, nv, b, cws[root_depth + i], left, v_left, right, v_right);
      }
      shared_states[threadIdx.x] = (threadIdx.x & 1) ? right : left;
      shared_values[threadIdx.x] = (threadIdx.x & 1) ? v_right : v_left;
    }
    __syncthreads();
  }

  if constexpr (range) {
    int4 pending[8];
    Group pending_values[8];
    int4 node = shared_states[threadIdx.x];
    Group value = shared_values[threadIdx.x];
    int level = frontier;
    const int count = 1 << tail_depth;
    const size_t offset = static_cast<size_t>(thread_index) << tail_depth;
    const auto correction = Group::From(cws[kInputBits].v);
    for (int index = 0; index < count; ++index) {
      while (level < kInputBits) {
        int4 left, right;
        Group left_value, right_value;
        {
          auto node_prg = prg;
          fss::detail::SchemeAccess::DcfExpandNode<Scheme>(node_prg, node,
              value, b, cws[level], left, left_value, right, right_value);
        }
        pending[level - frontier] = right;
        pending_values[level - frontier] = right_value;
        node = left;
        value = left_value;
        ++level;
      }
      auto term = Group::From(util::SetLsb(node, false));
      if (util::GetLsb(node)) term = term + correction;
      if (b) term = -term;
      ys[offset + index] = (value + term).Into();
      if (index + 1 < count) {
        int next = index + 1;
        level = kInputBits - 1;
        while ((next & 1) == 0) {
          --level;
          next >>= 1;
        }
        node = pending[level - frontier];
        value = pending_values[level - frontier];
        ++level;
      }
    }
  } else {
    // Phase 2: expand the subtree below depth frontier_depth, then convert to leaves.
    int4 nodes[1 << kTailCapacity];
    Group vs[1 << kTailCapacity];
    nodes[0] = shared_states[threadIdx.x];
    vs[0] = shared_values[threadIdx.x];
    int count = 1;
#pragma unroll
    for (int lev = frontier; lev < kInputBits; ++lev) {
#pragma unroll
      for (int k = count - 1; k >= 0; --k) {
        int4 left, right;
        Group v_left, v_right;
        {
          auto node_prg = prg;
          fss::detail::SchemeAccess::DcfExpandNode<Scheme>(node_prg, nodes[k], vs[k], b, cws[lev], left, v_left, right, v_right);
        }
        nodes[2 * k] = left;
        vs[2 * k] = v_left;
        nodes[2 * k + 1] = right;
        vs[2 * k + 1] = v_right;
      }
      count *= 2;
    }

    auto v_cw_np1 = Group::From(cws[kInputBits].v);
    const size_t out_base = static_cast<size_t>(thread_index) << tail_depth;
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
}

// Split-layout counterparts of the three kernels above. The correction buffers
// arrive as raw records and the boundary view reconstructs the per-layer Cw
// entries, so the shared traversal stays identical on both paths.
template <typename Scheme, int threads_per_block>
__global__ void DcfSplitEvalRangeKernel(bool b, int4 s0,
    const void *cws_buf, const void *vws_buf,
    typename DcfGpuSchemeTraits<Scheme>::InputType tr,
    size_t stride, int4 *ys, __uint128_t first,
    int depth, int frontier, int root, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using Cws = fss::detail::DcfCwsView<typename Scheme::Cw,
      typename DcfGpuSchemeTraits<Scheme>::InputType, DcfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const auto *vws = static_cast<const int4 *>(vws_buf);
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  __shared__ int4 states[threads_per_block];
  __shared__ typename DcfGpuSchemeTraits<Scheme>::GroupType values[threads_per_block];
  // A function-scope __shared__ array with a user-provided default constructor
  // is not initialized by the compiler, so fill it with the group identity.
  for (int i = static_cast<int>(threadIdx.x); i < threads_per_block; i += static_cast<int>(blockDim.x))
    values[i] = Group{};
  // The initialization must complete before the traversal writes or reads the array.
  __syncthreads();
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  DcfMemberEvalAllBlock<Scheme, 0, 0, true>(b, s0, Cws{cws, vws, tr, stride},
      ys, prg, blockIdx.x, thread_index, states, values, first, depth, frontier, root);
}

template <typename Scheme, int frontier_depth, int block_root_depth>
__global__ void DcfSplitMemberEvalAllKernel(bool b, int4 s0,
    const void *cws_buf, const void *vws_buf,
    typename DcfGpuSchemeTraits<Scheme>::InputType tr,
    size_t stride, int4 *ys, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using Cws = fss::detail::DcfCwsView<typename Scheme::Cw,
      typename DcfGpuSchemeTraits<Scheme>::InputType, DcfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const auto *vws = static_cast<const int4 *>(vws_buf);
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  constexpr int kBlockWidth = 1 << (frontier_depth - block_root_depth);
  __shared__ int4 shared_states[kBlockWidth];
  __shared__ Group shared_values[kBlockWidth];
  // A function-scope __shared__ array with a user-provided default constructor
  // is not initialized by the compiler, so fill it with the group identity.
  for (int i = static_cast<int>(threadIdx.x); i < kBlockWidth; i += static_cast<int>(blockDim.x))
    shared_values[i] = Group{};
  // The initialization must complete before the traversal writes or reads the array.
  __syncthreads();
  const int subtree_block_index = blockIdx.x;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  DcfMemberEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0,
      Cws{cws, vws, tr, stride},
      ys, prg, subtree_block_index, thread_index, shared_states, shared_values);
}

template <typename Scheme, int frontier_depth, int block_root_depth>
__global__ void DcfSplitMemberEvalAllBatchKernel(bool b, const int4 *s0s,
    const void *cws_buf, const void *vws_buf,
    const typename DcfGpuSchemeTraits<Scheme>::InputType *trs,
    size_t stride, int4 *ys, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using Cws = fss::detail::DcfCwsView<typename Scheme::Cw,
      typename DcfGpuSchemeTraits<Scheme>::InputType, DcfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const auto *vws = static_cast<const int4 *>(vws_buf);
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  constexpr int kInputBits = DcfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  constexpr int kBlockWidth = 1 << (frontier_depth - block_root_depth);
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * (size_t{1} << kInputBits);
  __shared__ int4 shared_states[kBlockWidth];
  __shared__ Group shared_values[kBlockWidth];
  // A function-scope __shared__ array with a user-provided default constructor
  // is not initialized by the compiler, so fill it with the group identity.
  for (int i = static_cast<int>(threadIdx.x); i < kBlockWidth; i += static_cast<int>(blockDim.x))
    shared_values[i] = Group{};
  // The initialization must complete before the traversal writes or reads the array.
  __syncthreads();
  DcfMemberEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      Cws{cws + key_index, vws + key_index, trs[key_index], stride},
      key_outputs, prg, subtree_block_index, thread_index, shared_states, shared_values);
}

// Device-key counterparts of the split kernels above: the party seed and the
// packed control are loaded from caller device storage, so no host value of
// the key material exists. The shared traversal is unchanged.
template <typename Scheme, int frontier_depth, int block_root_depth>
__global__ void DcfSplitEvalAllDeviceKeyKernel(bool b, const int4 *s0,
    const void *cws_buf, const void *vws_buf,
    const typename DcfGpuSchemeTraits<Scheme>::InputType *tr,
    size_t stride, int4 *ys, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using Cws = fss::detail::DcfCwsView<typename Scheme::Cw,
      typename DcfGpuSchemeTraits<Scheme>::InputType, DcfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const auto *vws = static_cast<const int4 *>(vws_buf);
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  constexpr int kBlockWidth = 1 << (frontier_depth - block_root_depth);
  __shared__ int4 shared_states[kBlockWidth];
  __shared__ Group shared_values[kBlockWidth];
  for (int i = static_cast<int>(threadIdx.x); i < kBlockWidth; i += static_cast<int>(blockDim.x))
    shared_values[i] = Group{};
  __syncthreads();
  const int subtree_block_index = blockIdx.x;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  DcfMemberEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0[0],
      Cws{cws, vws, tr[0], stride}, ys, prg, subtree_block_index, thread_index,
      shared_states, shared_values);
}

template <typename Scheme, int threads_per_block>
__global__ void DcfSplitEvalRangeDeviceKeyKernel(bool b, const int4 *s0,
    const void *cws_buf, const void *vws_buf,
    const typename DcfGpuSchemeTraits<Scheme>::InputType *tr,
    size_t stride, int4 *ys, __uint128_t first,
    int depth, int frontier, int root, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using Cws = fss::detail::DcfCwsView<typename Scheme::Cw,
      typename DcfGpuSchemeTraits<Scheme>::InputType, DcfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const auto *vws = static_cast<const int4 *>(vws_buf);
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  __shared__ int4 states[threads_per_block];
  __shared__ Group values[threads_per_block];
  for (int i = static_cast<int>(threadIdx.x); i < threads_per_block; i += static_cast<int>(blockDim.x))
    values[i] = Group{};
  __syncthreads();
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  DcfMemberEvalAllBlock<Scheme, 0, 0, true>(b, s0[0], Cws{cws, vws, tr[0], stride},
      ys, prg, blockIdx.x, thread_index, states, values, first, depth, frontier, root);
}

// Key-major batch counterpart: key k's block holds the seed components, then
// the value components, then the packed control record.
template <typename Scheme, int frontier_depth, int block_root_depth>
__global__ void DcfSplitEvalAllBatchDeviceKeyKernel(bool b, const int4 *s0s,
    const void *cws_buf, size_t array_records, size_t key_records, size_t key_count,
    int4 *ys, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using In = typename DcfGpuSchemeTraits<Scheme>::InputType;
  using Cws = fss::detail::DcfCwsView<typename Scheme::Cw, In, DcfGpuSchemeTraits<Scheme>::kInputBits>;
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  constexpr int kInputBits = DcfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  constexpr int kBlockWidth = 1 << (frontier_depth - block_root_depth);
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  const int4 *key = cws + static_cast<size_t>(key_index) * key_records;
  const int4 *key_vws = key + array_records;
  const In *tr = reinterpret_cast<const In *>(key + 2 * array_records);
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * (size_t{1} << kInputBits);
  __shared__ int4 shared_states[kBlockWidth];
  __shared__ Group shared_values[kBlockWidth];
  for (int i = static_cast<int>(threadIdx.x); i < kBlockWidth; i += static_cast<int>(blockDim.x))
    shared_values[i] = Group{};
  __syncthreads();
  DcfMemberEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      Cws{key, key_vws, tr[0], 1}, key_outputs, prg, subtree_block_index, thread_index,
      shared_states, shared_values);
}

// Host launchers for the device-key kernels; the Prg arrives by value from a
// host mirror of the device scheme, matching the public launch methods.
template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred,
    int frontier_depth = 16, int threads_per_block = 256>
__host__ inline cudaError_t DcfEvalAllGpuDeviceKey(bool b, const int4 *s0,
    const int4 *cws, size_t cws_count, const int4 *vws, size_t vws_count, const In *tr,
    size_t stride, int4 *ys, size_t ys_count, const Prg &prg, cudaStream_t stream) {
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr int kFrontierDepth = frontier_depth < in_bits ? frontier_depth : in_bits;
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static_assert(in_bits - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  constexpr int kThreadsPerBlockLog2 = [] {
    int requested_log2 = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++requested_log2;
    return requested_log2 < kFrontierDepth ? requested_log2 : kFrontierDepth;
  }();
  constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(in_bits) ||
      cws_count < required || vws_count < required || ys_count < kOutputsPerKey ||
      s0 == nullptr || cws == nullptr || vws == nullptr || tr == nullptr || ys == nullptr ||
      kOutputsPerKey > kMaxElements) return cudaErrorInvalidValue;
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  DcfSplitEvalAllDeviceKeyKernel<Scheme, kFrontierDepth, kBlockRootDepth>
      <<<size_t{1} << kBlockRootDepth, 1 << kThreadsPerBlockLog2, 0, stream>>>(
          b, s0, cws, vws, tr, stride, ys, prg);
  return cudaGetLastError();
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred,
    int frontier_depth = 16, int threads_per_block = 256>
__host__ inline cudaError_t DcfEvalRangeGpuDeviceKey(bool b, In first, In last,
    const int4 *s0, const int4 *cws, size_t cws_count, const int4 *vws, size_t vws_count,
    const In *tr, size_t stride, int4 *ys, size_t ys_count, const Prg &prg, cudaStream_t stream) {
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(in_bits) ||
      cws_count < required || vws_count < required || s0 == nullptr || cws == nullptr ||
      vws == nullptr || tr == nullptr || ys == nullptr) return cudaErrorInvalidValue;
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  return fss::detail::SplitRangeChunks<in_bits, frontier_depth, threads_per_block>(
      first, last, ys_count,
      [&](In chunk_first, int depth, size_t offset, int frontier, int thread_bits, int root,
          size_t blocks) {
        DcfSplitEvalRangeDeviceKeyKernel<Scheme, threads_per_block>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, s0, cws, vws, tr, stride,
                ys + offset, chunk_first, depth, frontier, root, prg);
        return cudaGetLastError();
      });
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred,
    int frontier_depth = 16, int threads_per_block = 256>
__host__ inline cudaError_t DcfEvalAllGpuBatchDeviceKey(bool b, const int4 *s0s,
    size_t key_count, const int4 *cws, size_t cws_count, size_t array_records,
    size_t key_records, int4 *ys, size_t ys_count, const Prg &prg, cudaStream_t stream) {
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  constexpr int kFrontierDepth = frontier_depth < in_bits ? frontier_depth : in_bits;
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static_assert(in_bits - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  constexpr int kThreadsPerBlockLog2 = [] {
    int requested_log2 = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++requested_log2;
    return requested_log2 < kFrontierDepth ? requested_log2 : kFrontierDepth;
  }();
  constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  constexpr int kBlocksPerKey = 1 << kBlockRootDepth;
  if (key_count == 0) return cudaSuccess;
  if (key_count > kMaxAddressableSize / kOutputsPerKey / sizeof(int4) ||
      key_count > kMaxAddressableSize / key_records / sizeof(int4) ||
      cws_count < key_count * key_records || ys_count < key_count * kOutputsPerKey ||
      s0s == nullptr || cws == nullptr || ys == nullptr ||
      kOutputsPerKey > kMaxElements) return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  DcfSplitEvalAllBatchDeviceKeyKernel<Scheme, kFrontierDepth, kBlockRootDepth>
      <<<key_count * kBlocksPerKey, 1 << kThreadsPerBlockLog2, 0, stream>>>(
          b, s0s, cws, array_records, key_records, key_count, ys, prg);
  return cudaGetLastError();
}

#endif  // defined(__CUDACC__)

}  // namespace gpu::detail

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline __host__ __device__ Dcf<in_bits, Group, Prg, In, pred>::Dcf(Prg prg)
    : prg_(prg) {}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline __host__ __device__ void Dcf<in_bits, Group, Prg, In, pred>::Gen(
    cuda::std::span<int4> cws, cuda::std::span<int4> vws, In &tr,
    cuda::std::span<const int4, 2> s0s, In a, int4 b_buf, size_t stride) {
  util::RequireArg(stride > 0, "Dcf::Gen: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits),
      "Dcf::Gen: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  util::RequireArg(cws.size() >= required && vws.size() >= required,
      "Dcf::Gen: cws and vws must hold in_bits*stride+1 elements");
  Generate(CwsWriter{cws.data(), vws.data(), stride, tr}, s0s.data(), a, b_buf);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Writer>
inline __host__ __device__ void Dcf<in_bits, Group, Prg, In, pred>::Generate(
    Writer writer, const int4 *s0s, In a, int4 b_buf) {
  int4 s0 = s0s[0];
  s0 = util::SetLsb(s0, false);
  int4 s1 = s0s[1];
  s1 = util::SetLsb(s1, false);
  bool t0 = false;
  bool t1 = true;
  Group v;
  b_buf = util::SetLsb(b_buf, false);

  for (int i = 0; i < in_bits; ++i) {
    auto [s0l, v0l_buf, s0r, v0r_buf] = prg_.Gen(s0);
    auto [s1l, v1l_buf, s1r, v1r_buf] = prg_.Gen(s1);

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
    writer.Write(i, s_cw, v_cw.Into(), tr_cw);
  }

  auto v_cw_np1 = Group::From(s1) + (-Group::From(s0)) + (-v);
  if (t1) v_cw_np1 = -v_cw_np1;
  writer.Finish(v_cw_np1.Into());
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline __host__ __device__ int4 Dcf<in_bits, Group, Prg, In, pred>::Eval(
    bool b, int4 s0, cuda::std::span<const int4> cws, cuda::std::span<const int4> vws,
    In tr, In x, size_t stride) {
  util::RequireArg(stride > 0, "Dcf::Eval: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits),
      "Dcf::Eval: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  util::RequireArg(cws.size() >= required && vws.size() >= required,
      "Dcf::Eval: cws and vws must hold in_bits*stride+1 elements");
  return EvalWalk(b, s0, CwsView{cws.data(), vws.data(), tr, stride}, x);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline __host__ __device__ int4 Dcf<in_bits, Group, Prg, In, pred>::EvalWalk(
    bool b, int4 s0, Cws cws, In x) {
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

    bool x_bit = (x >> (in_bits - 1 - i)) & 1;
    auto [child, value_buf] = detail::GenHalf<4>(prg_, s, x_bit);
    bool child_t = util::GetLsb(child);
    child = util::SetLsb(child, false);
    auto value = Group::From(util::SetLsb(value_buf, false));
    if (t) {
      child = util::Xor(child, s_cw);
      child_t = child_t ^ (x_bit ? tr_cw : tl_cw);
    }

    if (b) {
      v = v + (-value);
      if (t) v = v + (-v_cw);
    } else {
      v = v + value;
      if (t) v = v + v_cw;
    }

    s = child;
    t = child_t;
  }

  int4 v_cw_np1_buf = cws[in_bits].v;
  if ((v_cw_np1_buf.w & 1) != 0) {
    util::RequireArg(false, "Dcf::Eval: the terminal group value must have a zero LSB");
  }
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

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalRange(bool b, In first, In last,
    int4 s0, cuda::std::span<const int4> cws, cuda::std::span<const int4> vws, In tr,
    cuda::std::span<int4> ys, int par_depth, size_t stride) {
  if (stride == 0) throw std::invalid_argument("Dcf::EvalRange: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits))
    throw std::invalid_argument("Dcf::EvalRange: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required || vws.size() < required)
    throw std::invalid_argument("Dcf::EvalRange: cws and vws must hold in_bits*stride+1 elements");
  EvalRangeCws(b, first, last, s0,
      CwsView{cws.data(), vws.data(), tr, stride}, ys, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalRangeCws(bool b, In first,
    In last, int4 s0, Cws cws, cuda::std::span<int4> ys, int par_depth) {
  assert(par_depth >= 0);
  if (first > last)
    throw std::invalid_argument("Dcf::EvalRange: first must not exceed last");
  constexpr In domain_last = [] {
    if constexpr (in_bits == sizeof(In) * 8) return static_cast<In>(~In{0});
    else return static_cast<In>((In{1} << in_bits) - 1);
  }();
  if (last > domain_last)
    throw std::invalid_argument("Dcf::EvalRange: last must stay inside the input domain");
  const __uint128_t distance = static_cast<__uint128_t>(last) - first;
  if (distance >= std::numeric_limits<size_t>::max() / sizeof(int4))
    throw std::invalid_argument("Dcf::EvalRange: the interval is too large for size_t");
  const size_t count = static_cast<size_t>(distance) + 1;
  if (ys.size() < count)
    throw std::invalid_argument("Dcf::EvalRange: ys must hold last-first+1 entries");
  // The terminal LSB is a caller precondition. Check it on the calling thread,
  // so a Debug violation aborts there instead of inside an OpenMP worker;
  // Release builds do not check it.
  util::RequireArg(!util::GetLsb(cws[in_bits].v),
      "Dcf::EvalRange: the terminal group value must have a zero LSB");
  const size_t partitions = detail::ResolveParPartitions(par_depth, count);
  detail::ParallelFor(par_depth, partitions, [&](size_t partition) {
    const size_t remainder = count % partitions;
    const size_t begin = partition * (count / partitions) +
        (partition < remainder ? partition : remainder);
    const size_t length = count / partitions + (partition < remainder);
    const In partition_first = first + static_cast<In>(begin);
    const In partition_last = partition_first + static_cast<In>(length - 1);
    EvalRangeSerial(b, util::SetLsb(s0, b), Group{}, cws, ys.data() + begin,
        partition_first, partition_last, In{0}, domain_last, 0);
  });
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalRangeSerial(
    bool b, int4 st, Group value, Cws cws, int4 ys[],
    In first, In last, In node_first, In node_last, int level) {
  if (level == in_bits) {
    const int4 final_correction = cws[in_bits].v;
    // The shared entry validated the terminal before parallel dispatch.
    assert(!util::GetLsb(final_correction));
    auto term = Group::From(util::SetLsb(st, false));
    if (util::GetLsb(st)) term = term + Group::From(final_correction);
    if (b) term = -term;
    ys[static_cast<size_t>(node_first - first)] = (value + term).Into();
    return;
  }
  int4 left, right;
  Group left_value, right_value;
  ExpandNode(prg_, st, value, b, cws[level], left, left_value, right, right_value);
  const In midpoint = node_first + ((node_last - node_first) >> 1);
  if (first <= midpoint) {
    EvalRangeSerial(b, left, left_value, cws, ys, first, last,
        node_first, midpoint, level + 1);
  }
  if (last > midpoint) {
    EvalRangeSerial(b, right, right_value, cws, ys, first, last,
        midpoint + In{1}, node_last, level + 1);
  }
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalAll(bool b, int4 s0,
    cuda::std::span<const int4> cws, cuda::std::span<const int4> vws, In tr,
    int4 ys[], int par_depth, size_t stride) {
  if (stride == 0) throw std::invalid_argument("Dcf::EvalAll: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits))
    throw std::invalid_argument("Dcf::EvalAll: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required || vws.size() < required)
    throw std::invalid_argument("Dcf::EvalAll: cws and vws must hold in_bits*stride+1 elements");
  EvalAllCws(b, s0, CwsView{cws.data(), vws.data(), tr, stride}, ys, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalAllCws(
    bool b, int4 s0, Cws cws, int4 ys[], int par_depth) {
  assert(par_depth >= 0);
  static_assert(detail::FitsFullDomain<int4, in_bits>(),
      "full-domain output count and byte size must fit in size_t");

  // The terminal LSB is a caller precondition. Check it on the calling thread,
  // so a Debug violation aborts there instead of inside an OpenMP worker;
  // Release builds do not check it.
  util::RequireArg(!util::GetLsb(cws[in_bits].v),
      "Dcf::EvalAll: the terminal group value must have a zero LSB");

  int par_depth_ = fss::detail::ResolveParDepth(par_depth, in_bits);

  int4 st = s0;
  st = util::SetLsb(st, b);
  Group v;

  if constexpr (in_bits < kBatchPrefix + kBatchTail) {
    // Domains too small to fill an 8wide frontier keep the recursive tree walk.
    if (par_depth_ == 0) {
      EvalTree(b, st, cws, ys, 0, size_t{1} << in_bits, 0, 0, v);
    } else {
      int threads = fss::detail::ResolveParThreads(par_depth_, size_t{1} << in_bits);
#pragma omp parallel num_threads(threads)
#pragma omp single
      EvalTree(b, st, cws, ys, 0, size_t{1} << in_bits, 0, par_depth_, v);
    }
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
    for (int lv = 0; lv < split_depth; ++lv) ExpandFrontier(b, st_front.data(), v_front.data(), size_t{1} << lv, cws, lv);
    size_t span = (size_t{1} << in_bits) / num_tasks;
    fss::detail::ParallelFor(par_depth_, num_tasks, [&](size_t task) {
      EvalBatched8(b, st_front[task], v_front[task], cws, ys, task * span, split_depth);
    });
  } else {
    EvalBatched8(b, st, v, cws, ys, 0, 0);
  }
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalTree(
    bool b, int4 st, Cws cws, int4 ys[], size_t l, size_t r, int i, int par_depth_, Group v) {
  bool t = util::GetLsb(st);
  int4 s = st;
  s = util::SetLsb(s, false);

  if (i == in_bits) {
    int4 v_cw_np1_buf = cws[in_bits].v;
    // The shared entry validated the terminal before parallel dispatch.
    assert(!util::GetLsb(v_cw_np1_buf));
    auto term = Group::From(s);
    if (t) term = term + Group::From(v_cw_np1_buf);
    if (b) term = -term;
    v = v + term;
    assert(l + 1 == r);
    ys[l] = v.Into();
    return;
  }

  int4 stl, str;
  Group vl, vr;
  ExpandNode(prg_, st, v, b, cws[i], stl, vl, str, vr);

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

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::ExpandFrontier(
    bool b, int4 st_front[], Group v_front[], size_t w, Cws cws, int lv) {
  int4 s_cw = util::SetLsb(cws[lv].s, false);
  bool tl_cw = util::GetLsb(cws[lv].s);
  bool tr_cw = util::GetLsb(cws[lv].v);
  auto v_cw = Group::From(util::SetLsb(cws[lv].v, false));

  for (size_t j = w; j-- > 0;) {
    bool t = util::GetLsb(st_front[j]);
    int4 s = util::SetLsb(st_front[j], false);
    // The parent value must be read before the children overwrite the frontier:
    // for j = 0 the left child lands exactly on the parent's slot.
    Group v_parent = v_front[j];

    auto [sl, vl_buf, sr, vr_buf] = prg_.Gen(s);

    int m = -static_cast<int>(t);
    st_front[2 * j] = CorrectChild(sl, s_cw, m, m & static_cast<int>(tl_cw));
    st_front[2 * j + 1] = CorrectChild(sr, s_cw, m, m & static_cast<int>(tr_cw));
    v_front[2 * j] = CorrectVal(vl_buf, v_cw, t, b, v_parent);
    v_front[2 * j + 1] = CorrectVal(vr_buf, v_cw, t, b, v_parent);
  }
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline int4 Dcf<in_bits, Group, Prg, In, pred>::CorrectChild(int4 child, int4 s_cw, int m, int t_cw) {
  int4 res = {child.x ^ (s_cw.x & m), child.y ^ (s_cw.y & m), child.z ^ (s_cw.z & m), child.w ^ (s_cw.w & m)};
  res.w = (res.w & ~1) | ((child.w ^ t_cw) & 1);
  return res;
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline Group Dcf<in_bits, Group, Prg, In, pred>::CorrectVal(
    int4 base_buf, Group v_cw, bool t, bool b, Group v) {
  if constexpr (kGroupIsBytes) {
    // The XOR group: negation is the identity, so the correction is three XORs with
    // the cw masked by the parent control bit.
    int m = -static_cast<int>(t);
    int4 base = util::SetLsb(base_buf, false);
    int4 value = v.Into();
    int4 correction = v_cw.Into();
    int4 res;
    res.x = value.x ^ base.x ^ (correction.x & m);
    res.y = value.y ^ base.y ^ (correction.y & m);
    res.z = value.z ^ base.z ^ (correction.z & m);
    res.w = value.w ^ base.w ^ (correction.w & m);
    return Group::From(res);
  } else if constexpr (kGroupIsUint128) {
    // The clamped 127-bit integer group: mod-2^127 addition and negation in branchless
    // masked forms. Operands are < 2^127, so a uint128 sum loses at most the carry-out
    // at bit 127 and masking it off is the exact reduction; the party-1 subtraction
    // wraps the borrow the same way. Bit-equal with the generic operator sequence.
    constexpr __uint128_t kMask127 = (static_cast<__uint128_t>(1) << 127) - 1;
    __uint128_t m = t ? ~static_cast<__uint128_t>(0) : 0;
    __uint128_t x = (Group::From(util::SetLsb(base_buf, false)).Value() + (v_cw.Value() & m)) & kMask127;
    __uint128_t res;
    if (b) res = (v.Value() - x) & kMask127;
    else res = (v.Value() + x) & kMask127;
    int4 buf = detail::Pack(res);
    buf.w = static_cast<int>(static_cast<unsigned int>(buf.w) << 1);
    return Group::From(buf);
  } else {
    auto base = Group::From(util::SetLsb(base_buf, false));
    if (t) base = base + v_cw;
    if (b) base = -base;
    return v + base;
  }
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalBatched8(
    bool b, int4 st, Group v, Cws cws, int4 ys[], size_t l, int i) {
  int4 st_front[8];
  Group v_front[8];
  st_front[0] = st;
  v_front[0] = v;
  for (int lv = i; lv < i + kBatchPrefix; ++lv) ExpandFrontier(b, st_front, v_front, size_t{1} << (lv - i), cws, lv);
  EvalBatched8Group(b, st_front, v_front, cws, ys, l, i + kBatchPrefix);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <typename Cws>
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalBatched8Group(
    bool b, const int4 (&st_front)[8], const Group (&v_front)[8], Cws cws, int4 ys[], size_t l, int i) {
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
        auto out = detail::GenBatch<4>(prg_, seeds);
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
    // The shared entry validated the terminal before parallel dispatch.
    assert(!util::GetLsb(v_cw_np1_buf));
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
  auto out = detail::GenBatch<4>(prg_, seeds);
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

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
__host__ __device__ inline void Dcf<in_bits, Group, Prg, In, pred>::ExpandNode(
    Prg &prg, int4 st, Group v, bool b, const Cw &cw,
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

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
__host__ __device__ inline void Dcf<in_bits, Group, Prg, In, pred>::PathStep(
    Prg &prg, int4 st, Group v, bool b, const Cw &cw, bool take_right,
    int4 &next_st, Group &next_v) {
  cuda::std::array<int4, 2> raw;
  {
    auto node_prg = prg;
    raw = detail::GenHalf<4>(node_prg, util::SetLsb(st, false), take_right);
  }
  bool t = util::GetLsb(st);
  int4 child = util::SetLsb(raw[0], false);
  bool child_t = util::GetLsb(raw[0]);
  auto value = Group::From(util::SetLsb(raw[1], false));
  if (t) {
    child = util::Xor(child, util::SetLsb(cw.s, false));
    child_t = child_t ^ util::GetLsb(take_right ? cw.v : cw.s);
    value = value + Group::From(util::SetLsb(cw.v, false));
  }
  if (b) value = -value;
  next_st = util::SetLsb(child, child_t);
  next_v = value + v;
}

namespace gpu::detail {

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> && Groupable<Group> && Prgable<Prg, 4>)
struct DcfGpuSchemeTraits<Dcf<in_bits, Group, Prg, In, pred>> {
  using Backend = Prg;
  using GroupType = Group;
  using InputType = In;
  static constexpr int kInputBits = in_bits;
};

#if defined(__CUDACC__)

// Point-evaluate in_bits-level DCF keys, one thread per key. ys[k] is the
// party's share of the selected predicate at xs[k]. cws and vws hold dense
// level-major rows, the layout the split form of Gen() writes: layer i of key k
// is cws[i * nkeys + k] and vws[i * nkeys + k]. tr[k] carries key k's packed
// right-control bits, bit i for layer i, and the terminal correction word is
// vws's tail row. Group default construction provides the additive identity.
template <int in_bits, int unroll, bool b, typename Scheme, typename In, typename Query = In>
__global__ void DcfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cws, const int4 *vws, const In *tr,
    const Query *xs, int nkeys, typename DcfGpuSchemeTraits<Scheme>::Backend prg) {
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  Group v = Group::From(Zero4());
  bool t = b;
  In x = static_cast<In>(xs[tid]);
  In tr_k = tr[tid];

#pragma unroll unroll
  for (int i = 0; i < in_bits; ++i) {
    int4 s_cw = cws[(size_t)i * nkeys + tid];
    bool tl_cw = fss::util::GetLsb(s_cw);
    s_cw = fss::util::SetLsb(s_cw, false);
    int4 v_cw_buf = vws[(size_t)i * nkeys + tid];
    bool tr_cw = GetBit(tr_k, i);
    v_cw_buf = fss::util::SetLsb(v_cw_buf, false);
    auto v_cw = Group::From(v_cw_buf);

    auto g = prg.Gen(s);
    int4 sl = g[0], vl_buf = g[1], sr = g[2], vr_buf = g[3];

    bool tl = fss::util::GetLsb(sl);
    sl = fss::util::SetLsb(sl, false);
    vl_buf = fss::util::SetLsb(vl_buf, false);
    auto vl = Group::From(vl_buf);
    bool tr = fss::util::GetLsb(sr);
    sr = fss::util::SetLsb(sr, false);
    vr_buf = fss::util::SetLsb(vr_buf, false);
    auto vr = Group::From(vr_buf);

    if (t) {
      sl = fss::util::Xor(sl, s_cw);
      sr = fss::util::Xor(sr, s_cw);
      tl = tl ^ tl_cw;
      tr = tr ^ tr_cw;
    }

    bool x_bit = GetBit(x, in_bits - 1 - i);
    if constexpr (b) {
      v = v + (x_bit ? (-vr) : (-vl));
      if (t) v = v + (-v_cw);
    } else {
      v = v + (x_bit ? vr : vl);
      if (t) v = v + v_cw;
    }
    if (x_bit) {
      s = sr;
      t = tr;
    } else {
      s = sl;
      t = tl;
    }
  }

  auto v_out_cw = Group::From(vws[static_cast<size_t>(in_bits) * nkeys + tid]);
  if constexpr (b) {
    v = v + (-Group::From(s));
    if (t) v = v + (-v_out_cw);
  } else {
    v = v + Group::From(s);
    if (t) v = v + v_out_cw;
  }
  ys[tid] = v.Into();
}

/**
 * GPU point evaluation of nkeys DCF keys: ys[k] = Eval(b, seeds[k], cws, vws,
 * tr[k], xs[k]). One thread per key. The inputs are split-form key components:
 * dense level-major correction words, packed right-control bits, and the
 * terminal correction word in the group-value array's tail row.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds (device memory), one per key.
 * @param cws Level-major seed components (device memory), (in_bits + 1) * nkeys entries.
 * @param vws Level-major group-value components (device memory), (in_bits + 1) * nkeys entries.
 * @param tr Packed right-control bits (device memory), one per key.
 * @param xs Evaluation points (device memory), one per key.
 * @param ys Output array (device memory), one int4 per key.
 * @param nkeys Number of keys.
 * @param dcf A Dcf instance (holds the PRG).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, DcfPred pred, typename Query = In>
  requires QueryFor<Query, in_bits>
void DcfEvalPointGpu(bool b, cuda::std::span<const int4> seeds, cuda::std::span<const int4> cws,
    cuda::std::span<const int4> vws, cuda::std::span<const In> tr, cuda::std::span<const Query> xs, int4 *ys,
    int nkeys, const Dcf<in_bits, Group, Prg, In, pred> &dcf, cudaStream_t stream = 0) {
  if (nkeys < 0)
    throw std::invalid_argument("DcfEvalPointGpu: nkeys must be nonnegative");
  if (nkeys == 0) return;
  const size_t count = static_cast<size_t>(nkeys);
  constexpr size_t kLevels = static_cast<size_t>(in_bits) + 1;
  if (count > SIZE_MAX / kLevels || cws.size() < count * kLevels || vws.size() < count * kLevels)
    throw std::invalid_argument("DcfEvalPointGpu: cws and vws must hold (in_bits+1)*nkeys elements");
  if (seeds.size() < count || tr.size() < count || xs.size() < count || ys == nullptr)
    throw std::invalid_argument("DcfEvalPointGpu: seeds, tr, xs, and ys must cover nkeys entries");
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    DcfEvalPointKernel<in_bits, unroll, true, Scheme, In, Query><<<blocks, kBs, 0, stream>>>(
        ys, seeds.data(), cws.data(), vws.data(), tr.data(), xs.data(), nkeys,
        fss::detail::SchemeAccess::GetPrg(dcf));
  } else {
    DcfEvalPointKernel<in_bits, unroll, false, Scheme, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds.data(), cws.data(), vws.data(), tr.data(), xs.data(), nkeys,
        fss::detail::SchemeAccess::GetPrg(dcf));
  }
}

#endif  // defined(__CUDACC__)

}  // namespace gpu::detail

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <int frontier_depth, int threads_per_block>
struct Dcf<in_bits, Group, Prg, In, pred>::GpuLaunchShape {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 && (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static constexpr int kFrontierDepth = frontier_depth < in_bits ? frontier_depth : in_bits;
  static_assert(in_bits - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  static constexpr int kThreadsPerBlockLog2 = [] {
    int threads_per_block_log2 = 0;
    for (int thread_count = threads_per_block; thread_count > 1; thread_count >>= 1) ++threads_per_block_log2;
    return threads_per_block_log2 < kFrontierDepth ? threads_per_block_log2 : kFrontierDepth;
  }();
  static constexpr int kThreadsPerBlock = 1 << kThreadsPerBlockLog2;
  static constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  static_assert(kBlockRootDepth <= 30, "gpu block index exceeds the signed-int range");
  static constexpr size_t kBlocksPerKey = size_t{1} << (kBlockRootDepth <= 30 ? kBlockRootDepth : 0);
};

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dcf<in_bits, Group, Prg, In, pred>::EvalRangeGpu(
    bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws,
    cuda::std::span<const int4> vws, In tr, cuda::std::span<int4> ys,
    cudaStream_t stream, size_t stride) const {
#if defined(__CUDACC__)
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 || (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits)) ||
      cws.size() < static_cast<size_t>(in_bits) * stride + 1 ||
      vws.size() < static_cast<size_t>(in_bits) * stride + 1 ||
      cws.data() == nullptr || vws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  return detail::SplitRangeChunks<in_bits, frontier_depth, threads_per_block>(
      first, last, ys.size(),
      [&](In chunk_first, int depth, size_t offset, int frontier, int thread_bits, int root,
          size_t blocks) {
        gpu::detail::DcfSplitEvalRangeKernel<Scheme, threads_per_block>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, s0, cws.data(), vws.data(), tr,
                stride, ys.data() + offset, chunk_first, depth, frontier, root, prg_);
        return cudaGetLastError();
      });
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dcf<in_bits, Group, Prg, In, pred>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const int4> cws, cuda::std::span<const int4> vws,
    In tr, cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 ||
      (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits)))
    return cudaErrorInvalidValue;
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required || vws.size() < required || ys.size() < kOutputsPerKey ||
      cws.data() == nullptr || vws.data() == nullptr || ys.data() == nullptr ||
      kOutputsPerKey > std::numeric_limits<size_t>::max() / sizeof(int4)) return cudaErrorInvalidValue;
  gpu::detail::DcfSplitMemberEvalAllKernel<Scheme, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), vws.data(), tr, stride, ys.data(), prg_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dcf<in_bits, Group, Prg, In, pred>::EvalAllGpuBatch(
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const int4> cws,
    cuda::std::span<const int4> vws, cuda::std::span<const In> tr,
    cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  size_t key_count = s0s.size();
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
  if (cws.size() < required || vws.size() < required || tr.size() < key_count ||
      ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || vws.data() == nullptr ||
      tr.data() == nullptr || ys.data() == nullptr) return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::DcfSplitMemberEvalAllBatchKernel<Scheme, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), vws.data(), tr.data(), resolved_stride, ys.data(), prg_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

#if FSS_ENABLE_V1_COMPAT

// Deprecated v1.3.0 aggregate-Cw entries. Every wrapper copies between the
// caller's Cw layout and compile-time-sized split-component scratch buffers,
// then runs the current public method at stride 1. The old layout kept the
// right-control bit in the LSB of each layer's v; the split form carries it in
// bit i of tr, and the terminal seed is zero as the current form writes it.
template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
__host__ __device__ inline void Dcf<in_bits, Group, Prg, In, pred>::Gen(
    Cw cws[], const int4 s0s[2], In a, int4 b_buf) {
  int4 seed[in_bits + 1];
  int4 value[in_bits + 1];
  In tr;
  Gen(cuda::std::span<int4>(seed, in_bits + 1), cuda::std::span<int4>(value, in_bits + 1),
      tr, cuda::std::span<const int4, 2>(s0s, 2), a, b_buf);
  for (int i = 0; i < in_bits; ++i) {
    cws[i] = {seed[i], util::SetLsb(value[i], ((tr >> i) & 1) != 0)};
  }
  cws[in_bits] = {int4{0, 0, 0, 0}, value[in_bits]};
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
__host__ __device__ inline int4 Dcf<in_bits, Group, Prg, In, pred>::Eval(
    bool b, int4 s0, const Cw cws[], In x) {
  int4 seed[in_bits + 1];
  int4 value[in_bits + 1];
  In tr = 0;
  for (int i = 0; i < in_bits; ++i) {
    seed[i] = cws[i].s;
    value[i] = util::SetLsb(cws[i].v, false);
    tr |= static_cast<In>(util::GetLsb(cws[i].v)) << i;
  }
  seed[in_bits] = int4{0, 0, 0, 0};
  value[in_bits] = cws[in_bits].v;
  return Eval(b, s0, cuda::std::span<const int4>(seed, in_bits + 1),
      cuda::std::span<const int4>(value, in_bits + 1), tr, x);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 4>)
inline void Dcf<in_bits, Group, Prg, In, pred>::EvalAll(
    bool b, int4 s0, const Cw cws[], int4 ys[]) {
  int4 seed[in_bits + 1];
  int4 value[in_bits + 1];
  In tr = 0;
  for (int i = 0; i < in_bits; ++i) {
    seed[i] = cws[i].s;
    value[i] = util::SetLsb(cws[i].v, false);
    tr |= static_cast<In>(util::GetLsb(cws[i].v)) << i;
  }
  seed[in_bits] = int4{0, 0, 0, 0};
  value[in_bits] = cws[in_bits].v;
  EvalAll(b, s0, cuda::std::span<const int4>(seed, in_bits + 1),
      cuda::std::span<const int4>(value, in_bits + 1), tr, ys);
}

#endif  // FSS_ENABLE_V1_COMPAT

}  // namespace fss
