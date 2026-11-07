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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cuda/std/array>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>
#include <cuda/std/span>
#include <limits>
#include <fss/constraints.cuh>

#include <cstdint>
#include <stdexcept>

namespace fss {

namespace detail {
class SchemeAccess;
}  // namespace detail

namespace detail {

// Split correction-word view shared by the host and device split paths. Layer i is
// read from cws[i*stride] and the right-control correction is bit i of
// packed_tr. The terminal is rebuilt as a plain Cw with no control bit, so bits of
// packed_tr at or above in_bits are never read; the shift by in_bits would be
// undefined when in_bits equals the width of In. The layer LSB keeps the
// left-control correction exactly as the Cw layout does. operator[] recovers the Cw
// the shared algorithms consume, so their bodies stay identical on the Cw and split
// paths.
template <typename Cw, typename In, int in_bits>
struct DpfCwsView {
  const int4 *s;
  In packed_tr;
  size_t stride;

  __host__ __device__ inline Cw operator[](int i) const {
    const int4 s_cw = s[static_cast<size_t>(i) * stride];
    if (i >= in_bits) return Cw{s_cw, false};
    return Cw{s_cw, ((packed_tr >> i) & 1) != 0};
  }
};

}  // namespace detail

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
class GrottoDcf;

/**
 * 2-party DPF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @ingroup fss_api
 *
 * Gen() and scalar Eval() are callable from host and device when the backend supports
 * the execution space and all accessed storage is accessible there.
 * EvalRange() and EvalAll() are host operations. EvalAllGpu() and EvalAllGpuBatch() are host
 * launch methods compiled with a CUDA compiler; they do not run on the device.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
class Dpf {
public:
  /**
     * Correction word.
     *
     * ## Layout
     *
     * According to the paper, there are s, tl, tr to be stored.
     * tl is stored at the clamped bit of s.
     * @ingroup fss_api
     */
  struct __align__(32) Cw {
    int4 s;  ///< Seed correction, with the left control correction in its least significant bit.
    bool tr;  ///< Right control correction.
  };
  // For only 1 and aligned memory access on GPU
  static_assert(sizeof(Cw) == 32);

  /**
     * One party's key in the aggregate Cw layout: the correction words plus
     * the party's initial seed. Dmpf stores one of these per bucket as
     * Dmpf::BucketKey; single-point callers pass component arrays to Gen()
     * and Eval() instead of this type.
     * @ingroup fss_api
     */
  struct Key {
    Cw cws[in_bits + 1];  ///< Tree correction words followed by the output correction word.
    int4 s0;  ///< This party's initial seed.
  };

  /// Seed-component count Gen() writes at stride 1: in_bits layers plus the terminal.
  static constexpr int kCwCount = in_bits + 1;

  using Input = In;  ///< The caller-facing input-domain type.

  /**
   * Configure the pseudorandom generator.
   *
   * @param prg Backend used throughout generation and evaluation.
   */
  __host__ __device__ explicit Dpf(Prg prg);

  /**
   * Generate correction words in a caller-strided layout and packed control bits.
   *
   * Layer i is stored at cws[i*stride] and the terminal at cws[in_bits*stride].
   * tr receives root-order packed right-control bits: bit 0 is the root layer,
   * bit i belongs to layer i, and bits at or above in_bits are zero.
   * The terminal is stored unchanged as a plain int4.
   * The default stride of 1 stores the correction words contiguously.
   *
   * @param cws Output storage; its capacity includes in_bits*stride+1 elements.
   * @param tr Output packed right-control bits, with unused bits zero.
   * @param s0s Two initial seeds, retained by the caller.
   * @param a The secret point.
   * @param b_buf Output value, clamped before conversion to Group.
   * @param stride Positive element stride between correction-word entries.
   * @pre a is in [0, 2^in_bits).
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre Inputs and outputs do not overlap.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A stride or capacity violation aborts with a diagnostic in Debug
   * builds on both the host and the device, and is undefined behavior in
   * Release builds.
   * @post Only the described prefix is written, and storage beyond it is untouched.
   */
  __host__ __device__ void Gen(cuda::std::span<int4> cws, In &tr,
      cuda::std::span<const int4, 2> s0s, In a, int4 b_buf, size_t stride = 1);

  /**
   * Evaluate one point using separate correction components.
   *
   * Root-order packed bit i is the right-control correction at layer i.
   * The correction s LSB is the left-control correction.
   * Layer i is read from cws[i*stride] and the terminal from cws[in_bits*stride],
   * which Gen() writes as a plain int4 with its LSB zero.
   * Packed bits at and above in_bits are ignored.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer i at index i*stride and the
   * terminal at index in_bits*stride.
   * @param tr Packed right-control bits, bit i for layer i.
   * @param x Query point.
   * @param stride Positive stride between correction entries.
   * @return This party's output share.
   * @pre x is in [0, 2^in_bits).
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre cws[in_bits*stride].w & 1 == 0.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A stride, capacity, or terminal violation aborts with a diagnostic in
   * Debug builds on both the host and the device, and is undefined behavior in
   * Release builds.
   */
  __host__ __device__ int4 Eval(bool b, int4 s0, cuda::std::span<const int4> cws,
      In tr, In x, size_t stride = 1);

  /**
   * Evaluate the closed input interval [first, last] on the host from split correction words.
   *
   * Layer i is read from cws[i*stride] and the terminal from cws[in_bits*stride].
   * Packed bit i of tr is the right-control correction at layer i, and the
   * left-control correction is the LSB of the layer's cws entry, exactly as the
   * component form of Gen() writes them. Bits of tr at and above in_bits are ignored.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval.
   * @param last Last input in the interval, included in the output.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer i at index i*stride and the
   * terminal at index in_bits*stride.
   * @param tr Packed right-control bits, bit i for layer i.
   * @param ys Output storage with at least last-first+1 elements.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive element stride between correction-word entries.
   * @pre first <= last and both endpoints are in [0, 2^in_bits).
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre cws[in_bits*stride].w & 1 == 0. A violation aborts with a diagnostic
   * in Debug builds and is undefined behavior in Release builds.
   * @pre cws and ys.data() are nonnull and refer to accessible host storage.
   * @throws std::invalid_argument if first > last, an endpoint is outside the
   * input domain, the interval does not fit size_t, ys is too short, stride is
   * zero or too large, or cws is shorter than in_bits*stride+1.
   * @post ys[x-first] contains this party's output share for each x in the interval.
   * Any extra elements of ys remain unchanged.
   */
  void EvalRange(bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws,
      In tr, cuda::std::span<int4> ys, int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate the full domain on the host from split correction words.
   *
   * Layer i is read from cws[i*stride] and the terminal from cws[in_bits*stride].
   * Packed bit i of tr is the right-control correction at layer i, and the
   * left-control correction is the LSB of the layer's cws entry, exactly as the
   * component form of Gen() writes them. Bits of tr at and above in_bits are ignored.
   * par_depth controls OpenMP partitioning for this call.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer i at index i*stride and the
   * terminal at index in_bits*stride.
   * @param tr Packed right-control bits, bit i for layer i.
   * @param ys Output array of at least 2^in_bits elements.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive element stride between correction-word entries.
   * @pre The output element count and byte size fit size_t.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4),
   * stride <= (M-1)/in_bits and cws has at least in_bits*stride+1 elements.
   * @pre cws[in_bits*stride].w & 1 == 0. A violation aborts with a diagnostic
   * in Debug builds and is undefined behavior in Release builds.
   * @throws std::invalid_argument if stride is zero or too large or cws is
   * shorter than in_bits*stride+1.
   * @post ys[x] contains this party's output share for x.
   */
  void EvalAll(bool b, int4 s0, cuda::std::span<const int4> cws, In tr,
      int4 ys[], int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate the closed interval [first, last] on the given CUDA stream from split arrays.
   * Output ys[j] corresponds to first+j. Extra output capacity is unchanged.
   *
   * Layer i is read from cws[i*stride] and the terminal from cws[in_bits*stride].
   * Packed bit i of tr is the right-control correction at layer i, and the
   * left-control correction is the LSB of the layer's cws entry, exactly as the
   * component form of Gen() writes them. Bits of tr at and above in_bits are ignored.
   * Resources, subdivision and asynchronous error handling follow EvalRangeGpu().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input in the interval.
   * @param last Last input in the interval, included in the output.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device correction words, layer i at index i*stride and
   * the terminal at index in_bits*stride.
   * @param tr Packed right-control bits, bit i for layer i.
   * @param ys Device output storage. Entry x-first receives this party's share at x.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive element stride between correction-word entries.
   * @pre stride > 0 and cws holds at least in_bits*stride+1 elements.
   * @pre The buffers and referenced backend resources remain valid until work completes.
   * @return Immediate CUDA status. Invalid endpoints, buffers, strides, counts
   * or byte sizes return cudaErrorInvalidValue. Execution completion remains
   * asynchronous. The grid limit cannot be exceeded, because the subdivision
   * caps the block-root depth at 30.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalRangeGpu(bool b, In first, In last, int4 s0,
      cuda::std::span<const int4> cws, In tr, cuda::std::span<int4> ys,
      cudaStream_t stream = 0, size_t stride = 1) const;

  /**
   * Launch full-domain CUDA evaluation from split device arrays.
   *
   * Layer i is read from cws[i*stride] and the terminal from cws[in_bits*stride].
   * Packed bit i of tr is the right-control correction at layer i, and the
   * left-control correction is the LSB of the layer's cws entry, exactly as the
   * component form of Gen() writes them. Bits of tr at and above in_bits are ignored.
   * Backend values, resource rules and tuning rules match EvalAllGpu().
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed, passed by value as a kernel argument.
   * @param cws Device correction words, layer i at index i*stride and
   * the terminal at index in_bits*stride.
   * @param tr Packed right-control bits, bit i for layer i.
   * @param ys Device output storage containing at least 2^in_bits blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive element stride between correction-word entries.
   * @pre stride > 0 and cws holds at least in_bits*stride+1 elements.
   * @pre The buffers and referenced backend resources remain valid until work completes.
   * @return Immediate CUDA status. Invalid dimensions, buffers, strides, null
   * used buffers, or unaddressable sizes return cudaErrorInvalidValue. Execution
   * completion remains asynchronous.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const int4> cws,
      In tr, cuda::std::span<int4> ys, cudaStream_t stream = 0, size_t stride = 1) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Launch full-domain CUDA evaluation for s0s.size() keys from split device arrays.
   *
   * The batch uses the level-major layout: key k layer i is read from
   * cws[i*stride + k], key k's packed control is tr[k], and key k's terminal is
   * cws[in_bits*stride + k]. Packed bit i of tr[k] is the right-control correction
   * at layer i, and the left-control correction is the LSB of the layer's cws
   * entry, exactly as the split form of Gen() writes them. Bits of tr[k] at
   * and above in_bits are ignored.
   * stride selects the layout. 0 selects the dense level-major layout, in
   * which the stride is the key count of this call. Any other value is a
   * positive stride of at least s0s.size(). Uses the same resource, execution,
   * and tuning rules as EvalAllGpu(). All keys share this scheme's backend
   * configuration, and every key produces the whole domain.
   *
   * @param b Party index shared by the batch.
   * @param s0s Device storage containing this party's seed for each key.
   * @param cws Device level-major correction words.
   * @param tr Device packed right-control bits, one value per key.
   * @param ys Device key-major rows of 2^in_bits output blocks per key.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride 0 for the dense level-major layout, where the stride is the
   * key count of this call, or a positive element stride >= s0s.size().
   * @pre stride is 0 or at least s0s.size().
   * @pre cws holds at least in_bits*stride + s0s.size() elements, where a
   * stride of 0 counts as s0s.size(), and tr holds at least s0s.size() values.
   * @pre The buffers and referenced backend resources remain valid until work completes.
   * @return Immediate CUDA status. An empty batch succeeds without reading
   * buffers. Invalid dimensions, null used buffers, strides, or unaddressable
   * sizes return cudaErrorInvalidValue; an oversized grid returns
   * cudaErrorInvalidConfiguration. Execution completion remains asynchronous.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpuBatch(bool b, cuda::std::span<const int4> s0s,
      cuda::std::span<const int4> cws, cuda::std::span<const In> tr,
      cuda::std::span<int4> ys, cudaStream_t stream = 0, size_t stride = 0) const
    requires(in_bits < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  Prg prg_;

  template <int frontier_depth, int threads_per_block>
  struct GpuLaunchShape;

  // EvalAll() shape: a scalar prefix of kBatchPrefix levels grows the frontier to
  // 8 nodes, detail::GenBatch() expands 8 parents per level (pipelined through
  // Gen8() when the PRG offers a batched entry), and a kBatchTail-level tail
  // converts each 8-subtree group of 8 << kBatchTail leaves in local memory.
  static constexpr int kBatchPrefix = 3;
  static constexpr int kBatchTail = 3;

  friend class GrottoDcf<in_bits, Prg, In>;

  // Packed states store the control bit in their least significant bit.
  // correction retains the left control bit in its LSB; right_control is Cw::tr.
  // A backend reference lets CPU and GPU callers reuse the same calculation
  // without reconstructing a Scheme or copying its backend at every node.
  __host__ __device__ static void ExpandNode(Prg &prg, int4 state, int4 correction,
      bool right_control, int4 &left, int4 &right);

  __host__ __device__ static int4 PathStep(Prg &prg, int4 state, int4 correction,
      bool right_control, bool take_right);

  // Correction is a raw int4, or the Group already converted by the CPU batch
  // tail. Raw correction values are converted only when the leaf control bit is set.
  template <typename Correction>
  __host__ __device__ static int4 ConvertLeaf(bool party, int4 state, Correction &correction);

  // Split correction-word view over the caller's arrays. The host methods and the
  // split GPU kernels share this one view; see detail::DpfCwsView.
  using CwsView = detail::DpfCwsView<Cw, In, in_bits>;

  // Writes Cw slots in the legacy AoS layout, covering each full 32B slot.
  struct AosWriter {
    Cw *cws;

    __host__ __device__ void Write(int i, int4 s_cw, bool tr_cw) const {
      // Write both 16B halves so the full 32B slot is covered by unmasked stores.
      // {s_cw, tr_cw} alone leaves the padding undefined: NVCC emits a masked
      // 256-bit store, and the partial-sector write forces a write-allocate
      // round trip per level on GPU (measured ~2x Gen time).
      int4 *cw_slot = reinterpret_cast<int4 *>(&cws[i]);
      cw_slot[0] = s_cw;
      cw_slot[1] = make_int4(tr_cw, 0, 0, 0);
    }

    __host__ __device__ void Finish(int4 terminal) const {
      cws[in_bits] = {terminal, false};
    }
  };

  // Writes strided seed components and accumulates packed right-control bits locally.
  struct CwsWriter {
    int4 *cws;
    size_t stride;
    In &tr;
    In packed_tr = 0;

    __host__ __device__ void Write(int i, int4 s_cw, bool tr_cw) {
      cws[static_cast<size_t>(i) * stride] = s_cw;
      packed_tr |= static_cast<In>(tr_cw) << i;
    }

    __host__ __device__ void Finish(int4 terminal) {
      cws[static_cast<size_t>(in_bits) * stride] = terminal;
      tr = packed_tr;
    }
  };

  // Shared correction generation loop. The writer is passed by value and owns
  // only output access, so the existing algorithm and backend use stay unchanged.
  template <typename Writer>
  __host__ __device__ void Generate(Writer writer, const int4 *s0s, In a, int4 b_buf);

  // Walks x's path from the root and returns the leaf seed with its control bit in the LSB.
  // visit(t, x_bit) sees the control bit of every expanded node and the branch x takes from it.
  template <typename Cws, typename Visit>
  __host__ __device__ int4 Walk(bool b, int4 s0, Cws cws, In x, Visit &&visit);

  // Shared scalar evaluation over the Cw array or the split view.
  template <typename Cws>
  __host__ __device__ int4 EvalCws(bool b, int4 s0, Cws cws, In x);

  // Shared interval evaluation over the Cw array or the split view.
  template <typename Cws>
  void EvalRangeCws(bool b, In first, In last, int4 s0, Cws cws,
      cuda::std::span<int4> ys, int par_depth);

  // Shared full-domain evaluation over the Cw array or the split view.
  template <typename Cws>
  void EvalAllCws(bool b, int4 s0, Cws cws, int4 ys[], int par_depth);

  // Full-domain batch GPU launch over key-major rows of Cw, the storage
  // contract of DMPF bucket keys. Reached through detail::SchemeAccess only;
  // public callers use the level-major split batch API.
  template <int frontier_depth, int threads_per_block>
  __host__ cudaError_t EvalAllGpuBatchCw(bool b, cuda::std::span<const int4> s0s,
      cuda::std::span<const Cw> cws, cuda::std::span<int4> ys, cudaStream_t stream) const;

  // Closed node bounds avoid representing an unaddressable full-domain end.
  template <typename Cws>
  void EvalRangeSerial(bool b, In first, In last, int4 s0, Cws cws, int4 ys[]);

  // GCC 12.4.0 can discard these writes before the OpenMP walk after IPA
  // effect analysis. Keep this boundary opaque on GCC 12 releases before 12.5.
  // GCC PR115033 describes a related closure escape-analysis defect.
  template <typename Cws>
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 12 && __GNUC_MINOR__ <= 4
  __attribute__((noipa))
#endif
  std::vector<int4> BuildParallelFrontier(int4 st, Cws cws, int split_depth);

  // Expands w scalar frontier nodes (t in LSB) by one level with cws[lv].
  // The children overwrite the frontier in place: iteration is in reverse so
  // the writes at 2j, 2j+1 never clobber nodes still to be read.
  template <typename Cws>
  void ExpandFrontier(int4 frontier[], size_t w, Cws cws, int lv);

  // Applies the level correction to one expanded child branchlessly.
  // m is 0 or -1 from the parent control bit t, and t_cw is the cw control bit
  // masked by t. The cw seed is clamped, so the XOR leaves the child LSB free
  // for the packed control bit.
  __host__ __device__ static int4 CorrectChild(int4 child, int4 s_cw, int m, int t_cw);

  // Full-domain evaluation of the subtree rooted at st (t in LSB) at level i through an
  // 8-wide batched frontier: a kBatchPrefix-level scalar prefix, batched expansions, and
  // a kBatchTail-level tail converted in local memory.
  // Requires in_bits - i >= kBatchPrefix + kBatchTail.
  template <typename Cws>
  void EvalBatched8(bool b, int4 st, Cws cws, int4 ys[], size_t l, int i);

  // Evaluates a group of 8 consecutive frontier nodes (t in LSB) at level i, covering the
  // leaves [l, l + 8 << (in_bits - i)) in order. Requires in_bits - i >= kBatchTail.
  template <typename Cws>
  void EvalBatched8Group(bool b, const int4 (&frontier)[8], Cws cws, int4 ys[], size_t l, int i);
};

// -----------------------------------------------------------------------------

namespace gpu::detail {

// Hold the scheme behind a tag so wide Group non-type template parameters stay
// out of CUDA host stub signatures.
template <int n, typename GroupTag, typename Prg, typename In>
struct DpfGpuKernelDescriptor {
  using Scheme = Dpf<n, typename GroupTag::type, Prg, In>;
  using Backend = Prg;
};

// Point evaluation also needs the exact control and query types. They stay in
// the descriptor so kernel signatures carry only narrow template arguments.
template <int n, typename GroupTag, typename Prg, typename In, typename C, typename Q>
struct DpfPointGpuKernelDescriptor : DpfGpuKernelDescriptor<n, GroupTag, Prg, In> {
  using Control = C;
  using Query = Q;
};

template <typename Scheme>
struct DpfGpuSchemeTraits;

#if defined(__CUDACC__)

// The single-key and batch kernels share the same block-local frontier and tail.
// Each node keeps the former full-domain helper's by-value backend lifetime.
// Cws is a Cw pointer on the array path and a kernel-boundary split view
// on the split path, so the traversal body does not depend on the layout.
template <typename Scheme, int frontier_depth, int block_root_depth, bool range = false, typename Cws>
__device__ __forceinline__ void DpfEvalAllBlock(bool b, int4 s0, Cws cws,
    int4 *ys, typename DpfGpuSchemeTraits<Scheme>::Backend &prg, int subtree_block_index, int64_t thread_index, int4 *shared_frontier,
    __uint128_t first = 0, int subtree_depth = DpfGpuSchemeTraits<Scheme>::kInputBits,
    int range_frontier = frontier_depth, int range_block_root = block_root_depth) {
  constexpr int kInputBits = DpfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kTailCapacity = range ? 8 : kInputBits - frontier_depth;
  const int start_depth = range ? kInputBits - subtree_depth : 0;
  const int frontier = range ? start_depth + range_frontier : frontier_depth;
  const int root_depth = range ? start_depth + range_block_root : block_root_depth;
  const int tail_depth = kInputBits - frontier;
  const int block_subtree_depth = frontier - root_depth;
  // Phase 0: block root walk (top block_root_depth levels) by thread 0.
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < root_depth; ++i) {
      bool x_bit = i < start_depth
          ? ((first >> (kInputBits - 1 - i)) & 1) != 0
          : ((subtree_block_index >> (root_depth - 1 - i)) & 1) != 0;
      {
        // Preserve the old by-value path helper's copy and destruction boundary.
        auto node_prg = prg;
        st = fss::detail::SchemeAccess::DpfPathStep<Scheme>(node_prg, st, cws[i].s, cws[i].tr, x_bit);
      }
    }
    shared_frontier[0] = st;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels block_root_depth..frontier_depth.
#pragma unroll
  for (int i = 0; i < block_subtree_depth; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = {0, 0, 0, 0};
    if (active) n = shared_frontier[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      {
        // Preserve the old by-value expansion helper's copy and destruction boundary.
        auto node_prg = prg;
        fss::detail::SchemeAccess::DpfExpandNode<Scheme>(node_prg, n, cws[root_depth + i].s, cws[root_depth + i].tr, left, right);
      }
      shared_frontier[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  if constexpr (range) {
    int4 pending[8];
    int4 node = shared_frontier[threadIdx.x];
    int level = frontier;
    const int count = 1 << tail_depth;
    const size_t offset = static_cast<size_t>(thread_index) << tail_depth;
    for (int index = 0; index < count; ++index) {
      while (level < kInputBits) {
        int4 left, right;
        {
          auto node_prg = prg;
          fss::detail::SchemeAccess::DpfExpandNode<Scheme>(node_prg, node,
              cws[level].s, cws[level].tr, left, right);
        }
        pending[level - frontier] = right;
        node = left;
        ++level;
      }
      ys[offset + index] = fss::detail::SchemeAccess::DpfConvertLeaf<Scheme>(
          b, node, cws[kInputBits].s);
      if (index + 1 < count) {
        int next = index + 1;
        level = kInputBits - 1;
        while ((next & 1) == 0) {
          --level;
          next >>= 1;
        }
        node = pending[level - frontier];
        ++level;
      }
    }
  } else {
    // Phase 2: expand the subtree below depth frontier_depth, then convert to leaves.
    int4 nodes[1 << kTailCapacity];
    nodes[0] = shared_frontier[threadIdx.x];
    int count = 1;
#pragma unroll
    for (int lev = frontier; lev < kInputBits; ++lev) {
#pragma unroll
      for (int k = count - 1; k >= 0; --k) {
        int4 left, right;
        {
          // Preserve the old by-value expansion helper's copy and destruction boundary.
          auto node_prg = prg;
          fss::detail::SchemeAccess::DpfExpandNode<Scheme>(node_prg, nodes[k], cws[lev].s, cws[lev].tr, left, right);
        }
        nodes[2 * k] = left;
        nodes[2 * k + 1] = right;
      }
      count *= 2;
    }

    int4 v_cw_np1 = cws[kInputBits].s;
    const size_t output_offset = static_cast<size_t>(thread_index) << tail_depth;
#pragma unroll
    for (int k = 0; k < count; ++k) {
      ys[output_offset + k] = fss::detail::SchemeAccess::DpfConvertLeaf<Scheme>(b, nodes[k], v_cw_np1);
    }
  }
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void DpfEvalAllBatchKernel(bool b, const int4 *s0s, const void *cws_buf,
    int4 *ys, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  const auto *cws = static_cast<const typename Scheme::Cw *>(cws_buf);
  constexpr int kInputBits = DpfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  const auto *key_cws = cws + static_cast<size_t>(key_index) * (kInputBits + 1);
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * (size_t{1} << kInputBits);
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  DpfEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index], key_cws, key_outputs, prg, subtree_block_index, thread_index, shared_frontier);
}

// Split-layout counterparts of the three kernels above. The correction buffers
// arrive type-erased and the boundary view reconstructs the per-layer Cw
// entries, so the shared traversal stays identical on both paths.
template <typename Descriptor, int threads_per_block>
__global__ void DpfSplitEvalRangeKernel(bool b, int4 s0,
    const void *cws_buf, typename DpfGpuSchemeTraits<typename Descriptor::Scheme>::Input tr,
    size_t stride, int4 *ys, __uint128_t first,
    int depth, int frontier, int root, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::DpfCwsView<typename Scheme::Cw,
      typename DpfGpuSchemeTraits<Scheme>::Input,
      DpfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  __shared__ int4 states[threads_per_block];
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  DpfEvalAllBlock<Scheme, 0, 0, true>(b, s0, Cws{cws, tr, stride}, ys, prg,
      blockIdx.x, thread_index, states, first, depth, frontier, root);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void DpfSplitEvalAllKernel(bool b, int4 s0, const void *cws_buf,
    typename DpfGpuSchemeTraits<typename Descriptor::Scheme>::Input tr, size_t stride,
    int4 *ys, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::DpfCwsView<typename Scheme::Cw,
      typename DpfGpuSchemeTraits<Scheme>::Input,
      DpfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  DpfEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0, Cws{cws, tr, stride},
      ys, prg, blockIdx.x, thread_index, shared_frontier);
}

template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void DpfSplitEvalAllBatchKernel(bool b, const int4 *s0s, const void *cws_buf,
    const typename DpfGpuSchemeTraits<typename Descriptor::Scheme>::Input *trs, size_t stride,
    int4 *ys, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::DpfCwsView<typename Scheme::Cw,
      typename DpfGpuSchemeTraits<Scheme>::Input,
      DpfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  constexpr int kInputBits = DpfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * (size_t{1} << kInputBits);
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  DpfEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      Cws{cws + key_index, trs[key_index], stride}, key_outputs, prg,
      subtree_block_index, thread_index, shared_frontier);
}

// Device-key counterparts of the split kernels above: the party seed and the
// packed control live in caller device storage and are loaded here, so no host
// value of the key material exists. They feed the same shared traversal.
template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void DpfSplitEvalAllDeviceKeyKernel(bool b, const int4 *s0,
    const void *cws_buf, const typename DpfGpuSchemeTraits<typename Descriptor::Scheme>::Input *tr,
    size_t stride, int4 *ys, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::DpfCwsView<typename Scheme::Cw,
      typename DpfGpuSchemeTraits<Scheme>::Input,
      DpfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  DpfEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0[0],
      Cws{cws, tr[0], stride}, ys, prg, blockIdx.x, thread_index, shared_frontier);
}

template <typename Descriptor, int threads_per_block>
__global__ void DpfSplitEvalRangeDeviceKeyKernel(bool b, const int4 *s0,
    const void *cws_buf, const typename DpfGpuSchemeTraits<typename Descriptor::Scheme>::Input *tr,
    size_t stride, int4 *ys, __uint128_t first,
    int depth, int frontier, int root, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Cws = fss::detail::DpfCwsView<typename Scheme::Cw,
      typename DpfGpuSchemeTraits<Scheme>::Input,
      DpfGpuSchemeTraits<Scheme>::kInputBits>;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  __shared__ int4 states[threads_per_block];
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  DpfEvalAllBlock<Scheme, 0, 0, true>(b, s0[0], Cws{cws, tr[0], stride}, ys, prg,
      blockIdx.x, thread_index, states, first, depth, frontier, root);
}

// Key-major batch counterpart: key k's components occupy records
// [k*key_records, k*key_records + in_bits] and its control record follows, so
// the per-key view uses a unit stride.
template <typename Descriptor, int frontier_depth, int block_root_depth>
__global__ void DpfSplitEvalAllBatchDeviceKeyKernel(bool b, const int4 *s0s,
    const void *cws_buf, size_t key_records, size_t control_record,
    size_t key_count, int4 *ys, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using In = typename DpfGpuSchemeTraits<Scheme>::Input;
  using Cws = fss::detail::DpfCwsView<typename Scheme::Cw, In,
      DpfGpuSchemeTraits<Scheme>::kInputBits>;
  constexpr int kInputBits = DpfGpuSchemeTraits<Scheme>::kInputBits;
  constexpr int kBlocksPerKey = 1 << block_root_depth;
  const auto *cws = static_cast<const int4 *>(cws_buf);
  const int key_index = blockIdx.x / kBlocksPerKey;
  const int subtree_block_index = blockIdx.x - key_index * kBlocksPerKey;
  const int64_t thread_index = static_cast<int64_t>(subtree_block_index) * blockDim.x + threadIdx.x;
  if (thread_index >= (1LL << frontier_depth)) return;
  const int4 *key = cws + static_cast<size_t>(key_index) * key_records;
  const In *tr = reinterpret_cast<const In *>(key + control_record);
  int4 *key_outputs = ys + static_cast<size_t>(key_index) * (size_t{1} << kInputBits);
  __shared__ int4 shared_frontier[1 << (frontier_depth - block_root_depth)];
  DpfEvalAllBlock<Scheme, frontier_depth, block_root_depth>(b, s0s[key_index],
      Cws{key, tr[0], 1}, key_outputs, prg, subtree_block_index, thread_index, shared_frontier);
}

// Host launchers for the device-key kernels. The Prg is passed by value from a
// host mirror of the device scheme, matching the public launch methods.
template <int in_bits, typename Group, typename Prg, typename In, int frontier_depth = 16,
    int threads_per_block = 256>
__host__ inline cudaError_t DpfEvalAllGpuDeviceKey(bool b, const int4 *s0,
    const int4 *cws, size_t cws_count, const In *tr, size_t stride, int4 *ys, size_t ys_count,
    const Prg &prg, cudaStream_t stream) {
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr int kFrontierDepth = frontier_depth < in_bits ? frontier_depth : in_bits;
  if (stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(in_bits) ||
      cws_count < static_cast<size_t>(in_bits) * stride + 1 || ys_count < kOutputsPerKey ||
      s0 == nullptr || cws == nullptr || tr == nullptr || ys == nullptr ||
      kOutputsPerKey > kMaxElements)
    return cudaErrorInvalidValue;
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
  using Descriptor = DpfGpuKernelDescriptor<in_bits, typename fss::detail::GpuGroupTag<Group>::Type, Prg, In>;
  DpfSplitEvalAllDeviceKeyKernel<Descriptor, kFrontierDepth, kBlockRootDepth>
      <<<size_t{1} << kBlockRootDepth, 1 << kThreadsPerBlockLog2, 0, stream>>>(
          b, s0, cws, tr, stride, ys, prg);
  return cudaGetLastError();
}

template <int in_bits, typename Group, typename Prg, typename In, int frontier_depth = 16,
    int threads_per_block = 256>
__host__ inline cudaError_t DpfEvalRangeGpuDeviceKey(bool b, In first, In last,
    const int4 *s0, const int4 *cws, size_t cws_count, const In *tr, size_t stride,
    int4 *ys, size_t ys_count, const Prg &prg, cudaStream_t stream) {
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(in_bits) ||
      cws_count < static_cast<size_t>(in_bits) * stride + 1 || cws == nullptr ||
      s0 == nullptr || tr == nullptr || ys == nullptr) return cudaErrorInvalidValue;
  using Descriptor = DpfGpuKernelDescriptor<in_bits, typename fss::detail::GpuGroupTag<Group>::Type, Prg, In>;
  return fss::detail::SplitRangeChunks<in_bits, frontier_depth, threads_per_block>(
      first, last, ys_count,
      [&](In chunk_first, int depth, size_t offset, int frontier, int thread_bits, int root,
          size_t blocks) {
        DpfSplitEvalRangeDeviceKeyKernel<Descriptor, threads_per_block>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, s0, cws, tr, stride,
                ys + offset, chunk_first, depth, frontier, root, prg);
        return cudaGetLastError();
      });
}

template <int in_bits, typename Group, typename Prg, typename In, int frontier_depth = 16,
    int threads_per_block = 256>
__host__ inline cudaError_t DpfEvalAllGpuBatchDeviceKey(bool b, const int4 *s0s,
    size_t key_count, const int4 *cws, size_t cws_count, size_t key_records,
    size_t control_record, int4 *ys, size_t ys_count, const Prg &prg, cudaStream_t stream) {
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
  using Descriptor = DpfGpuKernelDescriptor<in_bits, typename fss::detail::GpuGroupTag<Group>::Type, Prg, In>;
  DpfSplitEvalAllBatchDeviceKeyKernel<Descriptor, kFrontierDepth, kBlockRootDepth>
      <<<key_count * kBlocksPerKey, 1 << kThreadsPerBlockLog2, 0, stream>>>(
          b, s0s, cws, key_records, control_record, key_count, ys, prg);
  return cudaGetLastError();
}

// Correction words arrive in the dense level-major split layout: layer i of
// key k is cws[i * key_count + k], the terminal layer is
// cws[in_bits * key_count + k], and tr[k] holds key k's packed right-control
// bits, bit i for layer i.
template <typename Descriptor, int unroll, bool b>
__global__ void DpfCoreEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cws,
    const typename Descriptor::Control *tr, const int4 *out_cw,
    const typename Descriptor::Query *xs, int key_count,
    typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Control = typename Descriptor::Control;
  using In = typename DpfGpuSchemeTraits<Scheme>::Input;
  constexpr int kInputBits = DpfGpuSchemeTraits<Scheme>::kInputBits;
  static_assert(kInputBits <= sizeof(In) * 8, "In too narrow for kInputBits");
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index >= key_count) return;
  int4 state = util::SetLsb(seeds[thread_index], b);
  In x = static_cast<In>(xs[thread_index]);
  Control ex = tr[thread_index];
#pragma unroll unroll
  for (int i = 0; i < kInputBits; ++i) {
    bool take_right = ((x >> (kInputBits - 1 - i)) & 1) != 0;
    bool right_control = ((ex >> i) & 1) != 0;
    state = fss::detail::SchemeAccess::DpfPathStep<Scheme>(
        prg, state, cws[static_cast<size_t>(i) * key_count + thread_index], right_control, take_right);
  }
  ys[thread_index] = fss::detail::SchemeAccess::DpfConvertLeaf<Scheme>(b, state, out_cw[thread_index]);
}

// The scheme is decomposed into its template arguments here so the kernel
// launch spells narrow descriptor arguments instead of the wide Dpf type.
template <typename Scheme, typename Control, typename Query>
struct DpfPointGpuLauncher;

template <int n, typename Group, typename Prg, typename In, typename Control, typename Query>
struct DpfPointGpuLauncher<Dpf<n, Group, Prg, In>, Control, Query> {
  template <int unroll>
  static void Launch(bool b, const int4 *seeds, const int4 *cws, const Control *tr,
      const int4 *out_cw, const Query *xs, int4 *ys, int key_count, Prg prg,
      cudaStream_t stream) {
    using Descriptor = DpfPointGpuKernelDescriptor<n,
        typename fss::detail::GpuGroupTag<Group>::Type, Prg, In, Control, Query>;
    constexpr int kThreadsPerBlock = 256;
    int block_count = (key_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (b) {
      DpfCoreEvalPointKernel<Descriptor, unroll, true>
          <<<block_count, kThreadsPerBlock, 0, stream>>>(ys, seeds, cws, tr, out_cw, xs, key_count, prg);
    } else {
      DpfCoreEvalPointKernel<Descriptor, unroll, false>
          <<<block_count, kThreadsPerBlock, 0, stream>>>(ys, seeds, cws, tr, out_cw, xs, key_count, prg);
    }
  }
};

template <int unroll, typename Scheme, typename Control, typename Query>
void DpfCoreEvalPointLaunch(bool b, const int4 *seeds, const int4 *cws, const Control *tr,
    const int4 *out_cw, const Query *xs, int4 *ys, int key_count,
    typename DpfGpuSchemeTraits<Scheme>::Backend prg, cudaStream_t stream) {
  if (key_count == 0) return;
  DpfPointGpuLauncher<Scheme, Control, Query>::template Launch<unroll>(
      b, seeds, cws, tr, out_cw, xs, ys, key_count, prg, stream);
}

/**
 * GPU point evaluation of nkeys DPF keys: ys[k] = Eval(b, seeds[k], cws, xs[k]).
 * One thread per key. cws holds dense level-major correction words, the layout
 * the split form of Gen() writes: layer i of key k is cws[i * nkeys + k] and
 * the terminal layer is cws[in_bits * nkeys + k]. tr[k] carries key k's packed
 * right-control bits, bit i for layer i.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds (device memory), one per key.
 * @param cws Level-major correction words (device memory), (in_bits + 1) * nkeys entries.
 * @param tr Packed right-control bits (device memory), one per key.
 * @param xs Evaluation points (device memory), one per key.
 * @param ys Output array (device memory), one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A Dpf instance (holds the PRG).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, typename Query = In>
  requires QueryFor<Query, in_bits>
void DpfEvalPointGpu(bool b, cuda::std::span<const int4> seeds, cuda::std::span<const int4> cws,
    cuda::std::span<const In> tr, cuda::std::span<const Query> xs, int4 *ys, int nkeys,
    const Dpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  if (nkeys < 0)
    throw std::invalid_argument("DpfEvalPointGpu: nkeys must be nonnegative");
  if (nkeys == 0) return;
  const size_t count = static_cast<size_t>(nkeys);
  constexpr size_t kLevels = static_cast<size_t>(in_bits) + 1;
  if (count > SIZE_MAX / kLevels || cws.size() < count * kLevels)
    throw std::invalid_argument("DpfEvalPointGpu: cws must hold (in_bits+1)*nkeys elements");
  if (seeds.size() < count || tr.size() < count || xs.size() < count || ys == nullptr)
    throw std::invalid_argument("DpfEvalPointGpu: seeds, tr, xs, and ys must cover nkeys entries");
  const int4 *out_cw = cws.data() + static_cast<size_t>(in_bits) * static_cast<size_t>(nkeys);
  DpfCoreEvalPointLaunch<unroll, Dpf<in_bits, Group, Prg, In>, In, Query>(b, seeds.data(), cws.data(), tr.data(),
      out_cw, xs.data(), ys, nkeys, fss::detail::SchemeAccess::GetPrg(dpf), stream);
}

#endif  // defined(__CUDACC__)

}  // namespace gpu::detail

namespace gpu::detail {

template <int n, typename Group, typename Prg, typename In>
struct DpfGpuSchemeTraits<Dpf<n, Group, Prg, In>> {
  using Backend = Prg;
  using Input = In;
  static constexpr int kInputBits = n;
};

}  // namespace gpu::detail

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
__host__ __device__ inline Dpf<in_bits, Group, Prg, In>::Dpf(Prg prg)
    : prg_(prg) {}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
__host__ __device__ inline void Dpf<in_bits, Group, Prg, In>::Gen(
    cuda::std::span<int4> cws, In &tr, cuda::std::span<const int4, 2> s0s,
    In a, int4 b_buf, size_t stride) {
  util::RequireArg(stride > 0, "Dpf::Gen: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits),
      "Dpf::Gen: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  util::RequireArg(cws.size() >= required,
      "Dpf::Gen: cws must hold in_bits*stride+1 elements");
  Generate(CwsWriter{cws.data(), stride, tr}, s0s.data(), a, b_buf);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Writer>
__host__ __device__ inline void Dpf<in_bits, Group, Prg, In>::Generate(
    Writer writer, const int4 *s0s, In a, int4 b_buf) {
  int4 s0 = s0s[0];
  s0 = util::SetLsb(s0, false);
  int4 s1 = s0s[1];
  s1 = util::SetLsb(s1, false);
  bool t0 = false;
  bool t1 = true;
  b_buf = util::SetLsb(b_buf, false);

  for (int i = 0; i < in_bits; ++i) {
    auto [s0l, s0r] = prg_.Gen(s0);
    auto [s1l, s1r] = prg_.Gen(s1);

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
    writer.Write(i, s_cw, tr_cw);
  }

  auto v_cw_np1 = Group::From(b_buf) + (-Group::From(s0)) + Group::From(s1);
  if (t1) v_cw_np1 = -v_cw_np1;
  writer.Finish(v_cw_np1.Into());
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
__host__ __device__ inline int4 Dpf<in_bits, Group, Prg, In>::Eval(bool b, int4 s0,
    cuda::std::span<const int4> cws, In tr, In x, size_t stride) {
  util::RequireArg(stride > 0, "Dpf::Eval: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(in_bits),
      "Dpf::Eval: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  util::RequireArg(cws.size() >= required,
      "Dpf::Eval: cws must hold in_bits*stride+1 elements");
  return EvalCws(b, s0, CwsView{cws.data(), tr, stride}, x);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
__host__ __device__ inline int4 Dpf<in_bits, Group, Prg, In>::EvalCws(
    bool b, int4 s0, Cws cws, In x) {
  int4 state = Walk(b, s0, cws, x, [](bool, bool) {});
  const int4 terminal = cws[in_bits].s;
  util::RequireArg((terminal.w & 1) == 0,
      "Dpf::Eval: the terminal group value must have a zero LSB");
  return ConvertLeaf(b, state, terminal);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
inline void Dpf<in_bits, Group, Prg, In>::EvalRange(bool b, In first, In last,
    int4 s0, cuda::std::span<const int4> cws, In tr, cuda::std::span<int4> ys,
    int par_depth, size_t stride) {
  if (stride == 0) throw std::invalid_argument("Dpf::EvalRange: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits))
    throw std::invalid_argument("Dpf::EvalRange: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required || cws.data() == nullptr)
    throw std::invalid_argument("Dpf::EvalRange: cws must hold in_bits*stride+1 elements");
  EvalRangeCws(b, first, last, s0, CwsView{cws.data(), tr, stride},
      ys, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline void Dpf<in_bits, Group, Prg, In>::EvalRangeCws(bool b, In first, In last,
    int4 s0, Cws cws, cuda::std::span<int4> ys, int par_depth) {
  assert(par_depth >= 0);
  if (first > last)
    throw std::invalid_argument("Dpf::EvalRange: first must not exceed last");
  if constexpr (in_bits < sizeof(In) * 8) {
    if ((last >> in_bits) != 0)
      throw std::invalid_argument("Dpf::EvalRange: last must stay inside the input domain");
  }
  const __uint128_t difference = static_cast<__uint128_t>(last) - first;
  if (difference >= std::numeric_limits<size_t>::max() / sizeof(int4))
    throw std::invalid_argument("Dpf::EvalRange: the interval is too large for size_t");
  const size_t count = static_cast<size_t>(difference) + 1;
  if (ys.size() < count || ys.data() == nullptr)
    throw std::invalid_argument("Dpf::EvalRange: ys must hold last-first+1 entries");
  util::RequireArg((cws[in_bits].s.w & 1) == 0,
      "Dpf::EvalRange: the terminal group value must have a zero LSB");
  const size_t partitions = detail::ResolveParPartitions(par_depth, count);
  detail::ParallelFor(par_depth, partitions, [&](size_t partition) {
    const size_t remainder = count % partitions;
    const size_t begin = partition * (count / partitions) +
        (partition < remainder ? partition : remainder);
    const size_t length = count / partitions + (partition < remainder);
    const In partition_first = first + static_cast<In>(begin);
    const In partition_last = partition_first + static_cast<In>(length - 1);
    EvalRangeSerial(b, partition_first, partition_last, s0, cws, ys.data() + begin);
  });
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline void Dpf<in_bits, Group, Prg, In>::EvalRangeSerial(bool b, In first, In last,
    int4 s0, Cws cws, int4 ys[]) {
  constexpr In domain_last = [] {
    if constexpr (in_bits == sizeof(In) * 8) return static_cast<In>(~In{0});
    else return static_cast<In>((In{1} << in_bits) - 1);
  }();
  struct Frame {
    int4 state;
    In low;
    In high;
    int level;
  };
  cuda::std::array<Frame, in_bits + 1> stack;
  size_t pending = 1;
  stack[0] = {util::SetLsb(s0, b), In{0}, domain_last, 0};
  while (pending != 0) {
    const Frame node = stack[--pending];
    if (node.level == in_bits) {
      const int4 terminal = cws[in_bits].s;
      ys[static_cast<size_t>(node.low - first)] = ConvertLeaf(b, node.state, terminal);
      continue;
    }
    const In middle = node.low + ((node.high - node.low) >> 1);
    int4 left;
    int4 right;
    ExpandNode(prg_, node.state, cws[node.level].s, cws[node.level].tr, left, right);
    if (last > middle) {
      assert(pending < stack.size());
      stack[pending++] = {right, static_cast<In>(middle + 1), node.high, node.level + 1};
    }
    if (first <= middle) {
      assert(pending < stack.size());
      stack[pending++] = {left, node.low, middle, node.level + 1};
    }
  }
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
inline void Dpf<in_bits, Group, Prg, In>::EvalAll(bool b, int4 s0,
    cuda::std::span<const int4> cws, In tr, int4 ys[], int par_depth, size_t stride) {
  if (stride == 0) throw std::invalid_argument("Dpf::EvalAll: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits))
    throw std::invalid_argument("Dpf::EvalAll: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required)
    throw std::invalid_argument("Dpf::EvalAll: cws must hold in_bits*stride+1 elements");
  EvalAllCws(b, s0, CwsView{cws.data(), tr, stride}, ys, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline void Dpf<in_bits, Group, Prg, In>::EvalAllCws(
    bool b, int4 s0, Cws cws, int4 ys[], int par_depth) {
  assert(par_depth >= 0);
  static_assert(detail::FitsFullDomain<int4, in_bits>(),
      "full-domain output count and byte size must fit in size_t");

  // The terminal LSB is a caller precondition. Check it on the calling thread,
  // so a Debug violation aborts there instead of inside an OpenMP worker;
  // Release builds do not check it.
  util::RequireArg((cws[in_bits].s.w & 1) == 0,
      "Dpf::EvalAll: the terminal group value must have a zero LSB");

  if constexpr (in_bits < kBatchPrefix + kBatchTail) {
    // Domains too small to fill an 8-wide frontier evaluate per input; the
    // point walk is cheaper than the batched machinery at this size.
    constexpr size_t n = size_t{1} << in_bits;
    const int4 terminal = cws[in_bits].s;
    fss::detail::ParallelFor(par_depth, n, [&](size_t x) {
      const int4 state = Walk(b, s0, cws, static_cast<In>(x), [](bool, bool) {});
      ys[x] = ConvertLeaf(b, state, terminal);
    });
    return;
  }

  int4 st = s0;
  st = util::SetLsb(st, b);

  int par_depth_ = fss::detail::ResolveParDepth(par_depth, in_bits);

  int split_depth = par_depth_;
  if (split_depth > in_bits - kBatchPrefix - kBatchTail) split_depth = in_bits - kBatchPrefix - kBatchTail;
  if (split_depth > 0) {
    // Split at a frontier of 2 ** split_depth subtree roots instead of per-node tasks,
    // so the sequential batched walk stays the fast default path.
    size_t num_tasks = size_t{1} << split_depth;
    std::vector<int4> frontier = BuildParallelFrontier(st, cws, split_depth);
    size_t span = (size_t{1} << in_bits) / num_tasks;
    fss::detail::ParallelFor(par_depth_, num_tasks, [&](size_t task) {
      EvalBatched8(b, frontier[task], cws, ys, task * span, split_depth);
    });
  } else {
    EvalBatched8(b, st, cws, ys, 0, 0);
  }
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
__host__ __device__ inline void Dpf<in_bits, Group, Prg, In>::ExpandNode(
    Prg &prg, int4 state, int4 correction, bool right_control, int4 &left, int4 &right) {
  int control = static_cast<int>(util::GetLsb(state));
  auto [raw_left, raw_right] = prg.Gen(util::SetLsb(state, false));
  int left_control = static_cast<int>(util::GetLsb(correction));
  int4 seed_correction = util::SetLsb(correction, false);
  left = CorrectChild(raw_left, seed_correction, -control, control & left_control);
  right = CorrectChild(raw_right, seed_correction, -control, control & static_cast<int>(right_control));
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
__host__ __device__ inline int4 Dpf<in_bits, Group, Prg, In>::PathStep(
    Prg &prg, int4 state, int4 correction, bool right_control, bool take_right) {
  int control = static_cast<int>(util::GetLsb(state));
  auto [raw_child] = detail::GenHalf<2>(prg, util::SetLsb(state, false), take_right);
  int child_control = static_cast<int>(take_right ? right_control : util::GetLsb(correction));
  return CorrectChild(raw_child, util::SetLsb(correction, false),
      -control, control & child_control);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Correction>
__host__ __device__ inline int4 Dpf<in_bits, Group, Prg, In>::ConvertLeaf(
    bool party, int4 state, Correction &correction) {
  auto value = Group::From(util::SetLsb(state, false));
  if (util::GetLsb(state)) {
    if constexpr (std::is_same_v<std::remove_cv_t<Correction>, int4>) value = value + Group::From(correction);
    else value = value + correction;
  }
  if (party) value = -value;
  return value.Into();
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws, typename Visit>
__host__ __device__ inline int4 Dpf<in_bits, Group, Prg, In>::Walk(
    bool b, int4 s0, Cws cws, In x, Visit &&visit) {
  int4 state = util::SetLsb(s0, b);
  for (int i = 0; i < in_bits; ++i) {
    bool x_bit = (x >> (in_bits - 1 - i)) & 1;
    visit(util::GetLsb(state), x_bit);
    state = PathStep(prg_, state, cws[i].s, cws[i].tr, x_bit);
  }
  return state;
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline std::vector<int4> Dpf<in_bits, Group, Prg, In>::BuildParallelFrontier(
    int4 st, Cws cws, int split_depth) {
  std::vector<int4> frontier(size_t{1} << split_depth);
  frontier[0] = st;
  for (int lv = 0; lv < split_depth; ++lv) ExpandFrontier(frontier.data(), size_t{1} << lv, cws, lv);
  return frontier;
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline void Dpf<in_bits, Group, Prg, In>::ExpandFrontier(int4 frontier[], size_t w, Cws cws, int lv) {
  int4 correction = cws[lv].s;
  bool right_control = cws[lv].tr;
  for (size_t j = w; j-- > 0;) {
    int4 left, right;
    ExpandNode(prg_, frontier[j], correction, right_control, left, right);
    frontier[2 * j] = left;
    frontier[2 * j + 1] = right;
  }
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
__host__ __device__ inline int4 Dpf<in_bits, Group, Prg, In>::CorrectChild(int4 child, int4 s_cw, int m, int t_cw) {
  int4 res = {child.x ^ (s_cw.x & m), child.y ^ (s_cw.y & m), child.z ^ (s_cw.z & m), child.w ^ (s_cw.w & m)};
  res.w = (res.w & ~1) | ((child.w ^ t_cw) & 1);
  return res;
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline void Dpf<in_bits, Group, Prg, In>::EvalBatched8(
    bool b, int4 st, Cws cws, int4 ys[], size_t l, int i) {
  int4 frontier[8];
  frontier[0] = st;
  for (int lv = i; lv < i + kBatchPrefix; ++lv) ExpandFrontier(frontier, size_t{1} << (lv - i), cws, lv);
  EvalBatched8Group(b, frontier, cws, ys, l, i + kBatchPrefix);
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <typename Cws>
inline void Dpf<in_bits, Group, Prg, In>::EvalBatched8Group(
    bool b, const int4 (&frontier)[8], Cws cws, int4 ys[], size_t l, int i) {
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
        auto out = detail::GenBatch<2>(prg_, seeds);
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
    // The shared entry validated the terminal before parallel dispatch.
    assert((v_cw_np1.w & 1) == 0);
    Group v_cw = Group::From(v_cw_np1);
    for (int j = 0; j < w; ++j) ys[l + j] = ConvertLeaf(b, cur[j], v_cw);
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
  auto out = detail::GenBatch<2>(prg_, seeds);
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

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
struct Dpf<in_bits, Group, Prg, In>::GpuLaunchShape {
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

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dpf<in_bits, Group, Prg, In>::EvalAllGpuBatchCw(
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::DpfGpuKernelDescriptor<in_bits, GroupTag, Prg, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kCwPerKey = in_bits + 1;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  size_t key_count = s0s.size();
  if (key_count == 0) return cudaSuccess;
  if (key_count > kMaxAddressableSize / kCwPerKey / sizeof(Cw) || key_count > kMaxAddressableSize / kOutputsPerKey / sizeof(int4) ||
      cws.size() < key_count * kCwPerKey || ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || ys.data() == nullptr) return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::DpfEvalAllBatchKernel<Descriptor, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), ys.data(), prg_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dpf<in_bits, Group, Prg, In>::EvalRangeGpu(
    bool b, In first, In last, int4 s0, cuda::std::span<const int4> cws, In tr,
    cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const {
#if defined(__CUDACC__)
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::DpfGpuKernelDescriptor<in_bits, GroupTag, Prg, In>;
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 || (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits)) ||
      cws.size() < static_cast<size_t>(in_bits) * stride + 1 ||
      cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  return detail::SplitRangeChunks<in_bits, frontier_depth, threads_per_block>(
      first, last, ys.size(),
      [&](In chunk_first, int depth, size_t offset, int frontier, int thread_bits, int root,
          size_t blocks) {
        gpu::detail::DpfSplitEvalRangeKernel<Descriptor, threads_per_block>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, s0, cws.data(), tr, stride,
                ys.data() + offset, chunk_first, depth, frontier, root, prg_);
        return cudaGetLastError();
      });
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dpf<in_bits, Group, Prg, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const int4> cws, In tr,
    cuda::std::span<int4> ys, cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::DpfGpuKernelDescriptor<in_bits, GroupTag, Prg, In>;
  constexpr size_t kOutputsPerKey = size_t{1} << in_bits;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (stride == 0 ||
      (stride > (kMaxElements - 1) / static_cast<size_t>(in_bits)))
    return cudaErrorInvalidValue;
  const size_t required = static_cast<size_t>(in_bits) * stride + 1;
  if (cws.size() < required || ys.size() < kOutputsPerKey ||
      cws.data() == nullptr || ys.data() == nullptr ||
      kOutputsPerKey > std::numeric_limits<size_t>::max() / sizeof(int4)) return cudaErrorInvalidValue;
  gpu::detail::DpfSplitEvalAllKernel<Descriptor, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), tr, stride, ys.data(), prg_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dpf<in_bits, Group, Prg, In>::EvalAllGpuBatch(
    bool b, cuda::std::span<const int4> s0s, cuda::std::span<const int4> cws,
    cuda::std::span<const In> tr, cuda::std::span<int4> ys,
    cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::DpfGpuKernelDescriptor<in_bits, GroupTag, Prg, In>;
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
  if (cws.size() < required || tr.size() < key_count ||
      ys.size() < key_count * kOutputsPerKey ||
      s0s.data() == nullptr || cws.data() == nullptr || tr.data() == nullptr ||
      ys.data() == nullptr) return cudaErrorInvalidValue;
  if (key_count > static_cast<size_t>(std::numeric_limits<int>::max()) / LaunchShape::kBlocksPerKey)
    return cudaErrorInvalidConfiguration;
  gpu::detail::DpfSplitEvalAllBatchKernel<Descriptor, LaunchShape::kFrontierDepth, LaunchShape::kBlockRootDepth>
      <<<key_count * LaunchShape::kBlocksPerKey, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0s.data(), cws.data(), tr.data(), resolved_stride, ys.data(), prg_);
  return cudaGetLastError();
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
