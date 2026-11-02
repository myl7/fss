// SPDX-License-Identifier: Apache-2.0
/**
 * @file dmpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed multi-point function (DMPF).
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret
 * Sharing_](https://eprint.iacr.org/2021/580) (@ref dmpf "1: the published version"),
 * Section 4.
 *
 * ## Definitions
 *
 * **Multi-point function**: for the input domain $\sG_{in} = \{0, 1\}^{\mathrm{in\_bits}}$, the output domain
 * $(\sG_{out}, +)$ that is a group, a set of $t$ pairs $(a_j, b_j)$ where $a_j \in \sG_{in}$ and
 * $b_j \in \sG_{out}$, a multi-point function $f$ is a function that for any input $x$, the output
 * $y$ has $y = b_j$ when $x = a_j$ for some $j$, otherwise $y = 0$.
 *
 * - Key generation: $Gen(1^\lambda, \{(a_j, b_j)\}) \rightarrow (k_0, k_1)$.
 * - Batch evaluation: $BatchEval(k_i, \{x\}) \rightarrow \{y_i\}$.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's LSB to 0, corresponding to
 * $\lambda = 127$. See Groupable for more details.
 *
 * The input domain width is at most 126 bits so the three-choice PRP domain fits 128 bits.
 *
 * The inner DPF uses `__uint128_t` as its input type and `bucket_bits` as the domain bit size.
 *
 * ## References
 *
 * 1. Leo de Castro, Antigoni Polychroniadou: Lightweight, Maliciously Secure Verifiable Function
 * Secret Sharing. EUROCRYPT 2022: 150-179. <https://doi.org/10.1007/978-3-031-06944-4_6>.
 * @anchor dmpf
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <span>
#include <vector>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/hash.cuh>
#include <fss/prp.cuh>
#include <fss/util.cuh>
#include <fss/dpf.cuh>
#include <fss/cuckoo_hash.cuh>
#include <fss/constraints.cuh>
#include <limits>

namespace fss {

namespace detail {
class SchemeAccess;
}  // namespace detail

/**
 * 2-party DMPF scheme.
 *
 * Gen(), BatchEval(), and EvalAll() are host operations.
 * Cuckoo hashing uses exactly three candidate buckets per input.
 * The PRP permutes [0, 3*kDomainSize), where kDomainSize = 2^in_bits.
 * kBucketCount is the allocated bucket count for max_points. Gen() returns the active
 * bucket count m_rt and bucket-domain size b_size_rt for the actual point count.
 * These runtime values, the PRP seed, and all per-bucket components must come
 * from the same successful Gen() call.
 * The supported point-count interval is [kMinPoints, max_points]. Increasing
 * bucket_bits can lower kMinPoints at the cost of more correction storage.
 *
 * @tparam in_bits Input domain bit size, in [1, 126].
 * @tparam max_points Maximum number of point functions. Must be >= 30.
 * Sizes arrays at compile time.
 * @tparam bucket_bits Inner DPF domain width in [1, 127], with enough bucket capacity.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam Prp See Permutable. Used for Cuckoo hashing.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam ch_lambda Target Cuckoo-hashing failure exponent in bits, used by the bucket-count estimate.
 *
 * @ingroup fss_api
 */
template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In = uint,
    int ch_lambda = 80>
  requires(ValidForInputBits<In, in_bits> && 1 <= in_bits && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)
class Dmpf {
  using InnerDpf = Dpf<bucket_bits, Group, Prg, __uint128_t>;

public:
  static constexpr int kBucketCount = cuckoo_hash::ChBucket(max_points, ch_lambda);  ///< Allocated bucket count.
  static constexpr __uint128_t kDomainSize = __uint128_t(1) << in_bits;  ///< Input domain size.
  /**
   * Smallest point count supported by this bucket_bits configuration.
   * Gen() accepts every t in [kMinPoints, max_points]. The lower bound is
   * at least 30, where the Cuckoo bucket-count estimate applies. Fewer
   * points use fewer buckets and can require a larger per-bucket domain.
   * The capacity condition is ceil(3*kDomainSize/ChBucket(t, ch_lambda))
   * <= 2^bucket_bits. The shared correction buffer contains
   * kBucketCount*(bucket_bits+1)*sizeof(Cw) bytes, so each extra bucket bit
   * adds kBucketCount*sizeof(Cw) bytes without changing the bucket count.
   */
  static constexpr int kMinPoints = [] {
    int lower = 30;
    int upper = max_points;
    while (lower < upper) {
      int middle = lower + (upper - lower) / 2;
      int buckets = cuckoo_hash::ChBucket(middle, ch_lambda);
      __uint128_t bucket_size = (3 * kDomainSize + buckets - 1) / buckets;
      if (bucket_size <= (__uint128_t{1} << bucket_bits)) upper = middle;
      else lower = middle + 1;
    }
    return lower;
  }();
  /**
   * Bucket domain size when the actual point count equals max_points.
   * A smaller runtime point count can give a larger bucket domain. Use the
   * b_size_rt returned by Gen() for the actual key's evaluation.
   */
  static constexpr __uint128_t kBucketSize = (kDomainSize * 3 + kBucketCount - 1) / kBucketCount;

  using Cw = typename InnerDpf::Cw;  ///< Correction word of the inner DPF.
  using BucketKey = typename InnerDpf::Key;  ///< One party's inner DPF key for a bucket.

  /**
   * DMPF key for one party.
   *
   * Stores the PRP seed, runtime parameters from Gen, and per-bucket inner DPF keys.
   * Both parties share each bucket's correction words and differ in the seed.
   * Fields support direct access. Field order, padding, and binary serialization are not specified.
   *
   * @ingroup fss_api
   */
  struct Key {
    int4 sigma;  ///< Shared seed for the Cuckoo-hash permutation.
    int m_rt;  ///< Runtime bucket count used during Gen.
    __uint128_t b_size_rt;  ///< Runtime bucket size used during Gen.
    BucketKey bks[kBucketCount];  ///< Inner DPF key for every allocated bucket.
  };

  /**
   * Construct a scheme with configured backends.
   *
   * @param prg Configured Prg backend.
   * @param prp Configured Prp backend.
   */
  __host__ __device__ explicit Dmpf(Prg prg, Prp prp);

  /**
   * Generate shared corrections with read-only seeds and one attempt.
   *
   * Makes one Cuckoo-hashing attempt, then generates each inner DPF.
   * Caller-owned sigma and s0s are preserved.
   *
   * @par Retrying a failed attempt
   * On return 1, draw a fresh uniformly random sigma block from a
   * cryptographically secure random generator and rerun the whole Gen with
   * the same points and payloads. Placement depends on sigma, not on the inner
   * DPF seeds, so s0s need not change for this retry. Incrementing sigma is not
   * resampling. The caller chooses the attempt limit and handles exhaustion.
   * Reuse of the output buffers is allowed. Only a successful call's outputs
   * and its sigma/s0s form a usable key.
   *
   * @param cws At least kBucketCount*(bucket_bits+1) writable corrections in bucket-major order.
   * @param m_rt Returned runtime bucket count.
   * @param b_size_rt Returned runtime bucket domain size.
   * @param sigma Uniformly random PRP seed from a cryptographically secure random generator.
   * @param s0s At least 2*kBucketCount independent uniformly random initial seeds
   * in party-major order: s0s[i] is party 0's seed for bucket i and
   * s0s[kBucketCount+i] is party 1's seed. Sample independently of sigma
   * with a cryptographically secure random generator.
   * @param as Alpha values of t point functions.
   * @param b_bufs Corresponding beta values, clamped before generation.
   * @param t Actual number of points in [kMinPoints, max_points].
   * @param max_evictions Eviction limit per inserted point. Exceeding it fails the attempt.
   * @pre as.size() >= t and b_bufs.size() >= t.
   * @pre The first t alpha values are distinct and belong to [0, kDomainSize).
   * @pre max_evictions is nonnegative.
   * @pre ceil(3*kDomainSize/cuckoo_hash::ChBucket(t, ch_lambda)) <= 2^bucket_bits.
   * @post On success, cws, sigma, and s0s define both parties' keys.
   * Correction outputs are unspecified on failure. m_rt and b_size_rt may
   * already have been written when the attempt fails; metadata alone is not a key.
   * @return 0 on success, or 1 if the Cuckoo-hashing attempt fails.
   */
  int Gen(cuda::std::span<Cw> cws, int &m_rt, __uint128_t &b_size_rt,
      int4 sigma, cuda::std::span<const int4> s0s,
      cuda::std::span<const In> as, cuda::std::span<const int4> b_bufs, int t, int max_evictions = 1000);

  /**
   * Batch evaluation method.
   *
   * Evaluates a batch of input points using caller-owned shared corrections and party seeds.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param sigma The effective PRP seed from Gen().
   * @param m_rt Runtime bucket count returned by Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only corrections in bucket-major order.
   * @param xs Input points to evaluate.
   * @param ys Output shares (pre-allocated, size >= xs.size()). Will be zero-initialized.
   */
  void BatchEval(bool b, int4 sigma, int m_rt, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const In> xs, cuda::std::span<int4> ys);

  /**
   * Evaluate the DMPF on all input points.
   *
   * @pre The full output and each inner bucket output have element counts and byte sizes that fit size_t.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param sigma The effective PRP seed from Gen().
   * @param m_rt Runtime bucket count returned by Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only corrections in bucket-major order.
   * @param ys Output shares (pre-allocated, size == kDomainSize). Will be zero-initialized.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   */
  void EvalAll(bool b, int4 sigma, int m_rt, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<int4> ys, int par_depth = 0)
    requires(in_bits < sizeof(size_t) * 8 && bucket_bits < sizeof(size_t) * 8);

  /**
   * Launch full-domain CUDA evaluation with internal temporary bucket outputs.
   *
   * The host instance supplies backend values as kernel arguments. Key/output
   * buffers and backend device resources are prepared and retained by the caller.
   * Temporary bucket outputs are allocated and released in the supplied
   * stream. No backend upload, deep copy, or synchronization is performed.
   * Prg, Prp, and Group must support the device execution and concurrency used.
   * This is one multipoint key with active_buckets inner DPF keys, not a batch
   * of independent multipoint keys.
   *
   * @tparam frontier_depth Inner DPF frontier depth, capped at bucket_bits. Defaults to 16.
   * @tparam threads_per_block Inner DPF block size, capped at the frontier width. Defaults to 256.
   * The requested block size is a power of two in [1,1024]. The inner DPF
   * derives its block-root depth, permits at most eight tail levels, and uses
   * signed-int block indices, exactly as Dpf::EvalAllGpuBatch().
   * @param b Party index, false for party 0 and true for party 1.
   * @param sigma Shared PRP seed from successful Gen().
   * @param active_buckets Runtime bucket count from that Gen(), in [1,kBucketCount].
   * @param bucket_size Runtime bucket size from that Gen(), equal to ceil(3*kDomainSize/active_buckets).
   * @param s0s Device storage with this party's seed for each active bucket.
   * @param cws Device storage of at least active_buckets*(bucket_bits+1) corrections, in bucket-major order.
   * @param ys Device output storage of at least 2^in_bits int4 blocks, in input order.
   * @param stream Stream on the device owning the referenced resources.
   * @pre this is accessible on the host. Buffers and backend resources remain valid until work completes.
   * @return The first immediate CUDA allocation, launch, or release failure, or cudaSuccess.
   * Invalid metadata, buffer lengths, null used buffers, or byte sizes return
   * cudaErrorInvalidValue. An unrepresentable merge grid returns
   * cudaErrorInvalidConfiguration. Execution completion is asynchronous.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 sigma, int active_buckets,
      __uint128_t bucket_size, cuda::std::span<const int4> s0s,
      cuda::std::span<const Cw> cws, cuda::std::span<int4> ys,
      cudaStream_t stream = 0) const
    requires(in_bits <= 38 && in_bits < sizeof(size_t) * 8 && bucket_bits < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  static constexpr int kCuckooChoices = 3;

  Prg prg_;
  Prp prp_;
};

// -----------------------------------------------------------------------------

namespace gpu::detail {

// Hold the merge group and backends behind a descriptor so wide Group
// non-type template parameters stay out of CUDA host stub signatures.
template <typename GroupTag, typename Prp, typename In>
struct DmpfMergeGpuKernelDescriptor {
  using Group = typename GroupTag::type;
  using PrpBackend = Prp;
  using InputType = In;
};

#if defined(__CUDACC__)

// Each domain input gathers the three PRP-selected inner DPF outputs.
template <int in_bits, int bucket_bits, typename Descriptor>
__global__ void DmpfMergeKernel(int4 *ys, int4 sigma, int active_buckets,
    __uint128_t bucket_size, const int4 *bucket_ys, typename Descriptor::PrpBackend prp) {
  using Group = typename Descriptor::Group;
  using In = typename Descriptor::InputType;
  using Prp = typename Descriptor::PrpBackend;
  constexpr size_t kOutputs = size_t{1} << in_bits;
  const size_t input_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (input_index >= kOutputs) return;
  In x = static_cast<In>(input_index);
  constexpr __uint128_t kInputDomain = __uint128_t{1} << in_bits;
  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp};
  int4 zero = {0, 0, 0, 0};
  auto acc = Group::From(zero);
  for (int k = 0; k < 3; ++k) {
    auto [bucket, j] = prp_hash.Locate(sigma, x, k, kInputDomain, bucket_size);
    if (bucket >= active_buckets) continue;
    acc = acc + Group::From(bucket_ys[static_cast<size_t>(bucket) * (size_t{1} << bucket_bits) + static_cast<size_t>(j)]);
  }
  ys[input_index] = acc.Into();
}

#endif

}  // namespace gpu::detail

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In,
    int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && 1 <= in_bits && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)
inline __host__ __device__ Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, ch_lambda>::Dmpf(Prg prg, Prp prp)
    : prg_(prg), prp_(prp) {}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In,
    int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && 1 <= in_bits && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)
inline int Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, ch_lambda>::Gen(
    cuda::std::span<Cw> cws, int &m_rt, __uint128_t &b_size_rt,
    int4 sigma, cuda::std::span<const int4> s0s,
    cuda::std::span<const In> as, cuda::std::span<const int4> b_bufs, int t, int max_evictions) {
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  assert(s0s.size() >= size_t{kBucketCount} * 2);
  assert(t <= max_points);
  assert(t >= kMinPoints);

  // Compute runtime bucket count and bucket size.
  int m_ = cuckoo_hash::ChBucket(t, ch_lambda);
  assert(m_ <= kBucketCount);
  __uint128_t n128 = static_cast<__uint128_t>(kDomainSize);
  __uint128_t bucket_size = (n128 * kCuckooChoices + m_ - 1) / m_;
  assert(bucket_size <= (__uint128_t{1} << bucket_bits));
  __uint128_t b_rt = bucket_size;

  m_rt = m_;
  b_size_rt = b_rt;

  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};
  cuckoo_hash::detail::Compact<Prp, In> compact{prp_};

  // Run Cuckoo hashing.
  std::vector<std::pair<int, int>> table(m_, {-1, -1});
  int ret = compact.Run(std::span<const In>(as.data(), t), m_, sigma, kDomainSize, b_rt, max_evictions,
      std::span<std::pair<int, int>>(table));

  // Generate inner DPF keys for each bucket.
  if (ret == 0) {
    InnerDpf inner_dpf{prg_};

    for (int i = 0; i < kBucketCount; ++i) {
      __uint128_t a_prime = 0;
      int4 b_buf_prime = {0, 0, 0, 0};

      if (i < m_ && table[i].first != -1) {
        int j = table[i].first;   // index into as
        int k = table[i].second;  // hash function that placed it
        auto [bucket, index] = prp_hash.Locate(sigma, as[j], k, kDomainSize, b_rt);
        a_prime = index;
        assert(a_prime < (__uint128_t{1} << bucket_bits));
        b_buf_prime = b_bufs[j];
      }

      const int4 seeds[2] = {s0s[i], s0s[size_t{kBucketCount} + i]};
      inner_dpf.Gen(cws.data() + static_cast<size_t>(i) * (bucket_bits + 1),
          seeds, a_prime, b_buf_prime);
    }

    return 0;
  }

  return 1;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In,
    int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && 1 <= in_bits && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)
inline void Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, ch_lambda>::BatchEval(
    bool b, int4 sigma, int m_rt, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const In> xs, cuda::std::span<int4> ys) {
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  size_t eta = xs.size();
  assert(ys.size() >= eta);

  __uint128_t b_rt = b_size_rt;

  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};

  // Build per-bucket input lists.
  // inputs[i] = vector of (within_bucket_index, original_input_index).
  std::vector<std::vector<std::pair<__uint128_t, size_t>>> inputs(kBucketCount);

  for (size_t omega = 0; omega < eta; ++omega) {
    // store for easier deduplication for this omega
    std::array<std::pair<int, __uint128_t>, kCuckooChoices> seen{};
    int num_seen = 0;

    for (int k = 0; k < kCuckooChoices; ++k) {
      auto [bucket, index] = prp_hash.Locate(sigma, xs[omega], k, kDomainSize, b_rt);
      if (bucket >= kBucketCount) continue;

      __uint128_t j = index;
      assert(j < (__uint128_t{1} << bucket_bits));

      // Deduplicate among this omega's located records only. Permu is a permutation over
      // [0, kDomainSize * kCuckooChoices), so distinct k never collide and the check is only a safeguard.
      bool dup = false;
      for (int s = 0; s < num_seen; ++s) {
        if (seen[s].first == bucket && seen[s].second == j) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        inputs[bucket].push_back({j, omega});
        seen[num_seen++] = {bucket, j};
      }
    }
  }

  // Initialize outputs.
  for (size_t i = 0; i < eta; ++i) {
    ys[i] = {0, 0, 0, 0};
  }

  // Evaluate per bucket.
  InnerDpf inner_dpf{prg_};
  for (int i = 0; i < kBucketCount; ++i) {
    for (auto &[j, omega] : inputs[i]) {
      int4 y = inner_dpf.Eval(b, s0s[i], cws.data() + static_cast<size_t>(i) * (bucket_bits + 1), j);

      // Accumulate output.
      ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();
    }
  }
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In,
    int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && 1 <= in_bits && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)
inline void Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, ch_lambda>::EvalAll(
    bool b, int4 sigma, int m_rt, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, int par_depth)
  requires(in_bits < sizeof(size_t) * 8 && bucket_bits < sizeof(size_t) * 8) {
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  assert(par_depth >= 0);
  static_assert(detail::FitsFullDomain<int4, in_bits>(),
      "full-domain output count and byte size must fit in size_t");
  constexpr size_t bucket_size = detail::FullDomainSize<int4, bucket_bits>();
  assert(ys.size() == kDomainSize);

  int bucket_count = m_rt;
  __uint128_t b_rt = b_size_rt;

  // build per-bucket input lists.
  // inputs[i] = vector of (within_bucket_index, original_input_index).
  std::vector<std::vector<std::pair<__uint128_t, size_t>>> inputs(bucket_count);
  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};

  for (size_t omega = 0; omega < kDomainSize; ++omega) {
    In x = static_cast<In>(omega);

    // store for easier deduplication for this omega
    std::array<std::pair<int, __uint128_t>, kCuckooChoices> seen{};
    int num_seen = 0;

    for (int k = 0; k < kCuckooChoices; ++k) {
      auto [bucket, index] = prp_hash.Locate(sigma, x, k, kDomainSize, b_rt);
      if (bucket >= bucket_count) continue;

      __uint128_t j = index;
      assert(j < (__uint128_t{1} << bucket_bits));

      // Deduplicate among this omega's located records (linear scan, fine for small kCuckooChoices).
      bool dup = false;

      for (int s = 0; s < num_seen; ++s) {
        if (seen[s].first == bucket && seen[s].second == j) {
          dup = true;
          break;
        }
      }

      if (!dup) {
        inputs[bucket].push_back({j, omega});
        seen[num_seen++] = {bucket, j};
      }
    }
  }

  // initialize outputs.
  for (size_t i = 0; i < kDomainSize; ++i) {
    ys[i] = {0, 0, 0, 0};
  }

  // evaluate per bucket and merge.
  InnerDpf inner_dpf{prg_};
  std::vector<int4> bucket_ys(bucket_size, {0, 0, 0, 0});
  for (int i = 0; i < bucket_count; ++i) {
    inner_dpf.EvalAll(b, s0s[i], cws.data() + static_cast<size_t>(i) * (bucket_bits + 1), bucket_ys.data(), par_depth);

    // merge
    for (auto &[j, omega] : inputs[i]) {
      ys[omega] = (Group::From(ys[omega]) + Group::From(bucket_ys[j])).Into();
    }
  }
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In,
    int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && 1 <= in_bits && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)

template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, ch_lambda>::EvalAllGpu(
    bool b, int4 sigma, int active_buckets, __uint128_t bucket_size,
    cuda::std::span<const int4> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> ys, cudaStream_t stream) const
  requires(in_bits <= 38 && in_bits < sizeof(size_t) * 8 && bucket_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  static_assert(std::is_trivially_copyable_v<Prp>, "the Prp backend must be device-callable");
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = gpu::detail::DmpfMergeGpuKernelDescriptor<GroupTag, Prp, In>;
  constexpr size_t kOutputs = size_t{1} << in_bits;
  constexpr size_t kBucketOutputs = size_t{1} << bucket_bits;
  constexpr size_t kCorrectionsPerBucket = bucket_bits + 1;
  constexpr size_t kMaxAddressableSize = std::numeric_limits<size_t>::max();
  constexpr int kMergeThreads = 256;
  constexpr size_t kMergeBlocks = (kOutputs + kMergeThreads - 1) / kMergeThreads;

  if (active_buckets <= 0 || active_buckets > kBucketCount) return cudaErrorInvalidValue;
  const __uint128_t expected_bucket_size =
      (kCuckooChoices * kDomainSize + active_buckets - 1) / active_buckets;
  if (bucket_size != expected_bucket_size || bucket_size > (__uint128_t{1} << bucket_bits))
    return cudaErrorInvalidValue;
  const size_t bucket_count = static_cast<size_t>(active_buckets);
  if (kOutputs > kMaxAddressableSize / sizeof(int4) ||
      bucket_count > kMaxAddressableSize / sizeof(Cw) / kCorrectionsPerBucket ||
      bucket_count > kMaxAddressableSize / sizeof(int4) / kBucketOutputs ||
      s0s.size() < bucket_count || cws.size() < bucket_count * kCorrectionsPerBucket ||
      ys.size() < kOutputs || s0s.data() == nullptr || cws.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  if (kMergeBlocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;

  gpu::detail::DeviceScratch bucket_storage(stream);
  cudaError_t status = bucket_storage.Allocate<int4>(bucket_count * kBucketOutputs);
  if (status != cudaSuccess) return status;
  auto *bucket_ys = bucket_storage.Get<int4>();
  InnerDpf inner{prg_};
  status = inner.template EvalAllGpuBatch<frontier_depth, threads_per_block>(b,
      s0s.first(bucket_count), cws.first(bucket_count * kCorrectionsPerBucket),
      cuda::std::span<int4>(bucket_ys, bucket_count * kBucketOutputs), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::DmpfMergeKernel<in_bits, bucket_bits, Descriptor>
      <<<kMergeBlocks, kMergeThreads, 0, stream>>>(ys.data(), sigma, active_buckets,
          bucket_size, bucket_ys, prp_);
  return gpu::detail::ReleaseScratch(cudaGetLastError(), {&bucket_storage});
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
