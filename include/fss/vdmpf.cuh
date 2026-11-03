// SPDX-License-Identifier: Apache-2.0
/**
 * @file vdmpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party verifiable distributed multi-point function (VDMPF).
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret
 * Sharing_](https://eprint.iacr.org/2021/580) (@ref vdmpf "1: the published version"),
 * Section 4.
 *
 * ## Definitions
 *
 * **Multi-point function**: for the input domain $\sG_{in} = \{0, 1\}^{\mathrm{in\_bits}}$, the output domain
 * $(\sG_{out}, +)$ that is a group, a set of $t$ pairs $(a_j, b_j)$ where $a_j \in \sG_{in}$ and
 * $b_j \in \sG_{out}$, a multi-point function $f$ is a function that for any input $x$, the output
 * $y$ has $y = b_j$ when $x = a_j$ for some $j$, otherwise $y = 0$.
 *
 * **VDMPF**: extends DMPF with verifiability. Uses Cuckoo hashing to distribute point functions
 * across buckets, then evaluates an inner VDPF per bucket.
 *
 * - Key generation: $Gen(1^\lambda, \{(a_j, b_j)\}) \rightarrow (k_0, k_1)$.
 * - Batch evaluation: $BatchEval(k_i, \{x\}) \rightarrow (\{y_i\}, \pi_i)$.
 * - Verification: $Verify(\pi_0, \pi_1) \rightarrow \{Accept, Reject\}$.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's LSB to 0, corresponding to
 * $\lambda = 127$. See Groupable for more details.
 *
 * The input domain width is at most 126 bits so the three-choice PRP domain fits 128 bits.
 *
 * The inner VDPF uses `__uint128_t` as its input type and `bucket_bits` as the domain bit size.
 *
 * ## References
 *
 * 1. Leo de Castro, Antigoni Polychroniadou: Lightweight, Maliciously Secure Verifiable Function
 * Secret Sharing. EUROCRYPT 2022: 150-179. <https://doi.org/10.1007/978-3-031-06944-4_6>.
 * @anchor vdmpf
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
#include <fss/vdpf.cuh>
#include <fss/cuckoo_hash.cuh>
#include <fss/constraints.cuh>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#if defined(__CUDACC__)
#include <cub/device/device_radix_sort.cuh>
#endif

namespace fss {

namespace detail {
class SchemeAccess;
}  // namespace detail

/**
 * 2-party VDMPF scheme.
 *
 * Gen() and CPU evaluation are host operations.
 * Cuckoo hashing uses exactly three candidate buckets per input.
 * The PRP permutes [0, 3*kDomainSize), where kDomainSize = 2^in_bits.
 * kBucketCount is the allocated bucket count for max_points. Gen() returns the active
 * bucket count m_rt and bucket-domain size b_size_rt for the actual point count.
 * These runtime values, the PRP seed, and all per-bucket arrays must come
 * from the same successful Gen() call.
 * The supported point-count interval is [kMinPoints, max_points]. Increasing
 * bucket_bits can lower kMinPoints at the cost of more correction word storage.
 * BatchEval() uses chain accumulation and BatchEvalMerkle() uses a Merkle fold.
 * Both parties must use the same accumulator, key configuration, and query sequence.
 * EvalAll() and EvalAllMerkle() use the same accumulators for the full domain
 * in increasing input order.
 * Eval() evaluates one input per call and returns the proof material for
 * Prove() (chain) or ProveMerkle() (Merkle). The per-input methods share the
 * fused paths' bucket-level folds; with the same key and query
 * sequence they produce the same proofs, and the same shares for the
 * commutative Group backends in this repository.
 *
 * @tparam in_bits Input domain bit size, in [1, 126].
 * @tparam max_points Maximum number of point functions. Must be >= 30.
 * Sizes arrays at compile time.
 * @tparam bucket_bits Inner VDPF domain width in [1, 127], with enough bucket capacity.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam XorHash See XorHashable. Paper's $H$: maps $(x, s)$ to $4\lambda$ bits.
 * @tparam Hash See Hashable. Paper's $H'$: maps $4\lambda$ bits to $2\lambda$ bits.
 * @tparam Prp See Permutable. Used for Cuckoo hashing.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam ch_lambda Target Cuckoo-hashing failure exponent in bits, used by the bucket-count estimate.
 *
 * @ingroup fss_api
 */
template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In = uint, int ch_lambda = 80>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
class Vdmpf {
  using InnerVdpf = Vdpf<bucket_bits, Group, Prg, XorHash, Hash, __uint128_t>;

  // Private: the fixed candidate count never leaves this class; the bound
  // checks in cuckoo_hash::detail are the only place that spells it out.
  // Unlike the member function bodies, the public EvalProof alias below is not
  // a complete-class context, so the count is declared before that first use
  // instead of at the bottom.
  static constexpr int kCuckooChoices = 3;  ///< Candidate count per input, fixed by the Cuckoo-hashing construction.

public:
  static constexpr int kBucketCount = cuckoo_hash::ChBucket(max_points, ch_lambda);  ///< Allocated bucket count.
  static constexpr __uint128_t kDomainSize = __uint128_t(1) << in_bits;  ///< Input domain size.
  /**
   * Smallest point count supported by this bucket_bits configuration.
   * Gen() accepts every t in [kMinPoints, max_points]. The lower bound is
   * at least 30, where the Cuckoo bucket-count estimate applies. Fewer
   * points use fewer buckets and can require a larger per-bucket domain.
   * The capacity condition is ceil(3*kDomainSize/ChBucket(t, ch_lambda))
   * <= 2^bucket_bits. The shared buffer of correction words contains
   * kBucketCount*(bucket_bits+1)*sizeof(Cw) bytes, so each extra bucket bit
   * adds kBucketCount*sizeof(Cw) bytes without changing the bucket count.
   */
  static constexpr int kMinPoints = [] {
    int lower = 30;
    int upper = max_points;
    while (lower < upper) {
      int middle = lower + (upper - lower) / 2;
      int buckets = cuckoo_hash::ChBucket(middle, ch_lambda);
      __uint128_t bucket_size = cuckoo_hash::detail::MinBucketSize(kDomainSize, kCuckooChoices, buckets);
      if (bucket_size <= (__uint128_t{1} << bucket_bits)) upper = middle;
      else lower = middle + 1;
    }
    return lower;
  }();
  /**
   * Bucket domain size when the actual point count equals max_points: the
   * smallest bucket domain size that keeps every located bucket below
   * kBucketCount, as returned by cuckoo_hash::detail::MinBucketSize(). A
   * smaller runtime point count can give a larger bucket domain. Use the
   * b_size_rt returned by Gen() for the actual key's evaluation.
   */
  static constexpr __uint128_t kBucketSize =
      cuckoo_hash::detail::MinBucketSize(kDomainSize, kCuckooChoices, kBucketCount);

  using Cw = typename InnerVdpf::Cw;  ///< Correction word of the inner VDPF.

  /**
   * Per-bucket VDPF key with shared correction words and one party's initial seed.
   * Fields support direct access. Field order, padding, and binary serialization are not specified.
   * @ingroup fss_api
   */
  struct BucketKey {
    Cw cws[bucket_bits + 1];  ///< Tree correction words and the bucket's final output correction word.
    int4 cs[4];  ///< Shared check seed for verifiable evaluation.
    int4 s0;  ///< This party's initial seed.
  };

  /**
   * VDMPF key for one party, containing the PRP seed and per-bucket keys.
   * Fields support direct access. Field order, padding, and binary serialization are not specified.
   * @ingroup fss_api
   */
  struct Key {
    int4 sigma;  ///< Shared seed for the Cuckoo-hash permutation.
    int m_rt;  ///< Runtime bucket count used during Gen.
    __uint128_t b_size_rt;  ///< Runtime bucket size used during Gen.
    BucketKey bks[kBucketCount];  ///< Inner VDPF key for every allocated bucket.
  };

  /**
   * Construct a scheme with configured backends.
   *
   * @param prg Configured Prg backend.
   * @param xor_hash Configured XorHash backend.
   * @param hash Configured Hash backend.
   * @param prp Configured Prp backend.
   */
  __host__ __device__ explicit Vdmpf(Prg prg, XorHash xor_hash, Hash hash, Prp prp);

  /**
   * Generate shared correction words with read-only seeds and one attempt.
   *
   * Makes one Cuckoo-hashing attempt and one attempt for each inner VDPF.
   * Caller-owned sigma and s0s are preserved.
   *
   * @par Retrying a failed attempt
   * On return 1, draw a fresh uniformly random sigma block from a
   * cryptographically secure random generator. Keep s0s unchanged because
   * placement does not depend on the inner VDPF seeds.
   * On return 2, keep sigma and draw fresh independent uniformly random seed
   * pairs for all kBucketCount buckets from a cryptographically secure random
   * generator. The bucket containing the failed inner VDPF is not reported.
   * Both inner VDPF failure conditions use this same resampling procedure.
   * In either case, rerun the whole Gen with the same points and payloads.
   * Incrementing seed words is not resampling. The caller chooses the attempt
   * limit and handles exhaustion. Reuse of the output buffers is allowed, but
   * discard all cws and cs from the failed attempt, including completed buckets.
   * Only outputs and input seeds from a successful call form a usable key.
   *
   * @param cws At least kBucketCount*(bucket_bits+1) writable correction words in bucket-major order, including each bucket's final output correction word.
   * @param cs At least 4*kBucketCount writable check-seed blocks in bucket-major order.
   * @param m_rt Returned runtime bucket count.
   * @param b_size_rt Returned runtime bucket domain size.
   * @param sigma Uniformly random PRP seed from a cryptographically secure random generator.
   * @param s0s At least 2*kBucketCount independent uniformly random initial seeds
   * in party-major order: s0s[i] is party 0's seed for bucket i and
   * s0s[kBucketCount+i] is party 1's seed. Sample independently of sigma
   * with a cryptographically secure random generator, including empty and inactive buckets.
   * @param as Alpha values of t point functions.
   * @param b_bufs Corresponding beta values. Will be clamped.
   * @pre kMinPoints <= t <= max_points, as.size() >= t, and b_bufs.size() >= t.
   * @pre The first t alpha values are distinct, belong to [0, kDomainSize), and the runtime bucket size fits bucket_bits.
   * @pre max_evictions is nonnegative.
   * @param t Actual number of points in [kMinPoints, max_points].
   * @pre ceil(3 * kDomainSize / cuckoo_hash::ChBucket(t, ch_lambda)) <= 2^bucket_bits.
   * @param max_evictions Eviction limit per inserted point. Exceeding it fails the attempt.
   * @post On success, cws, cs, sigma, and s0s define both parties' keys.
   * Correction-word outputs are unspecified on failure. m_rt and b_size_rt may
   * already have been written when the attempt fails; metadata alone is not a key.
   * @return 0 on success, 1 if Cuckoo placement fails, or 2 if an inner VDPF Gen fails.
   */
  int Gen(cuda::std::span<Cw> cws, cuda::std::span<int4> cs,
      int &m_rt, __uint128_t &b_size_rt,
      int4 sigma, cuda::std::span<const int4> s0s, cuda::std::span<const In> as,
      cuda::std::span<const int4> b_bufs, int t, int max_evictions = 1000);

  /**
   * Batch verifiable evaluation method.
   *
   * Evaluates caller-owned correction words and party seeds on a batch of inputs, producing shares and a chain proof.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param sigma The effective PRP seed from Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only correction words in bucket-major order, including each bucket's final output correction word.
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param xs Input points to evaluate.
   * @param ys Output shares (pre-allocated, size >= xs.size()). Will be zero-initialized.
   * @param pi Borrowed output storage for four proof blocks.
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  void BatchEval(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs,
      cuda::std::span<const In> xs, cuda::std::span<int4> ys,
      cuda::std::span<int4, 4> pi);

  /**
   * Per-candidate corrected per-point hashes of one input, in candidate order.
   * Entry k belongs to candidate k of the input; it stays zero when that
   * candidate locates outside the allocated buckets or repeats an earlier
   * candidate. Prove() and ProveMerkle() re-locate the candidates from sigma,
   * b_size_rt, and the input.
   * @ingroup fss_api
   */
  using EvalProof = cuda::std::array<cuda::std::array<int4, 4>, kCuckooChoices>;

  /**
   * Evaluate one input and return the corrected per-point hashes for Prove() or ProveMerkle().
   *
   * The per-input counterpart of the fused evaluation: with the same key and the
   * same query sequence, Eval() per input followed by Prove() produces the same
   * chain proof as one BatchEval() call, and by ProveMerkle() the same Merkle
   * proof as one BatchEvalMerkle() call. The proofs do not depend on the group
   * order. Shares match the fused paths for the commutative Group backends in
   * this repository: Eval() adds one input's candidate shares in candidate
   * order, while the fused paths add them in bucket order.
   *
   * Routing covers every bucket below the compile-time kBucketCount. Under a
   * key from the same successful Gen(), b_size_rt = ceil(3*kDomainSize/m_rt)
   * keeps every candidate below the bound ceil(3*kDomainSize/b_size_rt), which
   * is at most m_rt, so an inactive bucket cannot receive a candidate.
   * Inactive buckets still take part in the accumulators through their check
   * seeds: Prove() folds each bucket's cs and ProveMerkle() folds each
   * bucket's hashed cs.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param sigma The effective PRP seed from Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only correction words in bucket-major order, including each bucket's final output correction word.
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param x Input point to evaluate.
   * @param y Output share for this input.
   * @return Corrected per-point hashes in candidate order, as described by EvalProof.
   * @pre All key arrays and metadata come from the same successful Gen().
   * Device code traps instead; the trap aborts the CUDA context, so any
   * CUDA work later in the process is affected, which is why no test traps
   * on the device side.
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  __host__ __device__ EvalProof Eval(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs,
      In x, int4 &y);

  /**
   * Proof accumulation method for the per-input evaluation.
   *
   * The per-input counterpart of BatchEval(): accumulates the corrected per-point
   * hashes of Eval() into the chain proof with the same per-bucket and
   * cross-bucket folds as the fused path. With the same key and the same query
   * sequence, Eval() per input followed by Prove() gives the same proof as one
   * BatchEval() call, and the same shares for commutative group addition.
   *
   * Bucket ids and candidate indices are re-located from sigma, b_size_rt, and
   * xs, so proof_values stores corrected hashes only. Both parties must supply
   * the same xs order and the same accumulator. Chain and Merkle accumulators
   * must not be mixed: Prove() and ProveMerkle() produce different proofs from
   * the same values.
   *
   * @param sigma The effective PRP seed from Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param xs Input points evaluated by Eval() with this party's key, in the
   * order of proof_values.
   * @param proof_values proof_values[i] is the EvalProof returned by Eval() for
   * xs[i]. Entries for candidates outside the allocated buckets are not read.
   * @param pi Borrowed output storage for four proof blocks.
   * @pre All key arrays and metadata come from the same successful Gen().
   * @pre xs is nonempty and proof_values.size() >= xs.size().
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  void Prove(int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4> cs, cuda::std::span<const In> xs,
      cuda::std::span<const EvalProof> proof_values, cuda::std::span<int4, 4> pi);

  /**
   * Evaluate the full domain on the host and produce a chain proof.
   *
   * Outputs and proof match BatchEval() with inputs 0 through
   * kDomainSize - 1 in increasing order. No input array is required.
   * @param b Party index, false for party 0 and true for party 1.
   * @param sigma Shared PRP seed from successful Gen().
   * @param b_size_rt Runtime bucket domain size returned by that Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only correction words in bucket-major order.
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param ys Caller-owned output storage with at least 2^in_bits int4 entries.
   * @param pi Caller-owned output storage for four chain-proof blocks.
   * @pre All key arrays and metadata come from the same successful Gen().
   * @pre The full output and each temporary array's count and byte size fit size_t.
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  void EvalAll(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs, cuda::std::span<int4> ys,
      cuda::std::span<int4, 4> pi);

  /**
   * Verification method.
   *
   * @param pi0 Party 0 proof.
   * @param pi1 Party 1 proof using the same accumulator and queries.
   * @return True if proofs match (Accept), false otherwise (Reject).
   */
  static bool Verify(cuda::std::span<const int4, 4> pi0, cuda::std::span<const int4, 4> pi1);

  /**
   * Batch evaluation method with the Merkle-fold proof.
   *
   * Parallel-friendly alternative to BatchEval(): same output shares, and the
   * proof replaces the paper's sequential chains with Merkle-style folds. Per
   * bucket, the leaf nodes (hashes of the corrected per-point hashes, ordered
   * by query position, then hash-function index) form one canonical Merkle
   * tree. Adjacent nodes combine level by level, carrying an odd trailing node
   * unchanged. The bucket node hashes the concatenation of its hashed cs and
   * its tree root. An empty bucket contributes only its hashed cs.
   * The kBucketCount bucket nodes fold the same way into the final proof,
   * whose last two words stay zero. Execution block sizes and GPU work
   * partitioning do not change this tree or the proof. Every
   * combining step hashes 64B to 32B, so soundness reduces to the pairwise
   * collision resistance of the fold hash exactly like the VDPF Merkle fold;
   * see doc/vdpf_merkle_fold_security.md.
   *
   * The proof keeps the 64B shape, so Verify() is shared, but chain and fold
   * accumulators produce different proofs: both parties must pick the same
   * one. This host path is the sequential reference; the GPU path folds the
   * same construction and matches it bit-exactly.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param sigma The effective PRP seed from Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only correction words in bucket-major order, including each bucket's final output correction word.
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param xs Input points to evaluate.
   * @param ys Output shares (pre-allocated, size >= xs.size()). Will be zero-initialized.
   * @param pi Borrowed output storage for four proof blocks.
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  void BatchEvalMerkle(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs,
      cuda::std::span<const In> xs, cuda::std::span<int4> ys,
      cuda::std::span<int4, 4> pi);

  /**
   * Proof accumulation method for the per-input evaluation with the Merkle-fold proof.
   *
   * The per-input counterpart of BatchEvalMerkle(): folds the corrected per-point
   * hashes of Eval() with the same per-bucket and cross-bucket Merkle folds as
   * the fused path, in the order of the query sequence. With the same key and
   * the same query sequence, Eval() per input followed by ProveMerkle() gives
   * the same proof as one BatchEvalMerkle() call, and the same shares for
   * commutative group addition.
   *
   * Bucket ids and candidate indices are re-located from sigma, b_size_rt, and
   * xs, so proof_values stores corrected hashes only. Both parties must supply
   * the same xs order and the same accumulator. Chain and Merkle accumulators
   * must not be mixed: Prove() and ProveMerkle() produce different proofs from
   * the same values.
   *
   * @param sigma The effective PRP seed from Gen().
   * @param b_size_rt Runtime bucket domain size returned by Gen().
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param xs Input points evaluated by Eval() with this party's key, in the
   * order of proof_values.
   * @param proof_values proof_values[i] is the EvalProof returned by Eval() for
   * xs[i]. Entries for candidates outside the allocated buckets are not read.
   * @param pi Borrowed output storage for four proof blocks.
   * @pre All key arrays and metadata come from the same successful Gen().
   * @pre xs is nonempty and proof_values.size() >= xs.size().
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  void ProveMerkle(int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4> cs, cuda::std::span<const In> xs,
      cuda::std::span<const EvalProof> proof_values, cuda::std::span<int4, 4> pi);

  /**
   * Evaluate the full domain on the host and produce a Merkle proof.
   *
   * Outputs and proof match BatchEvalMerkle() with inputs 0 through
   * kDomainSize - 1 in increasing order. No input array is required.
   * @param b Party index, false for party 0 and true for party 1.
   * @param sigma Shared PRP seed from successful Gen().
   * @param b_size_rt Runtime bucket domain size returned by that Gen().
   * @param s0s This party's kBucketCount initial seeds in bucket order.
   * @param cws At least kBucketCount*(bucket_bits+1) read-only correction words in bucket-major order.
   * @param cs At least 4*kBucketCount read-only check-seed blocks in bucket-major order.
   * @param ys Caller-owned output storage with at least 2^in_bits int4 entries.
   * @param pi Caller-owned output storage for four Merkle-proof blocks.
   * @pre All key arrays and metadata come from the same successful Gen().
   * @pre The full output and each temporary array's count and byte size fit size_t.
   * @throws std::invalid_argument if b_size_rt is outside [kBucketSize, 2^bucket_bits].
   */
  void EvalAllMerkle(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs, cuda::std::span<int4> ys,
      cuda::std::span<int4, 4> pi);

  /**
   * Launch full-domain Merkle evaluation with internal temporary storage.
   *
   * The outputs and proof match BatchEvalMerkle() on inputs 0 through
   * 2^in_bits-1 in that order. No input-index array is required.
   * The host instance passes backend values as kernel arguments. Referenced
   * backend resources and device buffers are prepared and retained by the caller.
   * Temporary work arrays and CUB sorting storage are allocated and released
   * internally using CUDA stream-ordered allocation on stream. The CUB size
   * query runs before allocation. No backend upload, deep copy, event
   * management, or host/device synchronization is performed.
   *
   * @tparam threads_per_block Power-of-two block size in [1,1024], default 256.
   * @param b Party index, false for party 0 and true for party 1.
   * @param sigma Shared PRP seed from successful Gen().
   * @param b_size_rt Runtime bucket size from that Gen(). Every located
   * bucket then stays below kBucketCount exactly when b_size_rt is at least
   * ceil(3*kDomainSize/kBucketCount).
   * @param s0s Device storage of kBucketCount party seeds, one per bucket.
   * @param cws Device storage of kBucketCount*(bucket_bits+1) bucket-major correction words.
   * @param cs Device storage of 4*kBucketCount bucket-major check-seed blocks.
   * @param ys Device output storage of at least 2^in_bits int4 blocks.
   * @param pi Device output storage for four Merkle-proof blocks.
   * @param stream Stream on the device owning the referenced resources.
   * @pre this is host accessible. Buffers and referenced backend resources
   * remain valid until queued work completes. CUDA stream-ordered allocation
   * must be supported on the current device.
   * @return The first immediate CUDA allocation, launch, or release error.
   * Invalid sizes, metadata, or pointers return
   * cudaErrorInvalidValue. CUB's current int record count requires
   * 3*2^in_bits <= INT_MAX. Unrepresentable grids return cudaErrorInvalidConfiguration.
   * Completion remains asynchronous. Chain evaluation has no GPU counterpart.
   */
  template <int threads_per_block = 256>
  __host__ cudaError_t EvalAllMerkleGpu(bool b, int4 sigma,
      __uint128_t b_size_rt, cuda::std::span<const int4> s0s,
      cuda::std::span<const Cw> cws, cuda::std::span<const int4> cs,
      cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, cudaStream_t stream = 0) const
    requires(in_bits < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  // Throws if b_size_rt is not a bucket size that Gen() can return.
  static void RequireBucketSize(__uint128_t b_size_rt, const char *entry) {
    if (!cuckoo_hash::detail::ValidBucketSize(b_size_rt, kBucketSize, bucket_bits))
      throw std::invalid_argument(std::string(entry) + ": b_size_rt is outside [kBucketSize, 2^bucket_bits]");
  }

  Prg prg_;
  XorHash xor_hash_;
  Hash hash_;
  Prp prp_;

  struct FullDomainInputs {
    size_t count;
    size_t size() const { return count; }
    In operator[](size_t index) const { return static_cast<In>(index); }
  };

  // One candidate of an input that lies inside the allocated buckets and is not
  // a repeated location of an earlier candidate of the same input, as located by
  // LocateCandidates().
  struct LocatedCandidate {
    int bucket;  ///< Bucket that the candidate located.
    __uint128_t index;  ///< Within-bucket input.
    int candidate;  ///< Candidate in [0, kCuckooChoices).
  };

  // One located record of the split path's query sequence: the query index and
  // the candidate that located it. The proof values carry the corrected hashes,
  // so the within-bucket input is not needed here. Records of one bucket stay in
  // the canonical (query, candidate) order, as in the fused paths.
  struct RouteRecord {
    size_t omega;  ///< Query index in xs.
    int candidate;  ///< Candidate in [0, kCuckooChoices).
  };

  // Locates one input's candidates in candidate order, skipping buckets outside
  // the allocation and repeated locations of an earlier candidate. Returns the
  // number of entries written to candidates.
  __host__ __device__ static int LocateCandidates(cuckoo_hash::detail::PrpHash<Prp, In> &prp_hash,
      int4 sigma, __uint128_t b_size_rt, In x,
      cuda::std::array<LocatedCandidate, kCuckooChoices> &candidates);

  // Locates every input of the split path and groups the records by bucket.
  // Shared by Prove() and ProveMerkle() so that both accumulators consume the
  // same routing order.
  std::vector<std::vector<RouteRecord>> RouteInputs(int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const In> xs);

  // Folds one corrected per-point hash into a bucket's chain accumulator, the
  // same rule as Vdpf::Prove. Shared by EvalInputs() and Prove().
  void FoldChainNode(cuda::std::array<int4, 4> &pi_bucket, cuda::std::span<const int4, 4> pi_tilde);

  // Folds one bucket's chain accumulator into the cross-bucket proof. Shared by
  // EvalInputs() and Prove().
  void FoldChainBucket(cuda::std::span<int4, 4> pi, cuda::std::span<const int4, 4> pi_bucket);

  // Folds one bucket's leaf hashes into the bucket node: the bucket's hashed cs,
  // bound to the tree root of the leaves when the bucket is not empty. Shared by
  // EvalMerkleInputs() and ProveMerkle(); both hash the bucket's cs before the
  // bucket's leaves, as the fused path did before this helper existed.
  cuda::std::array<int4, 2> BucketMerkleNode(cuda::std::array<int4, 2> bucket_node,
      std::vector<cuda::std::array<int4, 2>> &leaves);

  // Folds the bucket nodes into the four-block Merkle proof. Shared by
  // EvalMerkleInputs() and ProveMerkle().
  void FoldMerkleBuckets(std::vector<cuda::std::array<int4, 2>> &bucket_nodes, cuda::std::span<int4, 4> pi);

  template <bool merkle>
  static size_t FullInputCount(__uint128_t bucket_size);

  template <typename Inputs>
  void EvalInputs(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs,
      Inputs xs, cuda::std::span<int4> ys,
      cuda::std::span<int4, 4> pi);

  template <typename Inputs>
  void EvalMerkleInputs(bool b, int4 sigma, __uint128_t b_size_rt,
      cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4> cs,
      Inputs xs, cuda::std::span<int4> ys,
      cuda::std::span<int4, 4> pi);

  // Level-by-level pairwise fold with an odd trailing node carrying to the
  // next level unchanged, as in Vdpf::ProveMerkle.
  cuda::std::array<int4, 2> FoldNodes(std::vector<cuda::std::array<int4, 2>> &nodes);
};

// -----------------------------------------------------------------------------

#if defined(__CUDACC__)

namespace gpu::detail {

using VdmpfFullNode = cuda::std::array<int4, 2>;

struct VdmpfFullStorage {
  size_t keys;
  size_t sorted_keys;
  size_t nodes;
  size_t sorted_nodes;
  size_t counts;
  size_t offsets;
  size_t bucket_nodes;
  size_t bucket_scratch;
  size_t cub;
  size_t bytes;
};

inline bool VdmpfAppendStorage(size_t count, size_t width, size_t alignment,
    size_t &bytes, size_t &offset) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  const size_t padding = (alignment - bytes % alignment) % alignment;
  if (bytes > kMax - padding) return false;
  offset = bytes + padding;
  if (count > (kMax - offset) / width) return false;
  bytes = offset + count * width;
  return true;
}

inline bool VdmpfPlanFullStorage(size_t records, size_t buckets, size_t cub_bytes,
    VdmpfFullStorage &storage) {
  size_t bytes = 0;
  if (buckets == std::numeric_limits<size_t>::max()) return false;
  if (!VdmpfAppendStorage(records, sizeof(uint32_t), alignof(uint32_t), bytes, storage.keys) ||
      !VdmpfAppendStorage(records, sizeof(uint32_t), alignof(uint32_t), bytes, storage.sorted_keys) ||
      !VdmpfAppendStorage(records, sizeof(VdmpfFullNode), alignof(VdmpfFullNode), bytes, storage.nodes) ||
      !VdmpfAppendStorage(records, sizeof(VdmpfFullNode), alignof(VdmpfFullNode), bytes, storage.sorted_nodes) ||
      !VdmpfAppendStorage(buckets, sizeof(uint32_t), alignof(uint32_t), bytes, storage.counts) ||
      !VdmpfAppendStorage(buckets + 1, sizeof(uint32_t), alignof(uint32_t), bytes, storage.offsets) ||
      !VdmpfAppendStorage(buckets, sizeof(VdmpfFullNode), alignof(VdmpfFullNode), bytes, storage.bucket_nodes) ||
      !VdmpfAppendStorage(buckets, sizeof(VdmpfFullNode), alignof(VdmpfFullNode), bytes, storage.bucket_scratch) ||
      !VdmpfAppendStorage(cub_bytes, 1, 1, bytes, storage.cub)) return false;
  storage.bytes = bytes;
  return true;
}

template <typename T>
inline T *VdmpfFullStorageAt(void *storage, size_t offset) {
  auto *bytes = static_cast<unsigned char *>(storage);
  return static_cast<T *>(static_cast<void *>(bytes + offset));
}

inline cudaError_t VdmpfFullLaunchBlocks(size_t count, int threads, int &blocks) {
  const size_t result = count / threads + (count % threads != 0);
  if (result > static_cast<size_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  blocks = static_cast<int>(result);
  return cudaSuccess;
}

// The VDPF descriptor keeps wide Group non-type template parameters out of
// CUDA host stubs. The correction words stay opaque on the host and recover their type
// in the device body.
template <int in_bits, int bucket_bits, typename Descriptor, typename In, typename Prp>
__global__ void VdmpfFullLeavesKernel(bool b, int4 sigma, __uint128_t bucket_size,
    int choices, const int4 *s0s, const void *cws_bytes, const int4 *cs, int4 *ys,
    uint32_t *buckets, VdmpfFullNode *nodes, uint32_t *counts, size_t count,
    typename Descriptor::PrgBackend prg, typename Descriptor::XorHashBackend xor_hash,
    typename Descriptor::HashBackend hash, Prp prp) {
  using Group = typename Descriptor::Group;
  using Inner = typename Descriptor::Scheme;
  const auto *cws = static_cast<const typename Descriptor::Cw *>(cws_bytes);
  size_t omega = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (omega >= count) return;
  Inner inner{prg, xor_hash, hash};
  cuckoo_hash::detail::PrpHash<Prp, In> hasher{prp};
  auto sum = Group::From(int4{0, 0, 0, 0});
  for (int k = 0; k < choices; ++k) {
    auto location = hasher.Locate(sigma, static_cast<In>(omega), k,
        __uint128_t{1} << in_bits, bucket_size, choices);
    size_t bucket = static_cast<size_t>(location.bucket);
    int4 value;
    auto corrected = inner.Eval(b, s0s[bucket],
        cuda::std::span<const typename Descriptor::Cw>(cws + bucket * (bucket_bits + 1), bucket_bits + 1),
        cuda::std::span<const int4, 4>(cs + bucket * 4, 4), location.index, value);
    // Stable sorting by bucket preserves this outer-input, candidate order.
    size_t record = static_cast<size_t>(choices) * omega + k;
    buckets[record] = static_cast<uint32_t>(bucket);
    nodes[record] = hash.Hash(cuda::std::span<const int4, 4>(corrected));
    atomicAdd(counts + bucket, uint32_t{1});
    sum = sum + Group::From(value);
  }
  ys[omega] = sum.Into();
}

static __global__ void VdmpfFullClearCountsKernel(uint32_t *counts, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) counts[i] = 0;
}

static __global__ void VdmpfFullOffsetsKernel(const uint32_t *counts, uint32_t *offsets, size_t count) {
  uint32_t offset = 0;
  for (size_t i = 0; i < count; ++i) {
    offsets[i] = offset;
    offset += counts[i];
  }
  offsets[count] = offset;
}

template <typename Hash>
__global__ void VdmpfFullBucketPairsKernel(const VdmpfFullNode *in, VdmpfFullNode *out,
    const uint32_t *counts, const uint32_t *offsets, size_t buckets,
    uint32_t stride, size_t total, Hash hash) {
  size_t slot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total) return;
  size_t lo = 0, hi = buckets - 1;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (slot < offsets[mid + 1]) hi = mid;
    else lo = mid + 1;
  }
  size_t base = offsets[lo];
  size_t local = slot - base;
  size_t count = (counts[lo] + static_cast<size_t>(stride) - 1) / stride;
  if (local >= (count + 1) / 2) return;
  auto left = in[base + 2 * local];
  if (2 * local + 1 == count) out[slot] = left;
  else {
    auto right = in[base + 2 * local + 1];
    int4 msg[4] = {left[0], left[1], right[0], right[1]};
    out[slot] = hash.Hash(cuda::std::span<const int4, 4>(msg));
  }
}

template <typename Hash>
__global__ void VdmpfFullBindRootsKernel(const int4 *cs, const VdmpfFullNode *roots,
    const uint32_t *counts, const uint32_t *offsets, VdmpfFullNode *buckets,
    size_t count, Hash hash) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto node = hash.Hash(cuda::std::span<const int4, 4>(cs + i * 4, 4));
  if (counts[i] != 0) {
    auto root = roots[offsets[i]];
    int4 msg[4] = {node[0], node[1], root[0], root[1]};
    node = hash.Hash(cuda::std::span<const int4, 4>(msg));
  }
  buckets[i] = node;
}

template <typename Hash>
__global__ void VdmpfFullPairsKernel(const VdmpfFullNode *in, VdmpfFullNode *out,
    size_t count, Hash hash) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= (count + 1) / 2) return;
  auto left = in[2 * i];
  if (2 * i + 1 == count) out[i] = left;
  else {
    auto right = in[2 * i + 1];
    int4 msg[4] = {left[0], left[1], right[0], right[1]};
    out[i] = hash.Hash(cuda::std::span<const int4, 4>(msg));
  }
}

static __global__ void VdmpfFullFinishKernel(const VdmpfFullNode *root, int4 *pi) {
  pi[0] = root[0][0];
  pi[1] = root[0][1];
  pi[2] = {0, 0, 0, 0};
  pi[3] = {0, 0, 0, 0};
}

}  // namespace gpu::detail
#endif

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline __host__ __device__ Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::Vdmpf(Prg prg, XorHash xor_hash, Hash hash, Prp prp)
    : prg_(prg), xor_hash_(xor_hash), hash_(hash), prp_(prp) {}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline int Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::Gen(
    cuda::std::span<Cw> cws, cuda::std::span<int4> cs,
    int &m_rt, __uint128_t &b_size_rt,
    int4 sigma, cuda::std::span<const int4> s0s, cuda::std::span<const In> as,
    cuda::std::span<const int4> b_bufs, int t, int max_evictions) {
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  assert(cs.size() >= size_t{kBucketCount} * 4);
  assert(s0s.size() >= size_t{kBucketCount} * 2);
  assert(t <= max_points);

  // Compute runtime bucket count and bucket size.
  assert(t >= kMinPoints);
  int m_ = cuckoo_hash::ChBucket(t, ch_lambda);
  assert(m_ <= kBucketCount);
  __uint128_t n128 = static_cast<__uint128_t>(kDomainSize);
  __uint128_t bucket_size = cuckoo_hash::detail::CeilDiv(n128 * kCuckooChoices, m_);
  assert(bucket_size <= (__uint128_t{1} << bucket_bits));
  __uint128_t b_rt = bucket_size;

  m_rt = m_;
  b_size_rt = b_rt;

  // Run Cuckoo hashing.
  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};
  std::vector<std::pair<int, int>> table(m_, {-1, -1});
  cuckoo_hash::detail::Compact<Prp, In> compact{prp_};
  int ret = compact.Run(std::span<const In>(as.data(), t), m_, sigma, kDomainSize, b_rt, max_evictions, std::span<std::pair<int, int>>(table));
  if (ret != 0) return 1;

  // Generate inner VDPF keys for each bucket.
  InnerVdpf inner_vdpf{prg_, xor_hash_, hash_};
  for (int i = 0; i < kBucketCount; ++i) {
    __uint128_t a_prime = 0;
    int4 b_buf_prime = {0, 0, 0, 0};

    if (i < m_ && table[i].first != -1) {
      int j = table[i].first;   // index into as
      int k = table[i].second;  // hash function that placed it
      auto [bucket, index] = prp_hash.Locate(sigma, as[j], k, kDomainSize, b_rt, kCuckooChoices);
      a_prime = index;
      assert(a_prime < (__uint128_t{1} << bucket_bits));
      b_buf_prime = b_bufs[j];
    }

    const int4 seeds[2] = {s0s[i], s0s[size_t{kBucketCount} + i]};
    ret = inner_vdpf.Gen(cws.data() + static_cast<size_t>(i) * (bucket_bits + 1),
        cuda::std::span<int4, 4>(cs.data() + static_cast<size_t>(i) * 4, 4),
        cuda::std::span<const int4, 2>(seeds, 2), a_prime, b_buf_prime);
    if (ret != 0) return 2;
  }

  return 0;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::BatchEval(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs,
    cuda::std::span<const In> xs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi) {
  RequireBucketSize(b_size_rt, "Vdmpf::BatchEval");
  EvalInputs(b, sigma, b_size_rt, s0s, cws, cs, xs, ys, pi);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline __host__ __device__ typename Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In,
    ch_lambda>::EvalProof
Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::Eval(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs,
    In x, int4 &y) {
  if (!cuckoo_hash::detail::ValidBucketSize(b_size_rt, kBucketSize, bucket_bits)) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    throw std::invalid_argument("Vdmpf::Eval: b_size_rt is outside [kBucketSize, 2^bucket_bits]");
#endif
  }
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  assert(cs.size() >= size_t{kBucketCount} * 4);
  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};
  cuda::std::array<LocatedCandidate, kCuckooChoices> located;
  int count = LocateCandidates(prp_hash, sigma, b_size_rt, x, located);
  EvalProof proof{};
  int4 share = {0, 0, 0, 0};
  InnerVdpf inner_vdpf{prg_, xor_hash_, hash_};
  for (int i = 0; i < count; ++i) {
    const LocatedCandidate &candidate = located[i];
    int4 value;
    proof[candidate.candidate] = inner_vdpf.Eval(b, s0s[candidate.bucket],
        cuda::std::span<const Cw>(cws.data() + static_cast<size_t>(candidate.bucket) * (bucket_bits + 1),
            bucket_bits + 1),
        cuda::std::span<const int4, 4>(cs.data() + static_cast<size_t>(candidate.bucket) * 4, 4),
        candidate.index, value);
    share = (Group::From(share) + Group::From(value)).Into();
  }
  y = share;
  return proof;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::Prove(
    int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4> cs, cuda::std::span<const In> xs,
    cuda::std::span<const EvalProof> proof_values, cuda::std::span<int4, 4> pi) {
  RequireBucketSize(b_size_rt, "Vdmpf::Prove");
  assert(cs.size() >= size_t{kBucketCount} * 4);
  assert(!xs.empty());
  assert(proof_values.size() >= xs.size());

  std::vector<std::vector<RouteRecord>> inputs = RouteInputs(sigma, b_size_rt, xs);
  pi[0] = int4{0, 0, 0, 0};
  pi[1] = int4{0, 0, 0, 0};
  pi[2] = int4{0, 0, 0, 0};
  pi[3] = int4{0, 0, 0, 0};
  for (int i = 0; i < kBucketCount; ++i) {
    const int4 *bucket_cs = cs.data() + static_cast<size_t>(i) * 4;
    // Initialize per-bucket proof from this bucket's cs.
    cuda::std::array<int4, 4> pi_bucket = {bucket_cs[0], bucket_cs[1], bucket_cs[2], bucket_cs[3]};
    for (const RouteRecord &record : inputs[i]) {
      FoldChainNode(pi_bucket, cuda::std::span<const int4, 4>(proof_values[record.omega][record.candidate]));
    }
    FoldChainBucket(pi, cuda::std::span<const int4, 4>(pi_bucket));
  }
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::EvalAll(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi) {
  RequireBucketSize(b_size_rt, "Vdmpf::EvalAll");
  const size_t count = FullInputCount<false>(b_size_rt);
  EvalInputs(b, sigma, b_size_rt, s0s, cws, cs,
      FullDomainInputs{count}, ys, pi);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline bool Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::Verify(
    cuda::std::span<const int4, 4> pi0, cuda::std::span<const int4, 4> pi1) {
  return InnerVdpf::Verify(pi0, pi1);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::BatchEvalMerkle(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs,
    cuda::std::span<const In> xs, cuda::std::span<int4> ys,
    cuda::std::span<int4, 4> pi) {
  RequireBucketSize(b_size_rt, "Vdmpf::BatchEvalMerkle");
  EvalMerkleInputs(b, sigma, b_size_rt, s0s, cws, cs, xs, ys, pi);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::ProveMerkle(
    int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4> cs, cuda::std::span<const In> xs,
    cuda::std::span<const EvalProof> proof_values, cuda::std::span<int4, 4> pi) {
  RequireBucketSize(b_size_rt, "Vdmpf::ProveMerkle");
  assert(cs.size() >= size_t{kBucketCount} * 4);
  assert(!xs.empty());
  assert(proof_values.size() >= xs.size());

  std::vector<std::vector<RouteRecord>> inputs = RouteInputs(sigma, b_size_rt, xs);
  std::vector<cuda::std::array<int4, 2>> bucket_nodes(kBucketCount);
  std::vector<cuda::std::array<int4, 2>> leaves;
  for (int i = 0; i < kBucketCount; ++i) {
    const int4 *bucket_cs = cs.data() + static_cast<size_t>(i) * 4;
    cuda::std::array<int4, 2> bucket_node = hash_.Hash(cuda::std::span<const int4, 4>(bucket_cs, 4));
    leaves.clear();
    for (const RouteRecord &record : inputs[i]) {
      leaves.push_back(hash_.Hash(
          cuda::std::span<const int4, 4>(proof_values[record.omega][record.candidate])));
    }
    bucket_nodes[i] = BucketMerkleNode(bucket_node, leaves);
  }
  FoldMerkleBuckets(bucket_nodes, pi);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::EvalAllMerkle(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs, cuda::std::span<int4> ys,
    cuda::std::span<int4, 4> pi) {
  RequireBucketSize(b_size_rt, "Vdmpf::EvalAllMerkle");
  const size_t count = FullInputCount<true>(b_size_rt);
  EvalMerkleInputs(b, sigma, b_size_rt, s0s, cws, cs,
      FullDomainInputs{count}, ys, pi);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline __host__ __device__ int
Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::LocateCandidates(
    cuckoo_hash::detail::PrpHash<Prp, In> &prp_hash, int4 sigma, __uint128_t b_size_rt, In x,
    cuda::std::array<LocatedCandidate, kCuckooChoices> &candidates) {  int count = 0;
  for (int k = 0; k < kCuckooChoices; ++k) {
    auto [bucket, index] = prp_hash.Locate(sigma, x, k, kDomainSize, b_size_rt, kCuckooChoices);
    if (bucket >= kBucketCount) continue;
    __uint128_t j = index;
    assert(j < (__uint128_t{1} << bucket_bits));
    // Deduplicate among this input's located candidates only. Permu is a
    // permutation over [0, kDomainSize * kCuckooChoices), so distinct k never
    // collide and the check is only a safeguard, as in the fused paths.
    bool duplicate = false;
    for (int i = 0; i < count; ++i) {
      if (candidates[i].bucket == bucket && candidates[i].index == j) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    candidates[count] = {bucket, j, k};
    ++count;
  }
  return count;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline std::vector<std::vector<typename Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In,
    ch_lambda>::RouteRecord>>
Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::RouteInputs(
    int4 sigma, __uint128_t b_size_rt, cuda::std::span<const In> xs) {
  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};
  std::vector<std::vector<RouteRecord>> inputs(kBucketCount);
  for (size_t omega = 0; omega < xs.size(); ++omega) {
    cuda::std::array<LocatedCandidate, kCuckooChoices> located;
    int count = LocateCandidates(prp_hash, sigma, b_size_rt, xs[omega], located);
    for (int i = 0; i < count; ++i) {
      inputs[located[i].bucket].push_back({omega, located[i].candidate});
    }
  }
  return inputs;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::FoldChainNode(
    cuda::std::array<int4, 4> &pi_bucket, cuda::std::span<const int4, 4> pi_tilde) {
  cuda::std::array<int4, 4> h_input =
      util::Xor(cuda::std::span<const int4, 4>(pi_bucket), cuda::std::span<const int4, 4>(pi_tilde));
  auto h_out = hash_.Hash(cuda::std::span<const int4, 4>(h_input));
  pi_bucket[0] = util::Xor(pi_bucket[0], h_out[0]);
  pi_bucket[1] = util::Xor(pi_bucket[1], h_out[1]);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::FoldChainBucket(
    cuda::std::span<int4, 4> pi, cuda::std::span<const int4, 4> pi_bucket) {
  cuda::std::array<int4, 4> cross_input =
      util::Xor(cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_bucket));
  auto cross_out = hash_.Hash(cuda::std::span<const int4, 4>(cross_input));
  pi[0] = util::Xor(pi[0], cross_out[0]);
  pi[1] = util::Xor(pi[1], cross_out[1]);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
template <typename Inputs>
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::EvalInputs(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs,
    Inputs xs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi) {
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  assert(cs.size() >= size_t{kBucketCount} * 4);
  size_t eta = xs.size();
  assert(ys.size() >= eta);
  __uint128_t b_rt = b_size_rt;

  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};

  // Build per-bucket input lists.
  // inputs[i] = vector of (within_bucket_index, original_input_index).
  std::vector<std::vector<std::pair<__uint128_t, size_t>>> inputs(kBucketCount);
  for (size_t omega = 0; omega < eta; ++omega) {
    // store for easier deduplication for this omega
    cuda::std::array<std::pair<int, __uint128_t>, kCuckooChoices> seen{};
    int num_seen = 0;

    for (int k = 0; k < kCuckooChoices; ++k) {
      auto [bucket, index] = prp_hash.Locate(sigma, xs[omega], k, kDomainSize, b_rt, kCuckooChoices);
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

  // Initialize outputs and proof.
  for (size_t i = 0; i < eta; ++i) {
    ys[i] = {0, 0, 0, 0};
  }
  pi[0] = int4{0, 0, 0, 0};
  pi[1] = int4{0, 0, 0, 0};
  pi[2] = int4{0, 0, 0, 0};
  pi[3] = int4{0, 0, 0, 0};

  // Evaluate per bucket.
  InnerVdpf inner_vdpf{prg_, xor_hash_, hash_};
  for (int i = 0; i < kBucketCount; ++i) {
    const Cw *bucket_cws = cws.data() + static_cast<size_t>(i) * (bucket_bits + 1);
    const int4 *bucket_cs = cs.data() + static_cast<size_t>(i) * 4;
    // Initialize per-bucket proof from this bucket's cs.
    cuda::std::array<int4, 4> pi_bucket = {bucket_cs[0], bucket_cs[1], bucket_cs[2], bucket_cs[3]};

    for (auto &[j, omega] : inputs[i]) {
      int4 y;
      auto pi_tilde = inner_vdpf.Eval(b, s0s[i], cuda::std::span<const Cw>(bucket_cws, bucket_bits + 1),
          cuda::std::span<const int4, 4>(bucket_cs, 4), j, y);

      // Accumulate output.
      ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();

      // Accumulate per-bucket proof (same as Vdpf::Prove logic).
      FoldChainNode(pi_bucket, cuda::std::span<const int4, 4>(pi_tilde));
    }

    // Cross-bucket proof accumulation: pi = pi XOR H'(pi XOR pi_bucket).
    FoldChainBucket(pi, cuda::std::span<const int4, 4>(pi_bucket));
  }
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
template <typename Inputs>
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::EvalMerkleInputs(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4, kBucketCount> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs,
    Inputs xs, cuda::std::span<int4> ys,
    cuda::std::span<int4, 4> pi) {
  assert(cws.size() >= size_t{kBucketCount} * (bucket_bits + 1));
  assert(cs.size() >= size_t{kBucketCount} * 4);
  size_t eta = xs.size();
  assert(ys.size() >= eta);
  __uint128_t b_rt = b_size_rt;

  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp_};

  // Build per-bucket input lists in the canonical (input, hash) order: the
  // GPU path reproduces this order through a stable sort of the same
  // per-(input, hash) records.
  std::vector<std::vector<std::pair<__uint128_t, size_t>>> inputs(kBucketCount);

  for (size_t omega = 0; omega < eta; ++omega) {
    for (int k = 0; k < kCuckooChoices; ++k) {
      auto [bucket, index] = prp_hash.Locate(sigma, xs[omega], k, kDomainSize, b_rt, kCuckooChoices);
      if (bucket >= kBucketCount) continue;
      __uint128_t j = index;
      assert(j < (__uint128_t{1} << bucket_bits));
      inputs[bucket].push_back({j, omega});
    }
  }

  for (size_t i = 0; i < eta; ++i) {
    ys[i] = {0, 0, 0, 0};
  }

  InnerVdpf inner_vdpf{prg_, xor_hash_, hash_};
  std::vector<cuda::std::array<int4, 2>> bucket_nodes(kBucketCount);
  std::vector<cuda::std::array<int4, 2>> nodes;
  for (int i = 0; i < kBucketCount; ++i) {
    const Cw *bucket_cws = cws.data() + static_cast<size_t>(i) * (bucket_bits + 1);
    const int4 *bucket_cs = cs.data() + static_cast<size_t>(i) * 4;
    cuda::std::array<int4, 2> bucket_node = hash_.Hash(cuda::std::span<const int4, 4>(bucket_cs, 4));
    nodes.clear();
    nodes.reserve(inputs[i].size());
    for (auto [j, omega] : inputs[i]) {
      int4 y;
      auto pi_tilde = inner_vdpf.Eval(b, s0s[i], cuda::std::span<const Cw>(bucket_cws, bucket_bits + 1),
          cuda::std::span<const int4, 4>(bucket_cs, 4), j, y);
      ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();
      nodes.push_back(hash_.Hash(cuda::std::span<const int4, 4>(pi_tilde)));
    }
    bucket_nodes[i] = BucketMerkleNode(bucket_node, nodes);
  }

  FoldMerkleBuckets(bucket_nodes, pi);
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
template <bool merkle>
inline size_t Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::FullInputCount(
    __uint128_t bucket_size) {
  constexpr size_t count = detail::FullDomainSize<int4, in_bits>();
  using Record = std::pair<__uint128_t, size_t>;
  static_assert(detail::FitsElementCount<std::vector<Record>>(kBucketCount),
      "bucket-record array count and byte size must fit in size_t");
  const __uint128_t total_records = kDomainSize * kCuckooChoices;
  const __uint128_t max_bucket_records = bucket_size < total_records ? bucket_size : total_records;
  assert(max_bucket_records <= size_t(-1) / sizeof(Record));
  if constexpr (merkle) {
    using Node = cuda::std::array<int4, 2>;
    static_assert(detail::FitsElementCount<Node>(kBucketCount),
        "bucket proof array count and byte size must fit in size_t");
    assert(max_bucket_records <= size_t(-1) / sizeof(Node));
  }
  return count;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline cuda::std::array<int4, 2> Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::BucketMerkleNode(
    cuda::std::array<int4, 2> bucket_node, std::vector<cuda::std::array<int4, 2>> &leaves) {
  if (!leaves.empty()) {
    cuda::std::array<int4, 2> root = FoldNodes(leaves);
    int4 msg[4] = {bucket_node[0], bucket_node[1], root[0], root[1]};
    bucket_node = hash_.Hash(cuda::std::span<const int4, 4>(msg));
  }
  return bucket_node;
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline void Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::FoldMerkleBuckets(
    std::vector<cuda::std::array<int4, 2>> &bucket_nodes, cuda::std::span<int4, 4> pi) {
  cuda::std::array<int4, 2> root = FoldNodes(bucket_nodes);
  pi[0] = root[0];
  pi[1] = root[1];
  pi[2] = int4{0, 0, 0, 0};
  pi[3] = int4{0, 0, 0, 0};
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
inline cuda::std::array<int4, 2> Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::FoldNodes(
    std::vector<cuda::std::array<int4, 2>> &nodes) {
  size_t count = nodes.size();
  assert(count >= 1);
  while (count > 1) {
    size_t pairs = count / 2;
    for (size_t i = 0; i < pairs; ++i) {
      nodes[i] = detail::VdpfMerkleFoldNode(hash_, nodes.data(), count, i);
    }
    if (count % 2 != 0) nodes[pairs] = nodes[count - 1];
    count = pairs + count % 2;
  }
  return nodes[0];
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In, int ch_lambda>
  requires(ValidForInputBits<In, in_bits> && in_bits <= 126 &&
      max_points >= 30 && ch_lambda >= 0 && 1 <= bucket_bits && bucket_bits <= 127 &&
      cuckoo_hash::ChBucket(max_points, ch_lambda) > 0 &&
      (((__uint128_t{3} << in_bits) + cuckoo_hash::ChBucket(max_points, ch_lambda) - 1) /
          cuckoo_hash::ChBucket(max_points, ch_lambda)) <= (__uint128_t{1} << bucket_bits) &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
template <int threads_per_block>
__host__ inline cudaError_t Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>::EvalAllMerkleGpu(
    bool b, int4 sigma, __uint128_t b_size_rt,
    cuda::std::span<const int4> s0s, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4> cs, cuda::std::span<int4> ys,
    cuda::std::span<int4, 4> pi, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor =
      gpu::detail::VdpfGpuKernelDescriptor<bucket_bits, GroupTag, Prg, XorHash, Hash, __uint128_t>;
  constexpr size_t kOutputs = size_t{1} << in_bits;
  constexpr size_t kBuckets = kBucketCount;
  constexpr size_t kCwStride = bucket_bits + 1;
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  if constexpr (kOutputs > static_cast<size_t>(std::numeric_limits<int>::max() / 3) ||
      kOutputs > kMax / sizeof(int4)) {
    return cudaErrorInvalidValue;
  } else {
    if (!cuckoo_hash::detail::ValidBucketSize(b_size_rt, kBucketSize, bucket_bits)) return cudaErrorInvalidValue;
    constexpr size_t kRecords = kOutputs * kCuckooChoices;
    size_t cub_bytes = 0;
    cudaError_t status = cub::DeviceRadixSort::SortPairs(nullptr, cub_bytes,
        static_cast<uint32_t *>(nullptr), static_cast<uint32_t *>(nullptr),
        static_cast<gpu::detail::VdmpfFullNode *>(nullptr),
        static_cast<gpu::detail::VdmpfFullNode *>(nullptr), static_cast<int>(kRecords), 0, 32, stream);
    if (status != cudaSuccess) return status;
    gpu::detail::VdmpfFullStorage layout;
    if (!gpu::detail::VdmpfPlanFullStorage(kRecords, kBuckets, cub_bytes, layout))
      return cudaErrorInvalidValue;
    if (kBuckets > kMax / sizeof(Cw) / kCwStride ||
        kBuckets > kMax / sizeof(int4) / 4 || s0s.size() < kBuckets ||
        cws.size() < kBuckets * kCwStride || cs.size() < kBuckets * 4 ||
        ys.size() < kOutputs || s0s.data() == nullptr || cws.data() == nullptr ||
        cs.data() == nullptr || ys.data() == nullptr || pi.data() == nullptr)
      return cudaErrorInvalidValue;
    int query_grid, record_grid, bucket_grid;
    status = gpu::detail::VdmpfFullLaunchBlocks(kOutputs, threads_per_block, query_grid);
    if (status != cudaSuccess) return status;
    status = gpu::detail::VdmpfFullLaunchBlocks(kRecords, threads_per_block, record_grid);
    if (status != cudaSuccess) return status;
    status = gpu::detail::VdmpfFullLaunchBlocks(kBuckets, threads_per_block, bucket_grid);
    if (status != cudaSuccess) return status;
    gpu::detail::DeviceScratch arena(stream);
    status = arena.Allocate<unsigned char>(layout.bytes);
    if (status != cudaSuccess) return status;
    void *temp_storage = arena.Get<unsigned char>();
    auto *keys = gpu::detail::VdmpfFullStorageAt<uint32_t>(temp_storage, layout.keys);
    auto *sorted_keys = gpu::detail::VdmpfFullStorageAt<uint32_t>(temp_storage, layout.sorted_keys);
    auto *nodes = gpu::detail::VdmpfFullStorageAt<gpu::detail::VdmpfFullNode>(temp_storage, layout.nodes);
    auto *sorted_nodes = gpu::detail::VdmpfFullStorageAt<gpu::detail::VdmpfFullNode>(temp_storage, layout.sorted_nodes);
    auto *counts = gpu::detail::VdmpfFullStorageAt<uint32_t>(temp_storage, layout.counts);
    auto *offsets = gpu::detail::VdmpfFullStorageAt<uint32_t>(temp_storage, layout.offsets);
    auto *bucket_nodes = gpu::detail::VdmpfFullStorageAt<gpu::detail::VdmpfFullNode>(temp_storage, layout.bucket_nodes);
    auto *bucket_scratch = gpu::detail::VdmpfFullStorageAt<gpu::detail::VdmpfFullNode>(temp_storage, layout.bucket_scratch);
    auto *cub_storage = gpu::detail::VdmpfFullStorageAt<unsigned char>(temp_storage, layout.cub);
    gpu::detail::VdmpfFullClearCountsKernel<<<bucket_grid, threads_per_block, 0, stream>>>(counts, kBuckets);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    gpu::detail::VdmpfFullLeavesKernel<in_bits, bucket_bits, Descriptor, In, Prp>
        <<<query_grid, threads_per_block, 0, stream>>>(b, sigma, b_size_rt,
            kCuckooChoices, s0s.data(), cws.data(), cs.data(), ys.data(), keys, nodes, counts, kOutputs,
            prg_, xor_hash_, hash_, prp_);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    status = cub::DeviceRadixSort::SortPairs(cub_storage, cub_bytes,
        keys, sorted_keys, nodes, sorted_nodes, static_cast<int>(kRecords), 0, 32, stream);
    if (status != cudaSuccess) return status;
    gpu::detail::VdmpfFullOffsetsKernel<<<1, 1, 0, stream>>>(counts, offsets, kBuckets);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    auto *roots = sorted_nodes;
    auto *root_scratch = nodes;
    size_t max_count = kRecords;
    for (uint32_t stride = 1; max_count > 1; stride *= 2) {
      gpu::detail::VdmpfFullBucketPairsKernel<<<record_grid, threads_per_block, 0, stream>>>(
          roots, root_scratch, counts, offsets, kBuckets, stride, kRecords, hash_);
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      auto *swap = roots;
      roots = root_scratch;
      root_scratch = swap;
      max_count = (max_count + 1) / 2;
    }
    gpu::detail::VdmpfFullBindRootsKernel<<<bucket_grid, threads_per_block, 0, stream>>>(
        cs.data(), roots, counts, offsets, bucket_nodes, kBuckets, hash_);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    auto *final_nodes = bucket_nodes;
    auto *final_scratch = bucket_scratch;
    size_t count = kBuckets;
    while (count > 1) {
      int grid;
      status = gpu::detail::VdmpfFullLaunchBlocks((count + 1) / 2, threads_per_block, grid);
      if (status != cudaSuccess) return status;
      gpu::detail::VdmpfFullPairsKernel<<<grid, threads_per_block, 0, stream>>>(final_nodes, final_scratch, count, hash_);
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      auto *swap = final_nodes;
      final_nodes = final_scratch;
      final_scratch = swap;
      count = (count + 1) / 2;
    }
    gpu::detail::VdmpfFullFinishKernel<<<1, 1, 0, stream>>>(final_nodes, pi.data());
    return gpu::detail::ReleaseScratch(cudaGetLastError(), {&arena});
  }
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
