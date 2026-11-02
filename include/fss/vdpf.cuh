// SPDX-License-Identifier: Apache-2.0
/**
 * @file vdpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party verifiable distributed point function (VDPF).
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret Sharing_](https://eprint.iacr.org/2021/580) (@ref vdpf "1: the published version").
 *
 * ## Definitions
 *
 * **Point function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, and $b \in \sG_{out}$, a point function $f_{a, b}$ is a function that for any input $x$, the output $y$ has $y = b$ only when $x = a$, otherwise $y = 0$.
 *
 * **VDPF**: extends DPF with verifiability. The evaluation produces both output shares and a proof. Two parties can compare proofs to detect malicious key modification.
 *
 * - Key generation: $Gen(1^\lambda, f_{a, b}) \rightarrow (k_0, k_1)$.
 * - Evaluation: $Eval(k_i, x) \rightarrow (y_{i,x}, \tilde\pi_{i,x})$.
 * - Proof accumulation: $Prove(\{\tilde\pi\}, cs) \rightarrow \pi$.
 * `ProveMerkle`, `EvalAllMerkle`, and `EvalRangeMerkle` replace the sequential chain with a parallel Merkle
 * fold over a fixed ordered sequence of corrected per-point hashes. See doc/vdpf_merkle_fold_security.md.
 * - Verification: $Verify(\pi_0, \pi_1) \rightarrow \{Accept, Reject\}$.
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
 * 1. Leo de Castro, Antigoni Polychroniadou: Lightweight, Maliciously Secure Verifiable Function Secret Sharing. EUROCRYPT 2022: 150-179. <https://doi.org/10.1007/978-3-031-06944-4_6>. @anchor vdpf
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/hash.cuh>
#include <fss/util.cuh>
#include <fss/constraints.cuh>
#include <fss/dpf.cuh>

namespace fss {

namespace detail {
class SchemeAccess;
}  // namespace detail

/**
 * 2-party VDPF scheme.
 *
 * Gen() and scalar Eval() support host and device calls when their backends do.
 * CPU full-domain, range, and proof accumulation run on the host.
 * The Gpu methods launch asynchronous device evaluation.
 * Prove(), EvalAll(), and EvalRange() use chain accumulation.
 * ProveMerkle(), EvalAllMerkle(), and EvalRangeMerkle() use a Merkle fold over a fixed ordered sequence.
 * Both parties must use the same accumulator, key configuration, and ordered inputs.
 * Chain and Merkle proofs have the same representation but different values.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam XorHash See XorHashable. Paper's $H$: maps $(x, s)$ to $4\lambda$ bits.
 * @tparam Hash See Hashable. Paper's $H'$: maps $4\lambda$ bits to $2\lambda$ bits.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 *
 * @ingroup fss_api
 */
template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In = uint>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
class Vdpf {
public:
  /**
   * Correction word.
   *
   * ## Layout
   *
   * According to the paper, there are s, tl, tr to be stored.
   * tl is stored at the clamped bit of s.
   * cws[in_bits] stores the output correction in s. Its tr is false after Gen()
   * and is ignored by evaluation.
   *
   * @ingroup fss_api
   */
  struct __align__(32) Cw {
    int4 s;  ///< Tree seed correction, or the final output correction at cws[in_bits].
    bool tr;  ///< Right control-bit correction. False and ignored in the final output word.
  };
  // For only 1 and aligned memory access on GPU
  static_assert(sizeof(Cw) == 32);

  /**
   * A party's key: combined tree/output corrections, verification seeds, and its initial seed.
   * Both parties share the same cws and cs and differ in the seed.
   *
   * @ingroup fss_api
   */
  struct Key {
    Cw cws[in_bits + 1];  ///< Tree corrections followed by the output correction in the final s field.
    int4 cs[4];  ///< Shared correction seed for verifiable evaluation.
    int4 s0;  ///< This party's initial seed.
  };

  /**
   * Construct a scheme with configured backends.
   *
   * @param prg Configured Prg backend.
   * @param xor_hash Configured XorHash backend.
   * @param hash Configured Hash backend.
   */
  __host__ __device__ explicit Vdpf(Prg prg, XorHash xor_hash, Hash hash);

  /**
   * Generate shared corrections in one attempt without modifying the seeds.
   *
   * @param cws Pre-allocated array of Cw as returns. The array size must be in_bits + 1, including the output correction.
   * @param cs Correction seed output ($4\lambda$ bits).
   * @param s0s Two independent uniformly random initial seed blocks from a
   * cryptographically secure random generator. Gen clears their control bits
   * internally without modifying this input storage.
   * @param a Secret point in [0, 2^in_bits).
   * @param b_buf Payload, clamped and converted to the group element.
   *
   * @par Retrying a failed attempt
   * On return 1, draw two fresh independent uniformly random seed blocks from
   * a cryptographically secure random generator, replace both s0s entries, and
   * call Gen again with the same a and b_buf. Incrementing seed words is not
   * resampling. The caller chooses the attempt limit and handles exhaustion.
   * Reuse of the output buffers is allowed, but discard all cws and cs values
   * from a failed attempt. Only outputs from a successful call form a key.
   *
   * @return 0 on success, 1 if cs is all zeros or $t_0 = t_1$ at the end (caller should resample seeds and retry).
   * @post The input seeds are unchanged. Correction outputs are unspecified on failure.
   *
   * The key for party i consists of cws + cs + s0s[i].
   * cws[in_bits].s holds the output correction and cws[in_bits].tr is false.
   * Evaluation ignores that final tr bit.
   */
  __host__ __device__ int Gen(
      Cw cws[], cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf);

  /**
   * Evaluate one input from caller-owned key components.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws At least in_bits + 1 corrections returned by Gen(), including the final output correction.
   * @param cs Shared correction seed returned by Gen().
   * @param x Input in the scheme's domain.
   * @param y Output share for this input.
   * @return Corrected per-point hash for Prove() or ProveMerkle().
   */
  __host__ __device__ cuda::std::array<int4, 4> Eval(bool b, int4 s0, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4, 4> cs, In x, int4 &y);

  /**
   * Proof accumulation method.
   *
   * Accumulates corrected per-point hashes (from Eval()) into a chain proof.
   * Both parties must supply the same ordered input sequence.
   *
   * @pre pi_tildes is nonempty.
   * @param pi_tildes Corrected per-point hashes returned by Eval().
   * @param cs Returned by Gen().
   * @param pi Borrowed output storage for four proof blocks.
   */
  void Prove(cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4, 4> pi);

  /**
   * Verification method.
   *
   * @param pi0 Party 0 proof.
   * @param pi1 Party 1 proof from the same accumulator and ordered inputs.
   * @return True if proofs match (Accept), false otherwise (Reject).
   */
  __host__ __device__ static bool Verify(cuda::std::span<const int4, 4> pi0, cuda::std::span<const int4, 4> pi1);

  /**
   * Evaluate the full domain and produce a chain proof from key components.
   *
   * The tree traversal supports OpenMP. Zero runs sequentially, and
   * positive depths split the traversal into 2^par_depth subtrees.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws At least in_bits + 1 corrections returned by Gen(), including the final output correction.
   * @param cs Shared correction seed returned by Gen().
   * @param ys Output shares in input order, with at least 2^in_bits entries.
   * @param pi Borrowed output storage for four chain-proof blocks.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The full-domain output count and byte size fit size_t.
   */
  void EvalAll(bool b, int4 s0, cuda::std::span<const Cw> cws,
      cuda::std::span<const int4, 4> cs, cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth = 0);

  /**
   * Evaluate the inclusive range [first, last] and produce a chain proof.
   *
   * Under the VDPF security assumptions, successful verification guarantees
   * at most one nonzero reconstructed output within this range, except with
   * negligible probability. It makes no guarantee outside the range.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param first First input, included in evaluation.
   * @param last Last input, included in evaluation.
   * @param s0 This party's initial seed.
   * @param cws At least in_bits + 1 corrections returned by Gen(), including the final output correction.
   * @param cs Shared correction seed returned by Gen().
   * @param ys Output shares in input order, with at least last - first + 1 entries.
   * @param pi Borrowed output storage for four chain-proof blocks for this range.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre first <= last < 2^in_bits. The range output count and byte size fit size_t.
   * A small range is supported even when the full input domain cannot be
   * materialized in addressable storage.
   */
  void EvalRange(bool b, In first, In last, int4 s0, cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth = 0);

  /**
   * Proof accumulation method (Merkle fold).
   *
   * Parallel alternative to Prove() over a fixed ordered sequence: hashes corrected per-point hashes into the
   * leaves of a binary Merkle tree over the given sequence and folds adjacent nodes with the
   * same hash level by level (an odd trailing node carries to the next level unchanged). The
   * proof is {root, cs[2], cs[3]}, the same 64B shape as the chain's, so Verify() is shared.
   * Every level is an OpenMP parallel batch honoring par_depth, so a par_depth of 0 makes
   * the fold sequential. See doc/vdpf_merkle_fold_security.md.
   *
   * @pre pi_tildes is nonempty.
   * @param pi_tildes Corrected per-point hashes returned by Eval().
   * @param cs Returned by Gen().
   * @param pi Borrowed output storage for four proof blocks.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   */
  void ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4, 4> pi, int par_depth = 0);

  /**
   * Full domain evaluation method with the Merkle-fold proof.
   *
   * Same output shares as EvalAll(); the proof comes from ProveMerkle() over the corrected
   * per-point hashes. The conversion phase and every fold level are OpenMP parallel honoring
   * par_depth, unlike EvalAll()'s sequential phase 2. See doc/vdpf_merkle_fold_security.md.
   *
   * @param b Party index. False for 0 and true for 1. $i$.
   * @param s0 This party's initial seed.
   * @param cws At least in_bits + 1 corrections returned by Gen(), including the final output correction.
   * @param cs Shared correction seed returned by Gen().
   * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
   * @param pi Borrowed output storage for four proof blocks.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The full-domain output count, output bytes, and per-point proof
   * scratch bytes fit size_t.
   */
  void EvalAllMerkle(bool b, int4 s0, cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth = 0);


  /**
   * Launch full-domain CUDA evaluation with the canonical Merkle-fold proof.
   *
   * Uses this host instance's backend values as kernel arguments. The caller
   * prepares device-accessible key/output buffers and every resource referenced
   * by the backends. Temporary block roots are allocated and released in the
   * supplied stream. No backend upload, deep copy, or synchronization is
   * performed. Group, Prg, XorHash, and Hash must support device execution.
   *
   * @tparam frontier_depth Frontier depth, capped at in_bits. Defaults to 16.
   * @tparam threads_per_block Power-of-two block size in [1, 1024], capped at
   * the frontier width. Defaults to 256. The remaining tail has at most eight
   * levels. The block-root depth is derived from these two values, and the
   * number of block roots must not exceed the resolved threads per block.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed, passed by value to the kernel.
   * @param cws Device storage with at least in_bits + 1 shared corrections.
   * @param cs Device storage containing the four shared check-seed blocks.
   * @param ys Device output storage with at least 2^in_bits int4 blocks.
   * @param pi Device output storage for the four-block Merkle proof.
   * @param stream Stream on the device owning the referenced resources.
   * @pre this is host-accessible. Buffers and backend resources remain valid
   * until work completes and permit the kernel's concurrent backend calls.
   * @return The first immediate CUDA allocation, launch, or release status.
   * Invalid buffers or unaddressable output storage return cudaErrorInvalidValue.
   * Execution is
   * asynchronous; the caller observes execution errors when synchronizing.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllMerkleGpu(bool b, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi,
      cudaStream_t stream = 0) const
    requires(in_bits < sizeof(size_t) * 8);


  /**
   * Evaluate [first, last] and apply ProveMerkle to its ordered corrected hashes.
   *
   * ys[j] is the share for first + j. Extra output capacity is unchanged.
   * The proof tree covers this sequence, with odd nodes carried unchanged.
   * Both parties must agree on the range, order, and accumulator externally.
   * For M = last - first + 1, proof scratch payload is at most
   * 96*M + 32*ceil(M/2) bytes, plus 16*M output bytes and allocator overhead.
   * @pre first <= last < 2^in_bits. Output and proof scratch bytes fit size_t.
   * @param par_depth Nonnegative OpenMP split depth. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   */
  void EvalRangeMerkle(bool b, In first, In last, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth = 0);

  /**
   * Launch range evaluation and the canonical ordered-sequence Merkle fold.
   *
   * Uses caller-owned device components and backend resources. Internal scratch
   * is allocated and released on stream. No upload or synchronization occurs.
   * Backend resources must support concurrent device calls and remain valid
   * until the stream completes. The host instance supplies backend values.
   * frontier_depth is relative to each aligned evaluation subtree, capped at
   * its depth. Deep subtrees are split privately to leave at most eight tail
   * levels and a block-root depth of at most 30. threads_per_block is a power
   * of two in [1, 1024], capped at each subtree frontier width.
   * Evaluation fuses share conversion and corrected leaf hashing. The ordered
   * fold uses two buffers with 32*(M + ceil(M/2)) scratch bytes for M inputs,
   * plus 16*M output bytes. These payload sizes exclude allocator overhead.
   * The leaf kernel keeps a bounded depth-first pending-sibling stack of at
   * most eight int4 states per thread and threads_per_block int4 shared
   * frontier states per block.
   * The proof uses actual inputs in increasing order and carries odd nodes.
   * Both parties must agree on the range, order, and Merkle mode externally.
   * @return First immediate CUDA error. Invalid endpoints, buffers, or byte
   * counts return cudaErrorInvalidValue. An oversized grid returns
   * cudaErrorInvalidConfiguration. Execution errors are observed by the caller
   * when synchronizing stream.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalRangeMerkleGpu(bool b, In first, In last, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
      cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi,
      cudaStream_t stream = 0) const;

private:
  friend class detail::SchemeAccess;

  Prg prg_;
  XorHash xor_hash_;
  Hash hash_;

  template <int frontier_depth, int threads_per_block>
  struct GpuLaunchShape;

  void EvalTree(
      int4 st, cuda::std::span<const Cw> cws, cuda::std::span<int4> ys, size_t l, size_t r, int i, int par_depth_);

  void EvalPrunedTree(int4 st, cuda::std::span<const Cw> cws, cuda::std::span<int4> ys, In l, In r, int i,
      int par_depth_, In first, In last);
};

// -----------------------------------------------------------------------------

namespace detail {

template <typename Hash>
__host__ __device__ inline cuda::std::array<int4, 2> VdpfMerkleFoldNode(
    Hash &hash, const cuda::std::array<int4, 2> *nodes, size_t count, size_t index) {
  const size_t left = 2 * index;
  if (left + 1 == count) return nodes[left];
  const int4 msg[4] = {nodes[left][0], nodes[left][1], nodes[left + 1][0], nodes[left + 1][1]};
  return hash.Hash(cuda::std::span<const int4, 4>(msg));
}

}  // namespace detail

namespace gpu::detail {

#if defined(__CUDACC__)

template <int in_bits, int threads_per_block, typename Group, typename Prg,
    typename XorHash, typename Hash, typename In, typename Cw, typename MathScheme>
__global__ void VdpfRangeMerkleLeavesKernel(bool b, In first, int subtree_depth,
    int frontier_depth, int block_root_depth, int4 s0, const Cw *cws, const int4 *cs,
    int4 *ys, cuda::std::array<int4, 2> *leaves, Prg prg, XorHash xor_hash, Hash hash) {
  const int root_depth = in_bits - subtree_depth;
  const int block_subtree_depth = frontier_depth - block_root_depth;
  const int tail_depth = subtree_depth - frontier_depth;
  __shared__ int4 frontier[threads_per_block];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
    for (int level = 0; level < root_depth; ++level) {
      const bool right = (first >> (in_bits - level - 1)) & 1;
      st = fss::detail::SchemeAccess::DpfPathStep<MathScheme>(
          prg, st, cws[level].s, cws[level].tr, right);
    }
    for (int level = 0; level < block_root_depth; ++level) {
      const bool right = (blockIdx.x >> (block_root_depth - level - 1)) & 1;
      st = fss::detail::SchemeAccess::DpfPathStep<MathScheme>(
          prg, st, cws[root_depth + level].s, cws[root_depth + level].tr, right);
    }
    frontier[0] = st;
  }
  __syncthreads();
  for (int level = 0; level < block_subtree_depth; ++level) {
    const bool active = threadIdx.x < (1u << (level + 1));
    int4 st = {0, 0, 0, 0};
    if (active) st = frontier[threadIdx.x >> 1];
    __syncthreads();
    if (active) {
      int4 left, right;
      const int cw_index = root_depth + block_root_depth + level;
      fss::detail::SchemeAccess::DpfExpandNode<MathScheme>(
          prg, st, cws[cw_index].s, cws[cw_index].tr, left, right);
      frontier[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }
  int4 pending[8];
  int4 node = frontier[threadIdx.x];
  const int frontier_level = root_depth + frontier_depth;
  int level = frontier_level;
  const int count = 1 << tail_depth;
  const size_t thread_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t offset = thread_index << tail_depth;
  const auto correction = Group::From(cws[in_bits].s);
  for (int i = 0; i < count; ++i) {
    while (level < in_bits) {
      int4 left, right;
      fss::detail::SchemeAccess::DpfExpandNode<MathScheme>(
          prg, node, cws[level].s, cws[level].tr, left, right);
      pending[level - frontier_level] = right;
      node = left;
      ++level;
    }
    const bool t = util::GetLsb(node);
    const int4 seed = util::SetLsb(node, false);
    auto value = Group::From(seed);
    if (t) value = value + correction;
    if (b) value = -value;
    ys[offset + i] = value.Into();
    const In x = static_cast<In>(first + offset + i);
    auto corrected = xor_hash.Hash(cuda::std::tuple<int4, const int4>{fss::detail::Pack(x), seed});
    if (t) corrected = util::Xor(cuda::std::span<const int4, 4>(corrected),
        cuda::std::span<const int4, 4>(cs, 4));
    leaves[offset + i] = hash.Hash(cuda::std::span<const int4, 4>(corrected));
    if (i + 1 < count) {
      int next = i + 1;
      level = in_bits - 1;
      while ((next & 1) == 0) {
        --level;
        next >>= 1;
      }
      node = pending[level - frontier_level];
      ++level;
    }
  }
}

template <typename Hash>
__global__ void VdpfRangeMerkleFoldKernel(const cuda::std::array<int4, 2> *nodes,
    size_t count, cuda::std::array<int4, 2> *next, Hash hash) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count / 2 + count % 2)
    next[index] = fss::detail::VdpfMerkleFoldNode(hash, nodes, count, index);
}

template <typename Node>
__global__ void VdpfRangeMerkleProofKernel(const Node *root,
    const int4 *cs, int4 *pi) {
  pi[0] = (*root)[0];
  pi[1] = (*root)[1];
  pi[2] = cs[2];
  pi[3] = cs[3];
}

template <int in_bits, int frontier_depth, int block_root_depth, typename Group, typename Prg, typename XorHash, typename Hash, typename In,
    typename Cw, typename MathScheme>
__global__ void VdpfMerkleFrontierKernel(bool b, int4 s0, const Cw *cws, const int4 *cs,
    int4 *ys, cuda::std::array<int4, 2> *roots, Prg prg, XorHash xor_hash, Hash hash) {
  constexpr int kTailDepth = in_bits - frontier_depth;
  constexpr int kBlockSubtreeDepth = frontier_depth - block_root_depth;
  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index >= (1 << frontier_depth)) return;

  // Phase 0: block root walk (top block_root_depth levels) by thread 0.
  __shared__ int4 shared_frontier[1 << kBlockSubtreeDepth];
  if (threadIdx.x == 0) {
    int4 st = util::SetLsb(s0, b);
#pragma unroll
    for (int i = 0; i < block_root_depth; ++i) {
      bool x_bit = (blockIdx.x >> (block_root_depth - 1 - i)) & 1;
      {
        auto node_prg = prg;
        st = fss::detail::SchemeAccess::DpfPathStep<MathScheme>(
            node_prg, st, cws[i].s, cws[i].tr, x_bit);
      }
    }
    shared_frontier[0] = st;
  }
  __syncthreads();

  // Phase 1: breadth-parallel expansion of the block's subtree, levels block_root_depth..frontier_depth.
#pragma unroll
  for (int i = 0; i < kBlockSubtreeDepth; ++i) {
    const bool active = threadIdx.x < (1 << (i + 1));
    int4 n = {0, 0, 0, 0};
    if (active) n = shared_frontier[threadIdx.x >> 1];
    // All parent reads must finish before any child overwrites the frontier.
    __syncthreads();
    if (active) {
      int4 left, right;
      {
        auto node_prg = prg;
        fss::detail::SchemeAccess::DpfExpandNode<MathScheme>(
            node_prg, n, cws[block_root_depth + i].s, cws[block_root_depth + i].tr, left, right);
      }
      shared_frontier[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  // Phase 2: expand the subtree below depth frontier_depth, convert each leaf to the output
  // share and the corrected verification hash, and fold the leaf hashes into
  // the thread's Merkle subtree on the fly.
  int4 nodes[1 << kTailDepth];
  nodes[0] = shared_frontier[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int lev = frontier_depth; lev < in_bits; ++lev) {
#pragma unroll
    for (int k = count - 1; k >= 0; --k) {
      int4 left, right;
      {
        auto node_prg = prg;
        fss::detail::SchemeAccess::DpfExpandNode<MathScheme>(
            node_prg, nodes[k], cws[lev].s, cws[lev].tr, left, right);
      }
      nodes[2 * k] = left;
      nodes[2 * k + 1] = right;
    }
    count *= 2;
  }

  auto ocw_group = Group::From(cws[in_bits].s);
  const int output_offset = thread_index << kTailDepth;
  cuda::std::array<int4, 2> stack[kTailDepth + 1];
#pragma unroll
  for (int k = 0; k < count; ++k) {
    int4 st_cur = nodes[k];
    bool t_cur = util::GetLsb(st_cur);
    int4 s_cur = util::SetLsb(st_cur, false);

    // Output share
    auto g = Group::From(s_cur);
    if (t_cur) g = g + ocw_group;
    if (b) g = -g;
    ys[output_offset + k] = g.Into();

    // Leaf: the corrected verification hash of the point
    int4 x_buf = fss::detail::Pack(static_cast<In>(output_offset + k));
    auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, s_cur});
    if (t_cur) {
      pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs, 4));
    }
    int4 leaf_msg[4] = {pi_tilde[0], pi_tilde[1], pi_tilde[2], pi_tilde[3]};
    cuda::std::array<int4, 2> node = hash.Hash(cuda::std::span<const int4, 4>(leaf_msg));

    // Stream the leaf into the thread's subtree: combine with the pending
    // sibling stack entries while the leaf index has trailing one bits.
    int rest = k;
    int lvl = 0;
#pragma unroll
    while (rest != 0 && (rest & 1) == 1) {
      int4 fold_msg[4] = {stack[lvl][0], stack[lvl][1], node[0], node[1]};
      node = hash.Hash(cuda::std::span<const int4, 4>(fold_msg));
      rest >>= 1;
      ++lvl;
    }
    stack[lvl] = node;
  }

  // Block fold of the threads' subtree roots, in thread order: adjacent pairs
  // at each stride, slot idx combining with idx + stride.
  __shared__ cuda::std::array<int4, 2> s_node[1 << kBlockSubtreeDepth];
  s_node[threadIdx.x] = stack[kTailDepth];
  __syncthreads();
#pragma unroll
  for (int stride = 1; stride < (1 << kBlockSubtreeDepth); stride <<= 1) {
    int idx = 2 * stride * threadIdx.x;
    if (idx < (1 << kBlockSubtreeDepth)) {
      int4 fold_msg[4] = {s_node[idx][0], s_node[idx][1], s_node[idx + stride][0], s_node[idx + stride][1]};
      s_node[idx] = hash.Hash(cuda::std::span<const int4, 4>(fold_msg));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) roots[blockIdx.x] = s_node[0];
}

// Folds the per-block subtree roots into the final 64B proof. The root count
// is (1 << frontier_depth) / threads_per_block = 2^block_root_depth, at most one block wide; only the first nroots
// shared slots participate, so the rest stay unwritten and unread.
template <int threads_per_block, typename Hash>
__global__ void VdpfMerkleFinalizeKernel(const cuda::std::array<int4, 2> *roots, int nroots, const int4 *cs,
    int4 *pi, Hash hash) {
  __shared__ cuda::std::array<int4, 2> s_node[threads_per_block];
  if (threadIdx.x < nroots) s_node[threadIdx.x] = roots[threadIdx.x];
  __syncthreads();
  for (int stride = 1; stride < nroots; stride <<= 1) {
    int idx = 2 * stride * threadIdx.x;
    if (idx < nroots) {
      int4 fold_msg[4] = {s_node[idx][0], s_node[idx][1], s_node[idx + stride][0], s_node[idx + stride][1]};
      s_node[idx] = hash.Hash(cuda::std::span<const int4, 4>(fold_msg));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    pi[0] = s_node[0][0];
    pi[1] = s_node[0][1];
    pi[2] = cs[2];
    pi[3] = cs[3];
  }
}

#endif

}  // namespace gpu::detail

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline __host__ __device__ Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Vdpf(Prg prg, XorHash xor_hash, Hash hash)
    : prg_(prg), xor_hash_(xor_hash), hash_(hash) {}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline __host__ __device__ cuda::std::array<int4, 4> Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Eval(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4, 4> cs, In x, int4 &y) {
  int4 s = s0;
  s = util::SetLsb(s, false);
  bool t = b;

  for (int i = 0; i < in_bits; ++i) {
    Cw cw = cws[i];
    int4 s_cw = cw.s;
    bool tl_cw = util::GetLsb(s_cw);
    s_cw = util::SetLsb(s_cw, false);
    bool tr_cw = cw.tr;

    bool x_bit = (x >> (in_bits - 1 - i)) & 1;
    auto [child] = detail::GenHalf<2>(prg_, s, x_bit);
    bool child_t = util::GetLsb(child);
    child = util::SetLsb(child, false);
    if (t) {
      child = util::Xor(child, s_cw);
      child_t = child_t ^ (x_bit ? tr_cw : tl_cw);
    }
    s = child;
    t = child_t;
  }

  // Output share
  auto g = Group::From(s);
  int4 ocw = cws[in_bits].s;
  assert((ocw.w & 1) == 0);
  if (t) g = g + Group::From(ocw);
  if (b) g = -g;

  // Corrected verification hash
  int4 x_buf = fss::detail::Pack(x);

  auto pi_tilde = xor_hash_.Hash(cuda::std::tuple<int4, const int4>{x_buf, s});
  if (t) {
    pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs));
  }
  y = g.Into();
  return pi_tilde;
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Prove(
    cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes, cuda::std::span<const int4, 4> cs,
    cuda::std::span<int4, 4> pi) {
  pi[0] = cs[0];
  pi[1] = cs[1];
  pi[2] = cs[2];
  pi[3] = cs[3];
  for (size_t i = 0; i < pi_tildes.size(); ++i) {
    cuda::std::array<int4, 4> h_input =
        util::Xor(cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_tildes[i]));
    auto h_out = hash_.Hash(cuda::std::span<const int4, 4>(h_input));
    pi[0] = util::Xor(pi[0], h_out[0]);
    pi[1] = util::Xor(pi[1], h_out[1]);
  }
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline __host__ __device__ bool Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Verify(
    cuda::std::span<const int4, 4> pi0, cuda::std::span<const int4, 4> pi1) {
  for (int i = 0; i < 4; ++i) {
    if (pi0[i].x != pi1[i].x || pi0[i].y != pi1[i].y || pi0[i].z != pi1[i].z || pi0[i].w != pi1[i].w) return false;
  }
  return true;
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalAll(
    bool b, int4 s0, cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs, cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth) {
  assert(par_depth >= 0);
  constexpr size_t kOutputCount = detail::FullDomainSize<int4, in_bits>();
  EvalRange(b, In{0}, static_cast<In>(kOutputCount - 1), s0, cws, cs, ys, pi, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalRange(
    bool b, In first, In last, int4 s0, cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth) {
  assert(par_depth >= 0);
  In max_input = static_cast<In>(~In{0});
  if constexpr (in_bits < sizeof(In) * 8) {
    max_input = static_cast<In>((In{1} << in_bits) - 1);
  }
  assert(first <= last && last <= max_input);
  const __uint128_t distance = static_cast<__uint128_t>(last) - first;
  assert(distance < static_cast<size_t>(-1) / sizeof(int4));
  const size_t num_inputs = static_cast<size_t>(distance) + 1;
  assert(cws.size() >= in_bits + 1);
  assert(ys.size() >= num_inputs);

  const int4 st = util::SetLsb(s0, b);
  const int par_depth_ = fss::detail::ResolveParDepth(par_depth, in_bits);

  // Traversal stores the packed seed and control bit in output storage temporarily.
  if (par_depth_ == 0) {
    EvalPrunedTree(st, cws, ys, In{0}, max_input, 0, 0, first, last);
  } else {
    const int threads = fss::detail::ResolveParThreads(par_depth_, num_inputs);

#pragma omp parallel num_threads(threads)
#pragma omp single
    EvalPrunedTree(st, cws, ys, In{0}, max_input, 0, par_depth_, first, last);
  }
  // Phase 2: sequential output computation and proof accumulation
  pi[0] = cs[0];
  pi[1] = cs[1];
  pi[2] = cs[2];
  pi[3] = cs[3];

  int4 ocw = cws[in_bits].s;
  assert((ocw.w & 1) == 0);
  auto ocw_group = Group::From(ocw);
  for (size_t offset = 0; offset < num_inputs; ++offset) {
    const In x = static_cast<In>(first + offset);
    int4 sj = ys[offset];
    bool tj = util::GetLsb(sj);
    sj = util::SetLsb(sj, false);

    // Output share
    auto g = Group::From(sj);
    if (tj) g = g + ocw_group;
    if (b) g = -g;
    ys[offset] = g.Into();

    // Proof accumulation
    int4 x_buf = fss::detail::Pack(x);

    auto pi_tilde = xor_hash_.Hash(cuda::std::tuple<int4, const int4>{x_buf, sj});
    if (tj) {
      pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs));
    }

    cuda::std::array<int4, 4> h_input =
        util::Xor(cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_tilde));
    auto h_out = hash_.Hash(cuda::std::span<const int4, 4>(h_input));
    pi[0] = util::Xor(pi[0], h_out[0]);
    pi[1] = util::Xor(pi[1], h_out[1]);
  }
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::ProveMerkle(
    cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes, cuda::std::span<const int4, 4> cs,
    cuda::std::span<int4, 4> pi, int par_depth) {
  assert(par_depth >= 0);
  size_t count = pi_tildes.size();
  assert(count >= 1);

  std::vector<cuda::std::array<int4, 2>> nodes(count);
  if constexpr (HasHash4<Hash>) {
    const size_t groups = count / 4;
    fss::detail::ParallelFor(par_depth, groups, [&](size_t g) {
      auto out =
          hash_.Hash4(cuda::std::span<const cuda::std::array<int4, 4>, 4>(pi_tildes.data() + 4 * g, 4));
      for (size_t j = 0; j < 4; ++j) nodes[4 * g + j] = out[j];
    });
    for (size_t i = 4 * groups; i < count; ++i) {
      nodes[i] = hash_.Hash(cuda::std::span<const int4, 4>(pi_tildes[i]));
    }
  } else {
    fss::detail::ParallelFor(par_depth, count, [&](size_t i) {
      nodes[i] = hash_.Hash(cuda::std::span<const int4, 4>(pi_tildes[i]));
    });
  }

  while (count > 1) {
    const size_t next_count = count / 2 + count % 2;
    std::vector<cuda::std::array<int4, 2>> next(next_count);
    if constexpr (HasHash4<Hash>) {
      const size_t pairs = count / 2;
      const size_t groups = pairs / 4;
      fss::detail::ParallelFor(par_depth, groups, [&](size_t g) {
        cuda::std::array<cuda::std::array<int4, 4>, 4> msgs;
        for (size_t j = 0; j < 4; ++j) {
          const auto *p = nodes.data() + 8 * g + 2 * j;
          msgs[j] = {p[0][0], p[0][1], p[1][0], p[1][1]};
        }
        auto out = hash_.Hash4(cuda::std::span<const cuda::std::array<int4, 4>, 4>(msgs));
        for (size_t j = 0; j < 4; ++j) next[4 * g + j] = out[j];
      });
      // Remaining complete pairs; an odd final node is carried by the helper.
      for (size_t i = 4 * groups; i < next_count; ++i) {
        next[i] = detail::VdpfMerkleFoldNode(hash_, nodes.data(), count, i);
      }
    } else {
      fss::detail::ParallelFor(par_depth, next_count, [&](size_t i) {
        next[i] = detail::VdpfMerkleFoldNode(hash_, nodes.data(), count, i);
      });
    }
    nodes = std::move(next);
    count = nodes.size();
  }

  pi[0] = nodes[0][0];
  pi[1] = nodes[0][1];
  pi[2] = cs[2];
  pi[3] = cs[3];
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalAllMerkle(
    bool b, int4 s0, cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, int par_depth) {
  constexpr size_t kOutputCount = detail::FullDomainSize<cuda::std::array<int4, 4>, in_bits>();
  EvalRangeMerkle(b, In{0}, static_cast<In>(kOutputCount - 1), s0, cws, cs, ys, pi, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalRangeMerkle(
    bool b, In first, In last, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4, 4> cs, cuda::std::span<int4> ys,
    cuda::std::span<int4, 4> pi, int par_depth) {
  assert(par_depth >= 0);
  In max_input = static_cast<In>(~In{0});
  if constexpr (in_bits < sizeof(In) * 8) {
    max_input = static_cast<In>((In{1} << in_bits) - 1);
  }
  assert(first <= last && last <= max_input);
  const __uint128_t distance = static_cast<__uint128_t>(last) - first;
  constexpr size_t kMaxBytes = size_t(-1);
  assert(distance < kMaxBytes / sizeof(cuda::std::array<int4, 4>));
  const size_t kOutputCount = static_cast<size_t>(distance) + 1;
  const size_t next_count = kOutputCount / 2 + kOutputCount % 2;
  assert(kOutputCount <= kMaxBytes / (sizeof(cuda::std::array<int4, 4>) + sizeof(cuda::std::array<int4, 2>)));
  const size_t initial_bytes = kOutputCount *
      (sizeof(cuda::std::array<int4, 4>) + sizeof(cuda::std::array<int4, 2>));
  assert(next_count <= (kMaxBytes - initial_bytes) / sizeof(cuda::std::array<int4, 2>));
  const int4 st = util::SetLsb(s0, b);
  assert(cws.size() >= in_bits + 1);
  assert(ys.size() >= kOutputCount);

  int par_depth_ = fss::detail::ResolveParDepth(par_depth, in_bits);

  // Traversal stores the packed seed and control bit in output storage temporarily.
  if (par_depth_ == 0) {
    EvalPrunedTree(st, cws, ys, In{0}, max_input, 0, 0, first, last);
  } else {
    int threads = fss::detail::ResolveParThreads(par_depth_, kOutputCount);
#pragma omp parallel num_threads(threads)
#pragma omp single
    EvalPrunedTree(st, cws, ys, In{0}, max_input, 0, par_depth_, first, last);
  }

  const size_t n = kOutputCount;
  int4 ocw = cws[in_bits].s;
  assert((ocw.w & 1) == 0);
  auto ocw_group = Group::From(ocw);
  std::vector<cuda::std::array<int4, 4>> leaves(n);
  fss::detail::ParallelFor(par_depth, n, [&](size_t j) {
    int4 sj = ys[j];
    bool tj = util::GetLsb(sj);
    sj = util::SetLsb(sj, false);

    auto g = Group::From(sj);
    if (tj) g = g + ocw_group;
    if (b) g = -g;
    ys[j] = g.Into();

    int4 x_buf = fss::detail::Pack(static_cast<In>(first + j));

    auto pi_tilde = xor_hash_.Hash(cuda::std::tuple<int4, const int4>{x_buf, sj});
    if (tj) {
      leaves[j] = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs));
    } else {
      leaves[j] = pi_tilde;
    }
  });
  ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>>(leaves), cuda::std::span<const int4, 4>(cs), pi, par_depth);
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline __host__ __device__ int Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Gen(
    Cw cws[], cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf) {
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
    // Both 16B halves written so the full 32B slot gets unmasked stores.
    // {s_cw, tr_cw} alone leaves padding undefined: NVCC emits a masked
    // 256-bit store, and the partial-sector write forces a write-allocate
    // round trip per level on GPU (measured ~2x Gen time).
    int4 *cw_slot = reinterpret_cast<int4 *>(&cws[i]);
    cw_slot[0] = s_cw;
    cw_slot[1] = make_int4(tr_cw, 0, 0, 0);
  }

  // Verification hash
  int4 a_buf = fss::detail::Pack(a);

  auto pi_tilde_0 = xor_hash_.Hash(cuda::std::tuple<int4, const int4>{a_buf, s0});
  auto pi_tilde_1 = xor_hash_.Hash(cuda::std::tuple<int4, const int4>{a_buf, s1});
  for (int i = 0; i < 4; ++i) cs[i] = util::Xor(pi_tilde_0[i], pi_tilde_1[i]);

  // A zero cs would let adversarial keys keep both parties' per-point hashes equal while
  // their control bits differ everywhere (see Assumption 2 of doc/vdpf_merkle_fold_security.md).
  if ((cs[0].x | cs[0].y | cs[0].z | cs[0].w | cs[1].x | cs[1].y | cs[1].z | cs[1].w | cs[2].x | cs[2].y |
          cs[2].z | cs[2].w | cs[3].x | cs[3].y | cs[3].z | cs[3].w) == 0) {
    return 1;
  }

  // Reject a failed generation attempt
  if (t0 == t1) return 1;

  // Output correction word
  auto v_cw = Group::From(b_buf) + (-Group::From(s0)) + Group::From(s1);
  if (t1) v_cw = -v_cw;
  cws[in_bits].s = v_cw.Into();
  cws[in_bits].tr = false;

  return 0;
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalTree(
    int4 st, cuda::std::span<const Cw> cws, cuda::std::span<int4> ys, size_t l, size_t r, int i, int par_depth_) {
  if (i == in_bits) {
    assert(l + 1 == r);
    ys[l] = st;
    return;
  }

  bool t = util::GetLsb(st);
  int4 s = st;
  s = util::SetLsb(s, false);

  Cw cw = cws[i];
  int4 s_cw = cw.s;
  bool tl_cw = util::GetLsb(s_cw);
  s_cw = util::SetLsb(s_cw, false);
  bool tr_cw = cw.tr;

  auto [sl, sr] = prg_.Gen(s);

  bool tl = util::GetLsb(sl);
  sl = util::SetLsb(sl, false);
  bool tr = util::GetLsb(sr);
  sr = util::SetLsb(sr, false);

  if (t) {
    sl = util::Xor(sl, s_cw);
    sr = util::Xor(sr, s_cw);
    tl = tl ^ tl_cw;
    tr = tr ^ tr_cw;
  }

  int4 stl = sl;
  stl = util::SetLsb(stl, tl);
  int4 str = sr;
  str = util::SetLsb(str, tr);

  size_t mid = (l + r) / 2;

  if (i < par_depth_) {
#pragma omp task
    EvalTree(stl, cws, ys, l, mid, i + 1, par_depth_);
#pragma omp task
    EvalTree(str, cws, ys, mid, r, i + 1, par_depth_);
#pragma omp taskwait
  } else {
    EvalTree(stl, cws, ys, l, mid, i + 1, par_depth_);
    EvalTree(str, cws, ys, mid, r, i + 1, par_depth_);
  }
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
inline void Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalPrunedTree(
    int4 st, cuda::std::span<const Cw> cws, cuda::std::span<int4> ys, In l, In r, int i,
    int par_depth_, In first, In last) {
  // this subtree is outside the range [first, last]
  if (r < first || l > last) {
    return;
  }
  // this is a full subtree
  if (l >= first && r <= last) {
    const size_t offset = static_cast<size_t>(l - first);
    const size_t count = static_cast<size_t>(r - l) + 1;
    EvalTree(st, cws, ys.subspan(offset, count), 0, count, i, par_depth_);
    return;
  }
  if (i == in_bits) {
    assert(l == r);
    // we only write leaves from [first, last]
    ys[static_cast<size_t>(l - first)] = st;
    return;
  }

  bool t = util::GetLsb(st);
  int4 s = st;
  s = util::SetLsb(s, false);

  Cw cw = cws[i];
  int4 s_cw = cw.s;
  bool tl_cw = util::GetLsb(s_cw);
  s_cw = util::SetLsb(s_cw, false);
  bool tr_cw = cw.tr;

  auto [sl, sr] = prg_.Gen(s);

  bool tl = util::GetLsb(sl);
  sl = util::SetLsb(sl, false);
  bool tr = util::GetLsb(sr);
  sr = util::SetLsb(sr, false);

  if (t) {
    sl = util::Xor(sl, s_cw);
    sr = util::Xor(sr, s_cw);
    tl = tl ^ tl_cw;
    tr = tr ^ tr_cw;
  }

  int4 stl = sl;
  stl = util::SetLsb(stl, tl);
  int4 str = sr;
  str = util::SetLsb(str, tr);

  const In mid = static_cast<In>(l + (r - l) / 2);

  if (i < par_depth_) {
    if (mid >= first) {
#pragma omp task
      EvalPrunedTree(stl, cws, ys, l, mid, i + 1, par_depth_, first, last);
    }
    if (mid < last) {
#pragma omp task
      EvalPrunedTree(str, cws, ys, static_cast<In>(mid + 1), r, i + 1, par_depth_, first, last);
    }
#pragma omp taskwait
  } else {
    if (mid >= first) {
      EvalPrunedTree(stl, cws, ys, l, mid, i + 1, par_depth_, first, last);
    }
    if (mid < last) {
      EvalPrunedTree(str, cws, ys, static_cast<In>(mid + 1), r, i + 1, par_depth_, first, last);
    }
  }
}


template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
template <int frontier_depth, int threads_per_block>
struct Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::GpuLaunchShape {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static constexpr int kFrontierDepth = frontier_depth < in_bits ? frontier_depth : in_bits;
  static_assert(in_bits - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  static constexpr int kThreadsPerBlockLog2 = [] {
    int requested_log2 = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++requested_log2;
    return requested_log2 < kFrontierDepth ? requested_log2 : kFrontierDepth;
  }();
  static constexpr int kThreadsPerBlock = 1 << kThreadsPerBlockLog2;
  static constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  static_assert(kBlockRootDepth <= 30, "gpu block index exceeds the signed-int range");
  static constexpr size_t kBlockCount = size_t{1} << (kBlockRootDepth <= 30 ? kBlockRootDepth : 0);
  static_assert(kBlockCount <= kThreadsPerBlock,
      "merkle block roots must fit within one final block");
};

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalAllMerkleGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws, cuda::std::span<const int4, 4> cs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = GpuLaunchShape<frontier_depth, threads_per_block>;
  constexpr size_t kOutputCount = size_t{1} << in_bits;
  if (cws.size() < in_bits + 1 || ys.size() < kOutputCount ||
      !detail::FitsElementCount<int4>(kOutputCount) ||
      !detail::FitsElementCount<cuda::std::array<int4, 2>>(LaunchShape::kBlockCount) ||
      cws.data() == nullptr || cs.data() == nullptr || ys.data() == nullptr ||
      pi.data() == nullptr) return cudaErrorInvalidValue;
  gpu::detail::DeviceScratch roots_storage(stream);
  cudaError_t status = roots_storage.Allocate<cuda::std::array<int4, 2>>(LaunchShape::kBlockCount);
  if (status != cudaSuccess) return status;
  auto *block_roots = roots_storage.Get<cuda::std::array<int4, 2>>();
  using MathScheme = Dpf<in_bits, Group, Prg, In>;
  gpu::detail::VdpfMerkleFrontierKernel<in_bits, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth, Group, Prg, XorHash, Hash, In, Cw, MathScheme>
      <<<LaunchShape::kBlockCount, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), cs.data(), ys.data(), block_roots, prg_, xor_hash_, hash_);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  gpu::detail::VdpfMerkleFinalizeKernel<LaunchShape::kThreadsPerBlock, Hash>
      <<<1, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          block_roots, static_cast<int>(LaunchShape::kBlockCount), cs.data(), pi.data(), hash_);
  return gpu::detail::ReleaseScratch(cudaGetLastError(), {&roots_storage});
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::EvalRangeMerkleGpu(
    bool b, In first, In last, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<const int4, 4> cs, cuda::std::span<int4> ys,
    cuda::std::span<int4, 4> pi, cudaStream_t stream) const {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
#if defined(__CUDACC__)
  using Node = cuda::std::array<int4, 2>;
  size_t count = 0;
  if (!detail::RangeOutputCount<in_bits>(first, last, ys.size(), count) ||
      cws.size() < in_bits + 1 || cws.data() == nullptr || cs.data() == nullptr ||
      ys.data() == nullptr || pi.data() == nullptr) return cudaErrorInvalidValue;
  const size_t second_count = count / 2 + count % 2;
  constexpr size_t kMaxNodes = size_t(-1) / sizeof(Node);
  if (count > kMaxNodes || second_count > kMaxNodes - count) return cudaErrorInvalidValue;
  constexpr int kThreadBits = [] {
    int bits = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++bits;
    return bits;
  }();
  constexpr int kTailCap = frontier_depth < 30 + kThreadBits ? frontier_depth : 30 + kThreadBits;
  constexpr int kChunkDepth = kTailCap + 8 < 30 + kThreadBits ? kTailCap + 8 : 30 + kThreadBits;
  const size_t fold_count = count / 2 + count % 2;
  const size_t fold_blocks = fold_count / threads_per_block + (fold_count % threads_per_block != 0);
  if (fold_blocks > static_cast<size_t>(INT_MAX)) return cudaErrorInvalidConfiguration;
  gpu::detail::DeviceScratch first_storage(stream), second_storage(stream);
  cudaError_t status = first_storage.Allocate<Node>(count);
  if (status != cudaSuccess) return status;
  status = second_storage.Allocate<Node>(second_count);
  if (status != cudaSuccess) return status;
  auto *nodes = first_storage.Get<Node>();
  auto *next = second_storage.Get<Node>();
  using MathScheme = Dpf<in_bits, Group, Prg, In>;
  status = detail::ForEachAlignedRange<in_bits>(first, last, kChunkDepth,
      [&](In chunk_first, int depth, size_t offset, size_t) {
        const int frontier = depth < frontier_depth ? depth : frontier_depth;
        const int thread_bits = frontier < kThreadBits ? frontier : kThreadBits;
        const int block_root_depth = frontier - thread_bits;
        const size_t blocks = size_t{1} << block_root_depth;
        if (blocks > static_cast<size_t>(INT_MAX)) return cudaErrorInvalidConfiguration;
        gpu::detail::VdpfRangeMerkleLeavesKernel<in_bits, threads_per_block,
            Group, Prg, XorHash, Hash, In, Cw, MathScheme>
            <<<blocks, 1 << thread_bits, 0, stream>>>(b, chunk_first, depth,
                frontier, block_root_depth, s0, cws.data(), cs.data(),
                ys.data() + offset, nodes + offset, prg_, xor_hash_, hash_);
        return cudaGetLastError();
      });
  if (status != cudaSuccess) return status;
  while (count > 1) {
    const size_t next_count = count / 2 + count % 2;
    const size_t blocks = next_count / threads_per_block + (next_count % threads_per_block != 0);
    gpu::detail::VdpfRangeMerkleFoldKernel<<<blocks, threads_per_block, 0, stream>>>(
        nodes, count, next, hash_);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    auto *previous = nodes;
    nodes = next;
    next = previous;
    count = next_count;
  }
  gpu::detail::VdpfRangeMerkleProofKernel<<<1, 1, 0, stream>>>(nodes, cs.data(), pi.data());
  return gpu::detail::ReleaseScratch(cudaGetLastError(), {&first_storage, &second_storage});
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

}  // namespace fss
