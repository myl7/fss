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
 * Gen makes 2 * (kDepth + 1) hash calls and Eval kDepth + 1, the costs of PackedHalfTreeDpf.
 * A key is 16 * (kDepth + 2) bytes with kDepth = in_bits - 7.
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
#include <memory>
#include <cuda/std/span>
#include <utility>
#include <stdexcept>

namespace fss {

template <typename Scheme>
class OwnedPackedParityTree;

namespace detail {
class SchemeAccess;
}  // namespace detail

/**
 * 2-party DCF scheme over F2 from the packed Half-Tree DPF (Grotto construction).
 *
 * @tparam in_bits Input domain bit size. Must exceed 7, the levels a 1-bit output block covers.
 * @tparam Hash Unary CCR hash backend. See CcrHashable and PackedHalfTreeDpf.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @ingroup fss_api
 *
 * Gen() and scalar Eval() are callable from host and device when the backend supports
 * the execution space and all accessed storage is accessible there.
 * Preprocess() and both EvalAll() overloads are host operations.
 * For N = 2^in_bits and B = N/128, preprocessing stores B packed leaf blocks
 * and a parity tree of 2*B-1 bool entries over whole blocks.
 * Tree queries use O(kDepth) XOR operations and no hash calls.
 * Raw scalar Eval() and the leaf-prefix route can give different individual
 * shares. Both parties must use the same route. Raw EvalAll() and preprocessed
 * evaluation use the same leaf-prefix route and give identical shares.
 */
template <int in_bits, typename Hash, typename In = uint>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
class HalfTreeGrottoDcf {
  using DpfType = PackedHalfTreeDpf<in_bits, 1, Hash, In>;

public:
  /** Number of one-bit output lanes in each packed leaf block. */
  static constexpr int kLanes = DpfType::kLanes;

  /**
   * Tree depth that is actually expanded, see PackedHalfTreeDpf::kDepth.
   */
  static constexpr int kDepth = DpfType::kDepth;

  using Cw = typename DpfType::Cw;  ///< Correction word type of the packed Half-Tree DPF.

  /**
   * A party's key, the PackedHalfTreeDpf key with out_bits = 1 and beta = 1.
   */
  using Key = typename DpfType::Key;

  /**
   * Configure the CCR hash and public instance seed S.
   *
   * @param hash Unary CCR hash backend used throughout generation and evaluation.
   * @param instance_seed Uniformly sampled public seed S, fixed for this instance.
   */
  __host__ __device__ explicit HalfTreeGrottoDcf(Hash hash, int4 instance_seed);

  /**
   * Key generation method. Delegates to PackedHalfTreeDpf::Gen() with beta = 1.
   *
   * @param cws Output array of kDepth + 1 shared correction words, including the final output correction word.
   * @param s0s 2 initial seeds. Users can randomly sample them.
   * @param a The secret comparison threshold in [0, 2^in_bits).
   */
  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a);

  /**
   * Key generation method for split Cws. Delegates to
   * PackedHalfTreeDpf::Gen() with beta = 1.
   *
   * Layer l is stored at cws[l*stride] and the packed-output correction word at
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter.
   *
   * @param cws Output storage; its capacity includes kDepth*stride+1 elements.
   * @param s0s 2 initial seeds. Users can randomly sample them.
   * @param a The secret comparison threshold in [0, 2^in_bits).
   * @param stride Positive element stride between layer entries. The default
   * stride of 1 stores the components contiguously.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre Inputs and outputs do not overlap.
   * @pre All accessed storage is accessible from the calling execution space.
   * @post cws and s0s[party] form the corresponding party's key.
   * @post Only the described prefix is written, and storage beyond it is untouched.
   */
  __host__ __device__ void Gen(cuda::std::span<int4> cws,
      cuda::std::span<const int4, 2> s0s, In a, size_t stride = 1);

  /**
   * Evaluation method, see "Evaluation" in the file details.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param s0 This party's seed from generation.
   * @param cws The kDepth + 1 shared correction words returned by Gen().
   * @param x Query point in [0, 2^in_bits).
   * @return bool share such that share_0 XOR share_1 = 1[alpha <= x].
   */
  __host__ __device__ bool Eval(bool b, int4 s0, const Cw cws[], In x);

  /**
   * Evaluation method for split Cws, see "Evaluation" in the file details.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer l at index l*stride and the
   * packed-output correction word at index kDepth*stride.
   * @param x Query point in [0, 2^in_bits).
   * @param stride Positive stride between Cw entries.
   * @return bool share such that share_0 XOR share_1 = 1[alpha <= x].
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre All accessed storage is accessible from the calling execution space.
   * @pre A stride or capacity violation aborts with a diagnostic in Debug
   * builds on both the host and the device, and is undefined behavior in
   * Release builds.
   */
  __host__ __device__ bool Eval(bool b, int4 s0, cuda::std::span<const int4> cws,
      In x, size_t stride = 1);

  /**
   * Borrowed, read-only packed leaves and their block-parity tree.
   *
   * For B = 2^kDepth, blocks[j] holds 128 DPF output-share bits in increasing
   * input order. p has 2*B-1 entries in level order. Its leaf p[B-1+j]
   * is the XOR of all bits in blocks[j]. Each internal entry is its children's XOR.
   * The arrays contain B int4 entries and 2*B-1 bool entries per party.
   * Their combined payload is B * sizeof(int4) + (2*B-1) * sizeof(bool) bytes.
   * Keeping both parties' trees uses twice that amount.
   * For in_bits = 8, kDepth = 1, so B = 2 and the arrays contain
   * two int4 entries and three bool entries per party.
   * Keys, evaluation outputs, allocator overhead, and temporary storage
   * used internally by GPU full-domain evaluation are separate.
   * Neither storage is owned or transferred between host and device by the view.
   * @ingroup fss_api
   */
  struct ParityTreeView {
    const int4 *blocks;  ///< Borrowed array of B packed output blocks.
    const bool *p;  ///< Borrowed array of 2*B-1 block-parity entries.
  };

  /**
   * Build packed leaves and their parity tree in caller-owned host storage.
   *
   * Reuses PackedHalfTreeDpf::EvalAll() and builds the block tree bottom-up.
   * For B = 2^kDepth, allocate B int4 entries for blocks and 2*B-1 bool
   * entries for p. Their combined payload is
   * B * sizeof(int4) + (2*B-1) * sizeof(bool) bytes per party.
   * Compute both element counts and byte counts without overflow before allocation.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws The kDepth+1 shared correction words returned by Gen().
   * @param blocks Writable output array of at least 2^kDepth int4 entries.
   * @param p Writable output array of at least 2^(kDepth+1)-1 bool entries.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre Packed-block and parity-tree element counts and byte sizes fit size_t.
   * @post ParityTreeView{blocks, p} can be passed to Eval() and EvalAll().
   */
  void Preprocess(bool b, int4 s0, const Cw cws[], int4 blocks[], bool p[], int par_depth = 0);

  /**
   * Build packed leaves and their parity tree from split Cws.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter. The produced arrays match the Cw form entry for entry.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws Correction words, layer l at index l*stride.
   * @param blocks Writable output array of at least 2^kDepth int4 entries.
   * @param p Writable output array of at least 2^(kDepth+1)-1 bool entries.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre Packed-block and parity-tree element counts and byte sizes fit size_t.
   * @throws std::invalid_argument if stride is zero or too large or cws is
   * shorter than kDepth*stride+1.
   * @post ParityTreeView{blocks, p} can be passed to Eval() and EvalAll().
   */
  void Preprocess(bool b, int4 s0, cuda::std::span<const int4> cws, int4 blocks[],
      bool p[], int par_depth = 0, size_t stride = 1);

  /**
   * Allocate and populate packed leaves and their parity tree in host storage.
   *
   * The returned owner holds independent storage; this instance keeps no key
   * or preprocessed tree. Its View() is accepted by Eval() and EvalAll().
   * For B = 2^kDepth, the owned arrays contain B int4 entries and 2*B-1
   * bool entries. Their combined payload is
   * B * sizeof(int4) + (2*B-1) * sizeof(bool) bytes per party, excluding
   * the owner object and allocator overhead.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared correction words returned by Gen().
   * @param par_depth Nonnegative CPU split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @return Move-only owner of the populated host storage.
   * @pre All allocated element counts and byte sizes fit size_t.
   */
  OwnedPackedParityTree<HalfTreeGrottoDcf> PreprocessOwned(
      bool b, int4 s0, const Cw cws[], int par_depth = 0);

  /**
   * Allocate and populate packed leaves and their parity tree from split Cws.
   *
   * The returned owner holds independent storage; this instance keeps no key
   * or preprocessed tree. Its View() is accepted by Eval() and EvalAll().
   * Layer l is read from cws[l*stride], as in the Cw form.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Correction words, layer l at index l*stride.
   * @param par_depth Nonnegative CPU split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @return Move-only owner of the populated host storage.
   * @pre stride > 0 and cws holds at least kDepth*stride+1 elements.
   * @pre All allocated element counts and byte sizes fit size_t.
   * @throws std::invalid_argument if stride is zero or too large or cws is
   * shorter than kDepth*stride+1.
   */
  OwnedPackedParityTree<HalfTreeGrottoDcf> PreprocessOwned(bool b, int4 s0,
      cuda::std::span<const int4> cws, int par_depth = 0, size_t stride = 1);

  /**
   * Query a preprocessed tree without hash calls.
   *
   * @param pt Borrowed packed parity tree. Both buffers must remain alive and
   * accessible in the execution space of this call.
   * @param x Query point in [0, 2^in_bits).
   * @return This party's XOR share of alpha <= x, using the leaf-prefix route.
   * @pre Packed-block and parity-tree element counts and byte sizes fit size_t.
   */
  __host__ __device__ static bool Eval(ParityTreeView pt, In x);

  /**
   * Evaluate all comparison shares directly from seeds and correction words on the host.
   *
   * Expands B = 2^kDepth packed blocks, then scans their bits into ys.
   * Allocates B temporary int4 entries in addition to the underlying packed
   * evaluator's traversal scratch. Does not build a parity tree.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws The kDepth+1 shared correction words returned by Gen().
   * @param ys Output array of at least 2^in_bits bool entries.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The output and packed-block element counts and byte sizes fit size_t.
   * @post ys[x] is this party's XOR share of alpha <= x, using the leaf-prefix route.
   */
  void EvalAll(bool b, int4 s0, const Cw cws[], bool ys[], int par_depth = 0);

  /**
   * Evaluate all comparison shares from split Cws on the host.
   *
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The construction carries no control bits, so there is no
   * packed-control parameter. Outputs match the Cw form share for share.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's initial seed.
   * @param cws Correction words, layer l at index l*stride.
   * @param ys Output array of at least 2^in_bits bool entries.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @param stride Positive stride between Cw entries.
   * @pre stride > 0.
   * @pre With M = numeric_limits<size_t>::max()/sizeof(int4), stride <= (M-1)/kDepth
   * and cws has at least kDepth*stride+1 elements.
   * @pre The output and packed-block element counts and byte sizes fit size_t.
   * @throws std::invalid_argument if stride is zero or too large or cws is
   * shorter than kDepth*stride+1.
   * @post ys[x] is this party's XOR share of alpha <= x, using the leaf-prefix route.
   */
  void EvalAll(bool b, int4 s0, cuda::std::span<const int4> cws, bool ys[],
      int par_depth = 0, size_t stride = 1);

  /**
   * Evaluate all comparison shares from preprocessed packed leaves on the host.
   *
   * Uses O(2^in_bits) XOR work, no hash calls, and no additional allocation.
   * @param pt Borrowed parity tree with storage accessible on the host.
   * @param ys Output array of at least 2^in_bits bool entries, disjoint from pt's storage.
   * @pre The output and packed-block element counts and byte sizes fit size_t.
   * @post ys matches raw EvalAll() for the same party, configuration and key components.
   */
  static void EvalAll(ParityTreeView pt, bool ys[]);

  /**
   * Build packed leaves and their parity tree on the GPU.
   *
   * Called on a host-accessible instance. Reuses PackedHalfTreeDpf::EvalAllGpu().
   * The hash value and instance seed are passed as kernel arguments; resources
   * referenced by the hash must already be device-accessible and allow concurrent use.
   * No allocation, upload, deep copy, or synchronization is performed.
   * For B = 2^kDepth, the caller-provided arrays use
   * B * sizeof(int4) + (2*B-1) * sizeof(bool) bytes per party.
   * @tparam frontier_depth Frontier depth, capped at kDepth. Defaults to 16.
   * @tparam threads_per_block Power-of-two block size in [1,1024], capped at
   * the frontier width. Defaults to 256. At most eight tail levels may remain.
   * The block-root depth is derived from these dimensions and must be at most 30.
   * @param b Party index.
   * @param s0 This party's initial seed, passed by value to the kernel.
   * @param cws Device storage of at least kDepth+1 correction words.
   * @param blocks Device output of at least 2^kDepth packed int4 blocks.
   * @param tree Device output of at least 2^(kDepth+1)-1 bool parity entries.
   * @param stream Stream on the device owning the referenced resources.
   * @pre Packed-block and parity-tree counts and byte counts fit size_t.
   * @pre Buffers and backend resources remain valid until queued work completes.
   * @return Immediate CUDA status. Invalid buffers or unaddressable counts
   * return cudaErrorInvalidValue; oversized grids return cudaErrorInvalidConfiguration.
   * Work remains asynchronous, and execution errors are observed on the stream.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t PreprocessGpu(bool b, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<int4> blocks,
      cuda::std::span<bool> tree, cudaStream_t stream = 0) const
    requires(kDepth < sizeof(size_t) * 8);

  /**
   * Build packed leaves and their parity tree on the GPU from split Cws.
   *
   * Reuses PackedHalfTreeDpf::EvalAllGpu(). Layer l is read from cws[l*stride] and
   * the packed-output correction word from cws[kDepth*stride]. The hash value and
   * instance seed are passed as kernel arguments; resources referenced by the hash
   * must already be device-accessible and allow concurrent use. No allocation,
   * upload, deep copy, or synchronization is performed.
   *
   * @param b Party index.
   * @param s0 This party's initial seed, passed by value to the kernel.
   * @param cws Device correction words, layer l at index l*stride.
   * @param blocks Device output of at least 2^kDepth packed int4 blocks.
   * @param tree Device output of at least 2^(kDepth+1)-1 bool parity entries.
   * @param stream Stream on the device owning the referenced resources.
   * @param stride Positive stride between Cw entries.
   * @pre stride > 0 and cws holds at least kDepth*stride+1 elements.
   * @pre Packed-block and parity-tree counts and byte counts fit size_t.
   * @pre Buffers and backend resources remain valid until queued work completes.
   * @return Immediate CUDA status. Invalid buffers or unaddressable counts
   * return cudaErrorInvalidValue; oversized grids return cudaErrorInvalidConfiguration.
   * Work remains asynchronous, and execution errors are observed on the stream.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t PreprocessGpu(bool b, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<int4> blocks, cuda::std::span<bool> tree,
      cudaStream_t stream = 0, size_t stride = 1) const
    requires(kDepth < sizeof(size_t) * 8);

  /**
   * Evaluate all comparison shares on the GPU without building a parity tree.
   *
   * Follows the leaf-prefix route, the same route as EvalAll(). Per-party shares
   * are not comparable across routes.
   *
   * Expands packed blocks, unpacks their bits, and scans their prefix XOR.
   * The host instance passes its hash value and instance seed to the packed
   * evaluator. Referenced backend resources must already be device-accessible.
   * Packed-block and CUB scan scratch are allocated and released on stream.
   * No backend upload, deep copy, synchronization, or event management occurs.
   * @tparam frontier_depth Packed frontier depth, capped at kDepth. Defaults to 16.
   * @tparam threads_per_block Power-of-two block size in [1,1024], default 256.
   * The packed evaluator caps its frontier width and permits at most eight tail levels.
   * @param b Party index.
   * @param s0 This party's seed, passed by value.
   * @param cws Device storage of at least kDepth+1 correction words.
   * @param ys Device output of at least 2^in_bits bool entries, in input order.
   * @param stream Stream used for evaluation, scanning, and temporary storage.
   * @pre Buffers and backend resources remain valid until queued work completes.
   * @return The first immediate CUDA failure, including allocation or release.
   * Invalid buffers/counts return cudaErrorInvalidValue. Work is asynchronous.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<bool> ys,
      cudaStream_t stream = 0) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Evaluate all comparison shares on the GPU from split Cws.
   *
   * Follows the leaf-prefix route, as in the Cw form. Per-party shares are not
   * comparable across routes.
   *
   * Expands packed blocks, unpacks their bits, and scans their prefix XOR.
   * Layer l is read from cws[l*stride] and the packed-output correction word from
   * cws[kDepth*stride]. The host instance passes its hash value and instance seed
   * to the packed evaluator. Referenced backend resources must already be
   * device-accessible. Packed-block and CUB scan scratch are allocated and
   * released on stream. No backend upload, deep copy, synchronization, or event
   * management occurs.
   *
   * @param b Party index.
   * @param s0 This party's seed, passed by value.
   * @param cws Device correction words, layer l at index l*stride.
   * @param ys Device output of at least 2^in_bits bool entries, in input order.
   * @param stream Stream used for evaluation, scanning, and temporary storage.
   * @param stride Positive stride between Cw entries.
   * @pre stride > 0 and cws holds at least kDepth*stride+1 elements.
   * @pre Buffers and backend resources remain valid until queued work completes.
   * @return The first immediate CUDA failure, including allocation or release.
   * Invalid buffers/counts return cudaErrorInvalidValue. Work is asynchronous.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0, cuda::std::span<const int4> cws,
      cuda::std::span<bool> ys, cudaStream_t stream = 0, size_t stride = 1) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Evaluate all comparison shares from device packed leaves.
   *
   * Unpacks blocks and scans their prefix XOR without hash calls. The parity
   * entries pt.p are not read. Scan scratch is allocated and released on stream.
   * @tparam threads_per_block Power-of-two unpack block size in [1,1024], default 256.
   * @param pt Borrowed view with at least 2^kDepth device packed blocks.
   * @param ys Device output of at least 2^in_bits bool entries, disjoint from pt.blocks.
   * @param stream Stream used for unpacking, scanning, and temporary storage.
   * @pre Input and output buffers remain valid until queued work completes.
   * @return The first immediate CUDA failure, including allocation or release.
   * Invalid buffers/counts return cudaErrorInvalidValue. Work is asynchronous.
   */
  template <int threads_per_block = 256>
  __host__ static cudaError_t EvalAllGpu(ParityTreeView pt,
      cuda::std::span<bool> ys, cudaStream_t stream = 0)
    requires(in_bits < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  Hash hash_;
  int4 instance_seed_;

  __host__ __device__ DpfType MakeDpf() const;

  // Split Cws view over the caller's array. The host methods and the
  // split GPU kernels share this one view; see detail::PackedHalfTreeCwsView.
  using CwsView = detail::PackedHalfTreeCwsView<Cw>;

  // Shared path evaluation over Cw arrays and split Cws views.
  template <typename Cws>
  __host__ __device__ bool EvalCws(bool b, int4 s0, Cws cws, In x);

  // Shared preprocessing over Cw arrays and split Cws views.
  template <typename Cws>
  void PreprocessCws(
      bool b, int4 s0, Cws cws, int4 blocks[], bool p[], int par_depth);

  // Shared full-domain evaluation over Cw arrays and split Cws views.
  template <typename Cws>
  void EvalAllCws(bool b, int4 s0, Cws cws, bool ys[], int par_depth);

  // Shared owner-producing preprocessing over Cw arrays and split Cws views.
  template <typename Cws>
  OwnedPackedParityTree<HalfTreeGrottoDcf> PreprocessOwnedCws(
      bool b, int4 s0, Cws cws, int par_depth);

  // Block-parity entries of the packed leaves, shared by both Preprocess() forms.
  static void BuildBlockTree(const int4 blocks[], bool p[]);

  // Prefix or suffix parity selected by the final tree bit of x.
  static __host__ __device__ bool LeafParity(int4 block, In x);

  static void ScanBlocks(const int4 blocks[], bool ys[]);
};

/**
 * Own packed host leaves and their block-parity tree.
 *
 * This move-only owner does not transfer storage to a GPU. Views borrow both
 * allocations and remain valid while those allocations remain alive.
 * @tparam Scheme HalfTreeGrottoDcf specialization describing the tree.
 * @ingroup fss_api
 */
template <typename Scheme>
class OwnedPackedParityTree {
public:
  /**
   * Take ownership of populated packed leaves and block-parity entries.
   * @param blocks Array of 2^Scheme::kDepth packed leaf blocks.
   * @param p Level-order parity tree with 2^(Scheme::kDepth+1)-1 bool entries.
   */
  explicit OwnedPackedParityTree(std::unique_ptr<int4[]> blocks, std::unique_ptr<bool[]> p);

  /** Return a view borrowing both allocations from this live owner. */
  typename Scheme::ParityTreeView View() const &;

  /** Reject a temporary owner whose borrowed view would immediately dangle. */
  typename Scheme::ParityTreeView View() const && = delete;

private:
  std::unique_ptr<int4[]> blocks_;
  std::unique_ptr<bool[]> p_;
};

// -----------------------------------------------------------------------------

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline HalfTreeGrottoDcf<in_bits, Hash, In>::HalfTreeGrottoDcf(Hash hash, int4 instance_seed)
    : hash_(hash), instance_seed_(instance_seed) {}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline void HalfTreeGrottoDcf<in_bits, Hash, In>::Gen(
    Cw cws[], const int4 s0s[2], In a) {
  MakeDpf().Gen(cws, s0s, a, 1);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline void HalfTreeGrottoDcf<in_bits, Hash, In>::Gen(
    cuda::std::span<int4> cws, cuda::std::span<const int4, 2> s0s, In a, size_t stride) {
  // The key is the packed DPF key with beta = 1, so the split layout is
  // PackedHalfTreeDpf's and the packed generator owns the stride and span guards.
  MakeDpf().Gen(cws, s0s, a, 1, stride);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline bool HalfTreeGrottoDcf<in_bits, Hash, In>::Eval(bool b, int4 s0, const Cw cws[], In x) {
  return EvalCws(b, s0, cws, x);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline bool HalfTreeGrottoDcf<in_bits, Hash, In>::Eval(bool b, int4 s0,
    cuda::std::span<const int4> cws, In x, size_t stride) {
  util::RequireArg(stride > 0, "HalfTreeGrottoDcf::Eval: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  util::RequireArg(stride <= (kMaxElements - 1) / static_cast<size_t>(kDepth),
      "HalfTreeGrottoDcf::Eval: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  util::RequireArg(cws.size() >= required,
      "HalfTreeGrottoDcf::Eval: cws must hold kDepth*stride+1 elements");
  return EvalCws(b, s0, CwsView{cws.data(), stride}, x);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <typename Cws>
__host__ __device__ inline bool HalfTreeGrottoDcf<in_bits, Hash, In>::EvalCws(
    bool b, int4 s0, Cws cws, In x) {
  DpfType dpf = MakeDpf();
  detail::GrottoTurns turns;
  int4 node = dpf.Walk(b, s0, cws, x, turns);
  return turns.share ^ LeafParity(dpf.ConvertLeaf(node, cws[kDepth].s), x);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::Preprocess(
    bool b, int4 s0, const Cw cws[], int4 blocks[], bool p[], int par_depth) {
  PreprocessCws(b, s0, cws, blocks, p, par_depth);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::Preprocess(bool b, int4 s0,
    cuda::std::span<const int4> cws, int4 blocks[], bool p[], int par_depth, size_t stride) {
  if (stride == 0)
    throw std::invalid_argument("HalfTreeGrottoDcf::Preprocess: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(kDepth))
    throw std::invalid_argument("HalfTreeGrottoDcf::Preprocess: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  if (cws.size() < required)
    throw std::invalid_argument("HalfTreeGrottoDcf::Preprocess: cws must hold kDepth*stride+1 elements");
  PreprocessCws(b, s0, CwsView{cws.data(), stride}, blocks, p, par_depth);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::PreprocessCws(
    bool b, int4 s0, Cws cws, int4 blocks[], bool p[], int par_depth) {
  assert(par_depth >= 0);
  static_assert(detail::FitsBinaryTree<bool, kDepth>(),
      "parity-tree count and byte size must fit in size_t");
  // The packed scheme owns the expansion body, shared with its own EvalAll() forms.
  MakeDpf().template EvalAllCws(b, s0, cws, blocks, par_depth);
  BuildBlockTree(blocks, p);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::BuildBlockTree(const int4 blocks[], bool p[]) {
  constexpr size_t count = detail::FullDomainSize<int4, kDepth>();
  for (size_t j = 0; j < count; ++j) {
    // A lane-only x selects an inclusive prefix, so lane 127 covers the whole block.
    p[count - 1 + j] = LeafParity(blocks[j], static_cast<In>(kLanes - 1));
  }
  for (size_t j = count - 2; j < count - 1; --j) {
    p[j] = p[2 * j + 1] ^ p[2 * j + 2];
  }
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline bool HalfTreeGrottoDcf<in_bits, Hash, In>::Eval(
    ParityTreeView pt, In x) {
  static_assert(detail::FitsBinaryTree<bool, kDepth>(),
      "parity-tree count and byte size must fit in size_t");
  constexpr size_t count = detail::FullDomainSize<int4, kDepth>();
  size_t block = static_cast<size_t>(x >> 7);
  bool prefix = LeafParity(pt.blocks[block], static_cast<In>(x & (kLanes - 1)));
  size_t cur = count - 1 + block;
  while (cur != 0) {
    // Right children have even indices. Include their left siblings' subtrees.
    if ((cur & 1) == 0) prefix ^= pt.p[cur - 1];
    cur = (cur - 1) / 2;
  }
  return prefix;
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAll(
    bool b, int4 s0, const Cw cws[], bool ys[], int par_depth) {
  EvalAllCws(b, s0, cws, ys, par_depth);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAll(bool b, int4 s0,
    cuda::std::span<const int4> cws, bool ys[], int par_depth, size_t stride) {
  if (stride == 0)
    throw std::invalid_argument("HalfTreeGrottoDcf::EvalAll: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(kDepth))
    throw std::invalid_argument("HalfTreeGrottoDcf::EvalAll: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  if (cws.size() < required)
    throw std::invalid_argument("HalfTreeGrottoDcf::EvalAll: cws must hold kDepth*stride+1 elements");
  EvalAllCws(b, s0, CwsView{cws.data(), stride}, ys, par_depth);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <typename Cws>
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAllCws(
    bool b, int4 s0, Cws cws, bool ys[], int par_depth) {
  assert(par_depth >= 0);
  constexpr size_t count = detail::FullDomainSize<int4, kDepth>();
  auto blocks = std::make_unique<int4[]>(count);
  // The packed scheme owns the expansion body, shared with its own EvalAll() forms.
  MakeDpf().template EvalAllCws(b, s0, cws, blocks.get(), par_depth);
  ScanBlocks(blocks.get(), ys);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAll(ParityTreeView pt, bool ys[]) {
  ScanBlocks(pt.blocks, ys);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline void HalfTreeGrottoDcf<in_bits, Hash, In>::ScanBlocks(const int4 blocks[], bool ys[]) {
  static_assert(detail::FitsFullDomain<bool, in_bits>(),
      "full-domain output count and byte size must fit in size_t");
  constexpr size_t count = detail::FullDomainSize<int4, kDepth>();
  bool prefix = false;
  for (size_t j = 0; j < count; ++j) {
    int4 block = blocks[j];
    unsigned int words[4] = {static_cast<unsigned int>(block.x), static_cast<unsigned int>(block.y),
        static_cast<unsigned int>(block.z), static_cast<unsigned int>(block.w)};
    for (int word = 0; word < 4; ++word) {
      for (int bit = 0; bit < 32; ++bit) {
        prefix ^= ((words[word] >> bit) & 1u) != 0;
        ys[j * kLanes + word * 32 + bit] = prefix;
      }
    }
  }
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline bool HalfTreeGrottoDcf<in_bits, Hash, In>::LeafParity(int4 block, In x) {
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

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
__host__ __device__ inline typename HalfTreeGrottoDcf<in_bits, Hash, In>::DpfType
HalfTreeGrottoDcf<in_bits, Hash, In>::MakeDpf() const {
  return DpfType{hash_, instance_seed_};
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeGrottoDcf<in_bits, Hash, In>::PreprocessGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<int4> blocks, cuda::std::span<bool> tree,
    cudaStream_t stream) const
  requires(kDepth < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = gpu::detail::GrottoGpuLaunchShape<kDepth, frontier_depth, threads_per_block>;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  constexpr size_t kTreeSize = kBlocks + (kBlocks - 1);
  if (!detail::FitsFullDomain<int4, kDepth>() || !detail::FitsBinaryTree<bool, kDepth>() ||
      cws.size() < kDepth + 1 || blocks.size() < kBlocks || tree.size() < kTreeSize ||
      cws.data() == nullptr || blocks.data() == nullptr || tree.data() == nullptr)
    return cudaErrorInvalidValue;
  constexpr size_t kParityBlocks = kBlocks / LaunchShape::kThreadsPerBlock +
      (kBlocks % LaunchShape::kThreadsPerBlock != 0);
  if (kParityBlocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  cudaError_t status = MakeDpf().template EvalAllGpu<frontier_depth, threads_per_block>(
      b, s0, cws, blocks, stream);
  if (status != cudaSuccess) return status;
  gpu::detail::GrottoPackedParityKernel<<<kParityBlocks, LaunchShape::kThreadsPerBlock, 0, stream>>>(
      blocks.data(), tree.data() + kBlocks - 1, kBlocks);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  return gpu::detail::BuildGrottoParityTree<kDepth, LaunchShape>(tree.data(), stream);
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeGrottoDcf<in_bits, Hash, In>::PreprocessGpu(
    bool b, int4 s0, cuda::std::span<const int4> cws, cuda::std::span<int4> blocks,
    cuda::std::span<bool> tree, cudaStream_t stream, size_t stride) const
  requires(kDepth < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = gpu::detail::GrottoGpuLaunchShape<kDepth, frontier_depth, threads_per_block>;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  constexpr size_t kTreeSize = kBlocks + (kBlocks - 1);
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (!detail::FitsFullDomain<int4, kDepth>() || !detail::FitsBinaryTree<bool, kDepth>() ||
      stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(kDepth) ||
      cws.size() < static_cast<size_t>(kDepth) * stride + 1 ||
      blocks.size() < kBlocks || tree.size() < kTreeSize ||
      cws.data() == nullptr || blocks.data() == nullptr || tree.data() == nullptr)
    return cudaErrorInvalidValue;
  constexpr size_t kParityBlocks = kBlocks / LaunchShape::kThreadsPerBlock +
      (kBlocks % LaunchShape::kThreadsPerBlock != 0);
  if (kParityBlocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  cudaError_t status = MakeDpf().template EvalAllGpu<frontier_depth, threads_per_block>(
      b, s0, cws, blocks, stream, stride);
  if (status != cudaSuccess) return status;
  gpu::detail::GrottoPackedParityKernel<<<kParityBlocks, LaunchShape::kThreadsPerBlock, 0, stream>>>(
      blocks.data(), tree.data() + kBlocks - 1, kBlocks);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  return gpu::detail::BuildGrottoParityTree<kDepth, LaunchShape>(tree.data(), stream);
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}


template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<bool> ys, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  constexpr size_t kOutputs = size_t{1} << in_bits;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  if (!detail::FitsFullDomain<bool, in_bits>() || !detail::FitsFullDomain<int4, kDepth>() ||
      kOutputs > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      cws.size() < kDepth + 1 || ys.size() < kOutputs ||
      cws.data() == nullptr || ys.data() == nullptr) return cudaErrorInvalidValue;
  int unpack_grid;
  cudaError_t status = gpu::detail::GrottoUnpackGrid(kOutputs, threads_per_block, unpack_grid);
  if (status != cudaSuccess) return status;
  size_t bytes = 0;
  status = cub::DeviceScan::InclusiveScan(nullptr, bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::DeviceScratch packed_storage(stream);
  gpu::detail::DeviceScratch scan_storage(stream);
  status = packed_storage.Allocate<int4>(kBlocks);
  if (status != cudaSuccess) return status;
  status = scan_storage.Allocate<unsigned char>(bytes == 0 ? 1 : bytes);
  if (status != cudaSuccess) return status;
  auto *blocks = packed_storage.Get<int4>();
  status = MakeDpf().template EvalAllGpu<frontier_depth, threads_per_block>(
      b, s0, cws, cuda::std::span<int4>(blocks, kBlocks), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::GrottoUnpackBitsKernel<<<unpack_grid, threads_per_block, 0, stream>>>(
      blocks, ys.data(), kOutputs);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveScan(scan_storage.Get<unsigned char>(), bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  return gpu::detail::ReleaseScratch(status, {&packed_storage, &scan_storage});
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const int4> cws, cuda::std::span<bool> ys,
    cudaStream_t stream, size_t stride) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  constexpr size_t kOutputs = size_t{1} << in_bits;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (!detail::FitsFullDomain<bool, in_bits>() || !detail::FitsFullDomain<int4, kDepth>() ||
      stride == 0 || stride > (kMaxElements - 1) / static_cast<size_t>(kDepth) ||
      kOutputs > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      cws.size() < static_cast<size_t>(kDepth) * stride + 1 || ys.size() < kOutputs ||
      cws.data() == nullptr || ys.data() == nullptr) return cudaErrorInvalidValue;
  int unpack_grid;
  cudaError_t status = gpu::detail::GrottoUnpackGrid(kOutputs, threads_per_block, unpack_grid);
  if (status != cudaSuccess) return status;
  size_t bytes = 0;
  status = cub::DeviceScan::InclusiveScan(nullptr, bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::DeviceScratch packed_storage(stream);
  gpu::detail::DeviceScratch scan_storage(stream);
  status = packed_storage.Allocate<int4>(kBlocks);
  if (status != cudaSuccess) return status;
  status = scan_storage.Allocate<unsigned char>(bytes == 0 ? 1 : bytes);
  if (status != cudaSuccess) return status;
  auto *blocks = packed_storage.Get<int4>();
  status = MakeDpf().template EvalAllGpu<frontier_depth, threads_per_block>(
      b, s0, cws, cuda::std::span<int4>(blocks, kBlocks), stream, stride);
  if (status != cudaSuccess) return status;
  gpu::detail::GrottoUnpackBitsKernel<<<unpack_grid, threads_per_block, 0, stream>>>(
      blocks, ys.data(), kOutputs);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveScan(scan_storage.Get<unsigned char>(), bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  return gpu::detail::ReleaseScratch(status, {&packed_storage, &scan_storage});
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <int threads_per_block>
__host__ inline cudaError_t HalfTreeGrottoDcf<in_bits, Hash, In>::EvalAllGpu(
    ParityTreeView pt, cuda::std::span<bool> ys, cudaStream_t stream)
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  constexpr size_t kOutputs = size_t{1} << in_bits;
  if (!detail::FitsFullDomain<bool, in_bits>() || !detail::FitsFullDomain<int4, kDepth>() ||
      kOutputs > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      ys.size() < kOutputs || ys.data() == nullptr || pt.blocks == nullptr) return cudaErrorInvalidValue;
  int unpack_grid;
  cudaError_t status = gpu::detail::GrottoUnpackGrid(kOutputs, threads_per_block, unpack_grid);
  if (status != cudaSuccess) return status;
  size_t bytes = 0;
  status = cub::DeviceScan::InclusiveScan(nullptr, bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::DeviceScratch scan_storage(stream);
  status = scan_storage.Allocate<unsigned char>(bytes == 0 ? 1 : bytes);
  if (status != cudaSuccess) return status;
  gpu::detail::GrottoUnpackBitsKernel<<<unpack_grid, threads_per_block, 0, stream>>>(
      pt.blocks, ys.data(), kOutputs);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveScan(scan_storage.Get<unsigned char>(), bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  return gpu::detail::ReleaseScratch(status, {&scan_storage});
#else
  static_assert(sizeof(Hash) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <typename Scheme>
inline OwnedPackedParityTree<Scheme>::OwnedPackedParityTree(
    std::unique_ptr<int4[]> blocks, std::unique_ptr<bool[]> p)
    : blocks_(std::move(blocks)), p_(std::move(p)) {}

template <typename Scheme>
inline typename Scheme::ParityTreeView OwnedPackedParityTree<Scheme>::View() const & {
  return {blocks_.get(), p_.get()};
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline OwnedPackedParityTree<HalfTreeGrottoDcf<in_bits, Hash, In>> HalfTreeGrottoDcf<in_bits, Hash, In>::PreprocessOwned(
    bool b, int4 s0, const Cw cws[], int par_depth) {
  return PreprocessOwnedCws(b, s0, cws, par_depth);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
inline OwnedPackedParityTree<HalfTreeGrottoDcf<in_bits, Hash, In>> HalfTreeGrottoDcf<in_bits, Hash, In>::PreprocessOwned(
    bool b, int4 s0, cuda::std::span<const int4> cws, int par_depth, size_t stride) {
  if (stride == 0)
    throw std::invalid_argument("HalfTreeGrottoDcf::PreprocessOwned: stride must be positive");
  constexpr size_t kMaxElements = SIZE_MAX / sizeof(int4);
  if (stride > (kMaxElements - 1) / static_cast<size_t>(kDepth))
    throw std::invalid_argument("HalfTreeGrottoDcf::PreprocessOwned: stride exceeds the correction array limit");
  const size_t required = static_cast<size_t>(kDepth) * stride + 1;
  if (cws.size() < required)
    throw std::invalid_argument("HalfTreeGrottoDcf::PreprocessOwned: cws must hold kDepth*stride+1 elements");
  return PreprocessOwnedCws(b, s0, CwsView{cws.data(), stride}, par_depth);
}

template <int in_bits, typename Hash, typename In>
  requires(ValidForInputBits<In, in_bits> && CcrHashable<Hash>)
template <typename Cws>
inline OwnedPackedParityTree<HalfTreeGrottoDcf<in_bits, Hash, In>>
HalfTreeGrottoDcf<in_bits, Hash, In>::PreprocessOwnedCws(
    bool b, int4 s0, Cws cws, int par_depth) {
  using Scheme = HalfTreeGrottoDcf<in_bits, Hash, In>;
  constexpr size_t count = detail::FullDomainSize<int4, Scheme::kDepth>();
  constexpr size_t tree_size = detail::FullBinaryTreeSize<bool, Scheme::kDepth>();
  auto blocks = std::make_unique<int4[]>(count);
  auto p = std::make_unique<bool[]>(tree_size);
  PreprocessCws(b, s0, cws, blocks.get(), p.get(), par_depth);
  return OwnedPackedParityTree<Scheme>(std::move(blocks), std::move(p));
}

namespace gpu::detail {

#if defined(__CUDACC__)

// Hold the scheme behind a tag so the shared packed point kernel does not spell
// the scheme's hash in the host stub signature.
template <int in_bits, typename Hash, typename In>
struct HalfTreeGrottoGpuKernelDescriptor {
  using Scheme = HalfTreeGrottoDcf<in_bits, Hash, In>;
  using Backend = Hash;
};

/**
 * GPU point evaluation of nkeys HalfTree Grotto DCF keys: ys[k] = Eval(b,
 * seeds[k], cws, xs[k]), the party's share of 1[alpha <= xs[k]]. One thread per
 * key. cws holds dense level-major correction words, as the split form of Gen()
 * writes them, including the final correction word row.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds (device memory), one per key.
 * @param cws Level-major correction words (device memory), (kDepth + 1) * nkeys entries.
 * @param xs Evaluation points (device memory), one per key.
 * @param ys Output array (device memory), one share bit per key.
 * @param nkeys Number of keys.
 * @param dcf A HalfTreeGrottoDcf instance (holds the CCR hash and instance seed).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Hash, typename In, typename Query = In>
  requires QueryFor<Query, in_bits>
void HalfTreeGrottoDcfEvalPointGpu(bool b, cuda::std::span<const int4> seeds, cuda::std::span<const int4> cws,
    cuda::std::span<const Query> xs, bool *ys, int nkeys, const HalfTreeGrottoDcf<in_bits, Hash, In> &dcf,
    cudaStream_t stream = 0) {
  if (nkeys < 0)
    throw std::invalid_argument("HalfTreeGrottoDcfEvalPointGpu: nkeys must be nonnegative");
  if (nkeys == 0) return;
  const size_t count = static_cast<size_t>(nkeys);
  constexpr size_t kLevels = static_cast<size_t>(HalfTreeGrottoDcf<in_bits, Hash, In>::kDepth) + 1;
  if (count > SIZE_MAX / kLevels || cws.size() < count * kLevels)
    throw std::invalid_argument("HalfTreeGrottoDcfEvalPointGpu: cws must hold (kDepth+1)*nkeys elements");
  if (seeds.size() < count || xs.size() < count || ys == nullptr)
    throw std::invalid_argument("HalfTreeGrottoDcfEvalPointGpu: seeds, xs, and ys must cover nkeys entries");
  using Descriptor = HalfTreeGrottoGpuKernelDescriptor<in_bits, Hash, In>;
  PackedHalfTreeDpfEvalPointLaunch<unroll, true, in_bits, Descriptor, In, Query, bool>(b, seeds.data(), cws.data(),
      xs.data(), ys, nkeys, fss::detail::SchemeAccess::GetHash(dcf),
      fss::detail::SchemeAccess::GetInstanceSeed(dcf), stream);
}

#endif  // defined(__CUDACC__)

}  // namespace gpu::detail

}  // namespace fss
