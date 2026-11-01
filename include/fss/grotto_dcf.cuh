// SPDX-License-Identifier: Apache-2.0
/**
 * @file grotto_dcf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed comparison function (DCF) over F2 from standard DPF.
 *
 * The scheme is from the paper, [_Grotto: Screaming fast (2+1)-PC for Z_{2^n} via
 * (2,2)-DPFs_](https://eprint.iacr.org/2023/108) (@ref grotto_dcf "1: the published version").
 *
 * Key generation is identical to standard BGI DPF. The comparison functionality
 * emerges from prefix-parity of the DPF control bits: the two parties' control
 * bits of a node XOR to the parity of the leaves below it (Theorem 4 of the paper).
 *
 * Output shares are in F2 (XOR sharing). Each party holds a single bool per query.
 * For inputs x: share_0 XOR share_1 = 1[alpha <= x].
 *
 * ## Evaluation
 *
 * Eval(b, s0, cws, x) runs the paper's prefix-parity algorithm (Section 4) on x's
 * root-to-leaf path: in_bits PRG calls per query and no preprocessing, the cost
 * of a DPF point evaluation. Preprocess() instead expands the whole domain, about
 * 2^in_bits PRG calls, into a parity segment tree that Eval(tree, x) then queries
 * without PRG calls. It pays off once a key answers more than about
 * 2^in_bits / in_bits queries. The two routes give different shares that
 * reconstruct to the same value only in pairs, so both parties must use the
 * same route.
 *
 * ## References
 *
 * 1. Kyle Storrier, Adithya Vadapalli, Allan Lyons, Ryan Henry: Grotto: Screaming fast (2+1)-PC for Z_{2^n} via (2, 2)-DPFs. CCS 2023: 2143-2157. <https://doi.org/10.1145/3576915.3623147>. <https://eprint.iacr.org/2023/108>. @anchor grotto_dcf
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cstddef>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>
#include <fss/constraints.cuh>
#include <cuda/std/span>
#include <memory>
#include <utility>

#include <cstdint>
#include <limits>
#if defined(__CUDACC__)
#include <cub/device/device_scan.cuh>
#endif

namespace fss {

template <typename Scheme>
class OwnedParityTree;

namespace detail {
class SchemeAccess;
}  // namespace detail

/**
 * 2-party DCF scheme over F2 from standard DPF (Grotto construction).
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Prg See Prgable. Must satisfy Prgable<Prg, 2> (same as DPF).
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @ingroup fss_api
 *
 * Gen() and scalar Eval() are callable from host and device when the backend supports
 * the execution space and all accessed storage is accessible there.
 * EvalAll() is a host operation.
 */
template <int in_bits, typename Prg, typename In = uint>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
class GrottoDcf {
  using DpfType = Dpf<in_bits, group::Bytes, Prg, In>;

public:
  using Cw = typename DpfType::Cw;  ///< Correction-word type of the underlying DPF.

  /**
     * A party's key: the correction words plus the party's initial seed.
     * Both parties share the same correction words and differ in the seed.
     * @ingroup fss_api
     */
  struct Key {
    Cw cws[in_bits + 1];  ///< Corrections of the underlying DPF with beta equal to zero.
    int4 s0;  ///< This party's initial seed.
  };

  /**
   * Configure the pseudorandom generator.
   *
   * @param prg Backend used throughout generation and evaluation.
   */
  __host__ __device__ explicit GrottoDcf(Prg prg);

  /**
   * Generate shared DPF corrections with beta equal to zero.
   *
   * @param cws Output array of in_bits + 1 correction words.
   * @param s0s Two initial seeds, retained by the caller.
   * @param a The secret comparison threshold.
   * @pre a is in [0, 2^in_bits).
   */
  __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a);

  /**
   * Evaluate a comparison share along x's path without preprocessing.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared corrections returned by Gen().
   * @param x Query point.
   * @return This party's XOR share of the predicate alpha <= x.
   * @pre x is in [0, 2^in_bits).
   */
  __host__ __device__ bool Eval(bool b, int4 s0, const Cw cws[], In x);

  /**
   * Borrowed, read-only parity tree in level order.
   *
   * For N = 2^in_bits, p has 2*N-1 bool entries. Root is p[0], leaf x
   * is p[N-1+x], and internal node j is p[2*j+1] XOR p[2*j+2].
   * The array payload uses (2*N-1) * sizeof(bool) bytes per party.
   * Keeping both parties' trees uses twice that amount.
   * For in_bits = 8, each tree has 511 bool entries, not 256.
   * Keys, evaluation outputs, allocator overhead, and temporary storage
   * used internally by GPU full-domain evaluation are separate.
   * The view does not own storage or transfer it between host and device.
   * Its storage must remain alive and accessible in the execution space of Eval().
   * @ingroup fss_api
   */
  struct ParityTreeView {
    const bool *p;  ///< Borrowed parity entries, including leaves.
  };

  /**
   * Expand leaf control bits and build a parity tree in caller-owned host storage.
   *
   * Uses O(N) PRG calls and O(N) XOR operations, where N = 2^in_bits.
   * Allocate 2*N-1 bool entries for p, using (2*N-1) * sizeof(bool) bytes
   * per party. This is the same level-order layout as ParityTreeView.
   * Compute the element count and byte count without overflow before allocation.
   * Use PreprocessOwned() for allocating convenience.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared corrections returned by Gen().
   * @param p Output buffer of at least 2^(in_bits+1)-1 writable bool entries.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The parity-tree element count and byte size fit size_t.
   * @post ParityTreeView{p} can be queried through Eval() and EvalAll().
   */
  void Preprocess(bool b, int4 s0, const Cw cws[], bool p[], int par_depth = 0);

  /**
   * Allocate and populate a parity tree in host storage.
   *
   * The returned owner holds independent storage; this instance keeps no key
   * or preprocessed tree. Its View() is accepted by Eval() and EvalAll().
   * The owned array contains 2*N-1 bool entries, where N = 2^in_bits.
   * Its payload is (2*N-1) * sizeof(bool) bytes per party, excluding
   * the owner object and allocator overhead.
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared corrections returned by Gen().
   * @param par_depth Nonnegative CPU split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @return Move-only owner of the populated host storage.
   * @pre All allocated element counts and byte sizes fit size_t.
   */
  OwnedParityTree<GrottoDcf> PreprocessOwned(
      bool b, int4 s0, const Cw cws[], int par_depth = 0);

  /**
   * Query a preprocessed tree with O(in_bits) XOR operations and no PRG calls.
   *
   * @param pt View of a tree filled by Preprocess() or PreprocessGpu().
   * @param x Query point in [0, 2^in_bits).
   * @return This party's XOR share of alpha <= x.
   * @pre The parity-tree element count and byte size fit size_t.
   * @pre Storage is accessible in this execution space.
   */
  __host__ __device__ static bool Eval(ParityTreeView pt, In x);

  /**
   * Evaluate the full domain from a host parity tree without PRG calls.
   *
   * Scans the leaves with O(N) XOR operations, where N = 2^in_bits.
   * @param pt View of a preprocessed tree accessible on the host.
   * @param ys Output array of at least N bool entries, disjoint from pt's storage.
   * @pre The parity-tree element count and byte size fit size_t.
   * @post ys[x] contains this party's XOR share of alpha <= x.
   */
  static void EvalAll(ParityTreeView pt, bool ys[]);

  /**
   * Evaluate comparison shares for the full domain on the host.
   *
   * Expands leaves directly into ys and scans them in place with O(N) PRG and
   * XOR work. No parity-tree storage or preprocessing is required.
   *
   * @param b Party index, false for party 0 and true for party 1.
   * @param s0 This party's seed from generation.
   * @param cws Shared corrections returned by Gen().
   * @param ys Output array of at least 2^in_bits bool entries.
   * @param par_depth Nonnegative OpenMP split depth for this call. Zero runs sequentially.
   * Positive depths require backends that support concurrent calls.
   * @pre The output element count and byte size fit size_t.
   * @post ys[x] contains this party's XOR share of alpha <= x.
   */
  void EvalAll(bool b, int4 s0, const Cw cws[], bool ys[], int par_depth = 0);

  /**
   * Build a parity tree on the GPU in caller-owned device storage.
   *
   * Called on a host-accessible instance. The PRG value is passed to the
   * kernel; any resources it references must already be device-accessible.
   * The PRG must support device execution and concurrent use.
   * No allocation, upload, deep copy, or synchronization is performed.
   * @tparam frontier_depth Frontier depth, capped at in_bits. Defaults to 16.
   * @tparam threads_per_block Power-of-two block size in [1,1024], capped at
   * the frontier width. Defaults to 256. At most eight tail levels may remain.
   * The block-root depth is derived from these dimensions and must be at most 30.
   * @param b Party index.
   * @param s0 This party's initial seed, passed by value to the kernel.
   * @param cws Device storage of at least in_bits+1 corrections.
   * @param tree Device output of at least 2^(in_bits+1)-1 bool entries.
   * Its array payload is (2*N-1) * sizeof(bool) bytes per party, where N = 2^in_bits.
   * @param stream Stream on the device owning the referenced resources.
   * @pre Tree element counts and byte counts fit size_t.
   * @pre Buffers and backend resources remain valid until queued work completes.
   * @return Immediate CUDA status. Invalid buffers or unaddressable counts
   * return cudaErrorInvalidValue; oversized grids return cudaErrorInvalidConfiguration.
   * Work remains asynchronous, and execution errors are observed on the stream.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t PreprocessGpu(bool b, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<bool> tree,
      cudaStream_t stream = 0) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Evaluate all comparison shares on the GPU without building a parity tree.
   *
   * The host instance passes its PRG value to the expansion kernel. Referenced
   * backend resources must already be device-accessible and permit concurrent use.
   * Temporary scan storage is queried, allocated, and released on stream.
   * No backend upload, deep copy, synchronization, or event management occurs.
   * @tparam frontier_depth Frontier depth, capped at in_bits. Defaults to 16.
   * @tparam threads_per_block Power-of-two block size in [1,1024], capped at
   * the frontier width. Defaults to 256. At most eight tail levels may remain.
   * @param b Party index.
   * @param s0 This party's seed, passed by value.
   * @param cws Device storage of at least in_bits+1 corrections.
   * @param ys Device output of at least 2^in_bits bool entries, in input order.
   * @param stream Stream used for expansion, scanning, and temporary storage.
   * @pre Buffers and referenced backend resources remain valid until work completes.
   * @return The first immediate CUDA failure, including allocation or release.
   * Invalid buffers/counts return cudaErrorInvalidValue. Work is asynchronous.
   */
  template <int frontier_depth = 16, int threads_per_block = 256>
  __host__ cudaError_t EvalAllGpu(bool b, int4 s0,
      cuda::std::span<const Cw> cws, cuda::std::span<bool> ys,
      cudaStream_t stream = 0) const
    requires(in_bits < sizeof(size_t) * 8);

  /**
   * Evaluate all comparison shares from a device parity tree.
   *
   * Scans the stored leaves without PRG calls. Temporary scan storage is
   * queried, allocated, and released on stream, without implicit synchronization.
   * @param pt Borrowed tree produced by PreprocessGpu(), with 2^(in_bits+1)-1 entries.
   * @param ys Device output of at least 2^in_bits bool entries, disjoint from pt.
   * @param stream Stream used for the scan and temporary storage.
   * @pre Input and output storage remain valid until queued work completes.
   * @return The first immediate CUDA failure. Invalid buffers/counts return
   * cudaErrorInvalidValue. Completion remains asynchronous.
   */
  __host__ static cudaError_t EvalAllGpu(ParityTreeView pt,
      cuda::std::span<bool> ys, cudaStream_t stream = 0)
    requires(in_bits < sizeof(size_t) * 8);

private:
  friend class detail::SchemeAccess;

  Prg prg_;

  // Expand the DPF tree and write leaf control bits.
  void ExpandTree(bool b, int4 s0, const Cw cws[], bool t[], int par_depth);

  void ExpandTreeRec(int4 st, const Cw cws[], bool t[], size_t l, size_t r, int i, int par_depth_);
};

/**
 * Own a host parity-tree allocation and expose a borrowed, read-only view.
 *
 * This move-only owner does not copy storage to a GPU. Views remain valid until
 * the allocation is destroyed or replaced, including after ownership is moved.
 * @tparam Scheme GrottoDcf specialization describing the tree.
 * @ingroup fss_api
 */
template <typename Scheme>
class OwnedParityTree {
public:
  /**
   * Take ownership of an already populated host parity tree.
   * @param p Level-order tree with 2^(in_bits+1)-1 entries for Scheme.
   */
  explicit OwnedParityTree(std::unique_ptr<bool[]> p);

  /** Return a view borrowing this live owner's host allocation. */
  typename Scheme::ParityTreeView View() const &;

  /** Reject a temporary owner whose borrowed view would immediately dangle. */
  typename Scheme::ParityTreeView View() const && = delete;

private:
  std::unique_ptr<bool[]> p_;
};

// -----------------------------------------------------------------------------


namespace gpu::detail {

template <int tree_depth, int frontier_depth, int threads_per_block>
struct GrottoGpuLaunchShape {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(1 <= threads_per_block && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  static constexpr int kFrontierDepth = frontier_depth < tree_depth ? frontier_depth : tree_depth;
  static_assert(tree_depth - kFrontierDepth <= 8,
      "gpu frontier leaves more than eight tail levels; specify a larger frontier_depth");
  static constexpr int kThreadsPerBlockLog2 = [] {
    int log2 = 0;
    for (int count = threads_per_block; count > 1; count >>= 1) ++log2;
    return log2 < kFrontierDepth ? log2 : kFrontierDepth;
  }();
  static constexpr int kThreadsPerBlock = 1 << kThreadsPerBlockLog2;
  static constexpr int kBlockRootDepth = kFrontierDepth - kThreadsPerBlockLog2;
  static_assert(kBlockRootDepth <= 30, "gpu block index exceeds the signed-int range");
  static constexpr size_t kBlocks = size_t{1} << (kBlockRootDepth <= 30 ? kBlockRootDepth : 0);
};

#if defined(__CUDACC__)

struct GrottoGpuBoolXor {
  __host__ __device__ bool operator()(bool lhs, bool rhs) const { return lhs != rhs; }
};

static __global__ void GrottoUnpackBitsKernel(const int4 *blocks, bool *leaves, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int4 block = blocks[i / 128];
  unsigned word = (i % 128) / 32;
  unsigned value = word == 0 ? static_cast<unsigned>(block.x) :
      word == 1 ? static_cast<unsigned>(block.y) : word == 2 ? static_cast<unsigned>(block.z) : static_cast<unsigned>(block.w);
  leaves[i] = ((value >> (i % 32)) & 1u) != 0;
}

inline cudaError_t GrottoUnpackGrid(size_t count, int threads, int &grid) {
  const size_t blocks = count / threads + (count % threads != 0);
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  grid = static_cast<int>(blocks);
  return cudaSuccess;
}

template <int input_bits, int frontier_depth, int block_root_depth, typename Dpf, typename Prg>
__global__ void GrottoLeafBitsKernel(bool party, int4 s0, const typename Dpf::Cw *cws,
    bool *leaves, Prg prg) {
  constexpr int kTailDepth = input_bits - frontier_depth;
  constexpr int kBlockSubtreeDepth = frontier_depth - block_root_depth;
  const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  __shared__ int4 shared_frontier[1 << kBlockSubtreeDepth];
  if (threadIdx.x == 0) {
    int4 state = util::SetLsb(s0, party);
#pragma unroll
    for (int level = 0; level < block_root_depth; ++level) {
      bool take_right = (blockIdx.x >> (block_root_depth - 1 - level)) & 1;
      auto node_prg = prg;
      state = fss::detail::SchemeAccess::DpfPathStep<Dpf>(
          node_prg, state, cws[level].s, cws[level].tr, take_right);
    }
    shared_frontier[0] = state;
  }
  __syncthreads();

#pragma unroll
  for (int level = 0; level < kBlockSubtreeDepth; ++level) {
    const bool active = threadIdx.x < (1 << (level + 1));
    int4 state = {0, 0, 0, 0};
    if (active) state = shared_frontier[threadIdx.x >> 1];
    __syncthreads();
    if (active) {
      int4 left, right;
      {
        auto node_prg = prg;
        fss::detail::SchemeAccess::DpfExpandNode<Dpf>(node_prg, state,
            cws[block_root_depth + level].s, cws[block_root_depth + level].tr, left, right);
      }
      shared_frontier[threadIdx.x] = (threadIdx.x & 1) ? right : left;
    }
    __syncthreads();
  }

  int4 nodes[1 << kTailDepth];
  nodes[0] = shared_frontier[threadIdx.x];
  int count = 1;
#pragma unroll
  for (int level = frontier_depth; level < input_bits; ++level) {
#pragma unroll
    for (int i = count - 1; i >= 0; --i) {
      int4 left, right;
      {
        auto node_prg = prg;
        fss::detail::SchemeAccess::DpfExpandNode<Dpf>(node_prg, nodes[i],
            cws[level].s, cws[level].tr, left, right);
      }
      nodes[2 * i] = left;
      nodes[2 * i + 1] = right;
    }
    count *= 2;
  }

  const size_t output_offset = static_cast<size_t>(thread_index) << kTailDepth;
#pragma unroll
  for (int i = 0; i < count; ++i) leaves[output_offset + i] = util::GetLsb(nodes[i]);
}

static __global__ void GrottoParityLevelKernel(bool *tree, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  size_t index = count - 1 + i;
  tree[index] = tree[2 * index + 1] ^ tree[2 * index + 2];
}

template <int top_level>
__global__ void GrottoParityTopKernel(bool *tree) {
#pragma unroll
  for (int level = top_level; level >= 0; --level) {
    int count = 1 << level;
    if (threadIdx.x < count) {
      int index = count - 1 + threadIdx.x;
      tree[index] = tree[2 * index + 1] ^ tree[2 * index + 2];
    }
    __syncthreads();
  }
}

static __global__ void GrottoPackedParityKernel(const int4 *blocks, bool *leaves, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int4 block = blocks[i];
  unsigned value = static_cast<unsigned>(block.x) ^ static_cast<unsigned>(block.y) ^
      static_cast<unsigned>(block.z) ^ static_cast<unsigned>(block.w);
  leaves[i] = (__popc(value) & 1) != 0;
}

template <int leaf_bits, typename LaunchShape>
inline cudaError_t BuildGrottoParityTree(bool *tree, cudaStream_t stream) {
  constexpr int kThreads = LaunchShape::kThreadsPerBlock;
  constexpr int kTop = LaunchShape::kThreadsPerBlockLog2 < leaf_bits - 1
      ? LaunchShape::kThreadsPerBlockLog2 : leaf_bits - 1;
  for (int level = leaf_bits - 1; level > kTop; --level) {
    size_t count = size_t{1} << level;
    size_t grid = count / kThreads + (count % kThreads != 0);
    GrottoParityLevelKernel<<<grid, kThreads, 0, stream>>>(tree, count);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  if constexpr (leaf_bits > 0) {
    GrottoParityTopKernel<kTop><<<1, kThreads, 0, stream>>>(tree);
    return cudaGetLastError();
  }
  return cudaSuccess;
}

#endif

}  // namespace gpu::detail

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
__host__ __device__ inline GrottoDcf<in_bits, Prg, In>::GrottoDcf(Prg prg)
    : prg_(prg) {}

namespace detail {

// State of the prefix-parity algorithm (Section 4 of the paper) on x's path. Called with the
// control bit t of every expanded node and the branch x_bit taken from it, share accumulates
// the control bits of the nodes where the path turns, taking the edge into the root as a left
// edge. GrottoDcf and HalfTreeGrottoDcf share it and differ only in the leaf step.
struct GrottoTurns {
  bool share = false;
  bool prev_bit = false;

  __host__ __device__ void operator()(bool t, bool x_bit) {
    if (x_bit != prev_bit) share ^= t;
    prev_bit = x_bit;
  }

  // Leaf step when each leaf is a single input: XOR the leaf's control bit if the path ends
  // with a left edge.
  __host__ __device__ bool PointLeaf(bool t) const {
    return share != (!prev_bit && t);
  }
};

}  // namespace detail

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
__host__ __device__ inline void GrottoDcf<in_bits, Prg, In>::Gen(Cw cws[], const int4 s0s[2], In a) {
  DpfType dpf{prg_};
  dpf.Gen(cws, s0s, a, int4{0, 0, 0, 0});
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
__host__ __device__ inline bool GrottoDcf<in_bits, Prg, In>::Eval(bool b, int4 s0, const Cw cws[], In x) {
  detail::GrottoTurns turns;
  int4 leaf = DpfType{prg_}.Walk(b, s0, cws, x, turns);
  return turns.PointLeaf(util::GetLsb(leaf));
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
inline void GrottoDcf<in_bits, Prg, In>::Preprocess(bool b, int4 s0, const Cw cws[], bool p[], int par_depth) {
  assert(par_depth >= 0);
  static_assert(detail::FitsBinaryTree<bool, in_bits>(),
      "parity-tree count and byte size must fit in size_t");
  constexpr size_t n = detail::FullDomainSize<bool, in_bits>();
  ExpandTree(b, s0, cws, p + n - 1, par_depth);
  for (size_t j = n - 2; j < n - 1; --j) {
    p[j] = p[2 * j + 1] ^ p[2 * j + 2];
  }
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
__host__ __device__ inline bool GrottoDcf<in_bits, Prg, In>::Eval(ParityTreeView pt, In x) {
  static_assert(detail::FitsBinaryTree<bool, in_bits>(),
      "parity-tree count and byte size must fit in size_t");
  constexpr size_t N = detail::FullDomainSize<bool, in_bits>();
  In e = static_cast<In>(x) + 1;

  // e == 0 means x + 1 overflowed, i.e., e = N (entire domain)
  if (e == 0 || e == N) return pt.p[0];

  bool pi = false;
  size_t cur = 0;
  for (int i = 0; i < in_bits; ++i) {
    bool e_bit = (e >> (in_bits - 1 - i)) & 1;
    if (e_bit) {
      pi ^= pt.p[2 * cur + 1];
      cur = 2 * cur + 2;
    } else {
      cur = 2 * cur + 1;
    }
  }
  return pi;
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
inline void GrottoDcf<in_bits, Prg, In>::EvalAll(ParityTreeView pt, bool ys[]) {
  static_assert(detail::FitsBinaryTree<bool, in_bits>(),
      "parity-tree count and byte size must fit in size_t");
  constexpr size_t n = detail::FullDomainSize<bool, in_bits>();
  bool prefix = false;
  for (size_t x = 0; x < n; ++x) {
    prefix ^= pt.p[n - 1 + x];
    ys[x] = prefix;
  }
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
inline void GrottoDcf<in_bits, Prg, In>::EvalAll(bool b, int4 s0, const Cw cws[], bool ys[], int par_depth) {
  assert(par_depth >= 0);
  constexpr size_t N = detail::FullDomainSize<bool, in_bits>();

  // Phase 1: expand tree to get leaf control bits into ys[]
  ExpandTree(b, s0, cws, ys, par_depth);

  // Phase 2b: prefix-sum scan (running XOR)
  // ys[x] currently holds leaf x's control bit.
  // Transform to: ys[x] = XOR of control bits [0..x] = share of 1[alpha <= x].
  for (size_t x = 1; x < N; ++x) {
    ys[x] = ys[x] ^ ys[x - 1];
  }
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
inline void GrottoDcf<in_bits, Prg, In>::ExpandTree(bool b, int4 s0, const Cw cws[], bool t[], int par_depth) {
  int4 st = s0;
  st = util::SetLsb(st, b);

  size_t l = 0;
  constexpr size_t r = detail::FullDomainSize<bool, in_bits>();
  int i = 0;

  int par_depth_ = fss::detail::ResolveParDepth(par_depth, in_bits);

  if (par_depth_ == 0) {
    ExpandTreeRec(st, cws, t, l, r, i, 0);
    return;
  }
  int threads = fss::detail::ResolveParThreads(par_depth_, r);
#pragma omp parallel num_threads(threads)
#pragma omp single
  ExpandTreeRec(st, cws, t, l, r, i, par_depth_);
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
inline void GrottoDcf<in_bits, Prg, In>::ExpandTreeRec(
    int4 st, const Cw cws[], bool t[], size_t l, size_t r, int i, int par_depth_) {
  bool tc = util::GetLsb(st);
  int4 s = st;
  s = util::SetLsb(s, false);

  if (i == in_bits) {
    assert(l + 1 == r);
    t[l] = tc;
    return;
  }

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

  if (tc) {
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
    ExpandTreeRec(stl, cws, t, l, mid, i + 1, par_depth_);
#pragma omp task
    ExpandTreeRec(str, cws, t, mid, r, i + 1, par_depth_);
#pragma omp taskwait
  } else {
    ExpandTreeRec(stl, cws, t, l, mid, i + 1, par_depth_);
    ExpandTreeRec(str, cws, t, mid, r, i + 1, par_depth_);
  }
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t GrottoDcf<in_bits, Prg, In>::PreprocessGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<bool> tree, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = gpu::detail::GrottoGpuLaunchShape<in_bits, frontier_depth, threads_per_block>;
  constexpr size_t kLeaves = size_t{1} << in_bits;
  constexpr size_t kTreeSize = kLeaves + (kLeaves - 1);
  if (!detail::FitsBinaryTree<bool, in_bits>() || cws.size() < in_bits + 1 ||
      tree.size() < kTreeSize || cws.data() == nullptr || tree.data() == nullptr)
    return cudaErrorInvalidValue;
  constexpr size_t kParityWidth = kLeaves / 2;
  constexpr size_t kParityBlocks = kParityWidth / LaunchShape::kThreadsPerBlock +
      (kParityWidth % LaunchShape::kThreadsPerBlock != 0);
  if (kParityBlocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  gpu::detail::GrottoLeafBitsKernel<in_bits, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth, DpfType>
      <<<LaunchShape::kBlocks, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), tree.data() + kLeaves - 1, prg_);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  return gpu::detail::BuildGrottoParityTree<in_bits, LaunchShape>(tree.data(), stream);
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}


template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
template <int frontier_depth, int threads_per_block>
__host__ inline cudaError_t GrottoDcf<in_bits, Prg, In>::EvalAllGpu(
    bool b, int4 s0, cuda::std::span<const Cw> cws,
    cuda::std::span<bool> ys, cudaStream_t stream) const
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  using LaunchShape = gpu::detail::GrottoGpuLaunchShape<in_bits, frontier_depth, threads_per_block>;
  constexpr size_t kOutputs = size_t{1} << in_bits;
  if (!detail::FitsFullDomain<bool, in_bits>() ||
      kOutputs > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      cws.size() < in_bits + 1 || ys.size() < kOutputs ||
      cws.data() == nullptr || ys.data() == nullptr) return cudaErrorInvalidValue;
  size_t bytes = 0;
  cudaError_t status = cub::DeviceScan::InclusiveScan(nullptr, bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::DeviceScratch scan_storage(stream);
  status = scan_storage.Allocate<unsigned char>(bytes == 0 ? 1 : bytes);
  if (status != cudaSuccess) return status;
  gpu::detail::GrottoLeafBitsKernel<in_bits, LaunchShape::kFrontierDepth,
      LaunchShape::kBlockRootDepth, DpfType>
      <<<LaunchShape::kBlocks, LaunchShape::kThreadsPerBlock, 0, stream>>>(
          b, s0, cws.data(), ys.data(), prg_);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveScan(scan_storage.Get<unsigned char>(), bytes,
      ys.data(), ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  return gpu::detail::ReleaseScratch(status, {&scan_storage});
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
__host__ inline cudaError_t GrottoDcf<in_bits, Prg, In>::EvalAllGpu(
    ParityTreeView pt, cuda::std::span<bool> ys, cudaStream_t stream)
  requires(in_bits < sizeof(size_t) * 8) {
#if defined(__CUDACC__)
  constexpr size_t kOutputs = size_t{1} << in_bits;
  if (!detail::FitsBinaryTree<bool, in_bits>() ||
      kOutputs > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      ys.size() < kOutputs || ys.data() == nullptr || pt.p == nullptr) return cudaErrorInvalidValue;
  const bool *leaves = pt.p + kOutputs - 1;
  size_t bytes = 0;
  cudaError_t status = cub::DeviceScan::InclusiveScan(nullptr, bytes,
      leaves, ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  if (status != cudaSuccess) return status;
  gpu::detail::DeviceScratch scan_storage(stream);
  status = scan_storage.Allocate<unsigned char>(bytes == 0 ? 1 : bytes);
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveScan(scan_storage.Get<unsigned char>(), bytes,
      leaves, ys.data(), gpu::detail::GrottoGpuBoolXor{}, static_cast<int64_t>(kOutputs), stream);
  return gpu::detail::ReleaseScratch(status, {&scan_storage});
#else
  static_assert(sizeof(Prg) == 0, "gpu launch requires a CUDA compiler");
  return cudaErrorNotSupported;
#endif
}

template <typename Scheme>
inline OwnedParityTree<Scheme>::OwnedParityTree(std::unique_ptr<bool[]> p)
    : p_(std::move(p)) {}

template <typename Scheme>
inline typename Scheme::ParityTreeView OwnedParityTree<Scheme>::View() const & {
  return {p_.get()};
}

template <int in_bits, typename Prg, typename In>
  requires(ValidForInputBits<In, in_bits> && Prgable<Prg, 2>)
inline OwnedParityTree<GrottoDcf<in_bits, Prg, In>> GrottoDcf<in_bits, Prg, In>::PreprocessOwned(
    bool b, int4 s0, const Cw cws[], int par_depth) {
  constexpr size_t tree_size = detail::FullBinaryTreeSize<bool, in_bits>();
  auto p = std::make_unique<bool[]>(tree_size);
  Preprocess(b, s0, cws, p.get(), par_depth);
  return OwnedParityTree<GrottoDcf<in_bits, Prg, In>>(std::move(p));
}

}  // namespace fss
