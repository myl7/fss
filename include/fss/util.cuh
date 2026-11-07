// SPDX-License-Identifier: Apache-2.0
/**
 * @file util.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cassert>

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fss::util {

/** Return the bitwise XOR of two 128-bit blocks. @ingroup fss_api */
__host__ __device__ inline int4 Xor(int4 lhs, int4 rhs) {
  return {lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w};
}

/**
 * Return the element-wise XOR of two equally sized fixed-length block sequences.
 *
 * The common extent N enforces equal lengths at compile time. Dynamic extents
 * are not supported. The result owns its N blocks and does not allocate memory.
 * @tparam N Number of blocks, known at compile time.
 * @ingroup fss_api
 */
template <size_t N>
  requires(N != cuda::std::dynamic_extent)
__host__ __device__ inline cuda::std::array<int4, N> Xor(
    cuda::std::span<const int4, N> lhs, cuda::std::span<const int4, N> rhs) {
  cuda::std::array<int4, N> out{};
  for (size_t i = 0; i < N; ++i) out[i] = Xor(lhs[i], rhs[i]);
  return out;
}

/** Return val with bit 0 of its fourth word set to bit. @ingroup fss_api */
__host__ __device__ inline int4 SetLsb(int4 val, bool bit) {
  if (bit) val.w |= 1;
  else val.w &= ~1;
  return val;
}

/** Return bit 0 of the fourth word of val. @ingroup fss_api */
__host__ __device__ inline bool GetLsb(int4 val) {
  return (val.w & 1) != 0;
}

/**
 * Check a caller precondition.
 *
 * A Debug build aborts with a diagnostic when ok is false, on both the host
 * and the device. A Release build does not check it, and a violation is
 * undefined behavior.
 * @param ok The condition the caller must satisfy.
 * @param msg The precondition description, naming the function and the
 * violated requirement.
 * @ingroup fss_api
 */
__host__ __device__ inline void RequireArg(bool ok, const char *msg) {
  assert(ok && msg);
}

}  // namespace fss::util

// -----------------------------------------------------------------------------

namespace fss::detail {

// Check subtraction before adding one, including a 128-bit full-domain range.
template <int in_bits, typename In>
__host__ inline bool RangeOutputCount(In first, In last, size_t capacity, size_t &count) {
  if (first > last) return false;
  if constexpr (in_bits < sizeof(In) * CHAR_BIT) {
    if ((last >> in_bits) != 0) return false;
  }
  In difference = last - first;
  constexpr size_t kMaxCount = std::numeric_limits<size_t>::max() / sizeof(int4);
  if (difference >= kMaxCount) return false;
  count = static_cast<size_t>(difference) + 1;
  return count <= capacity;
}


// Subdivide [first, last] into aligned chunks and compute the launch geometry
// shared by the range GPU launchers: per-chunk frontier, thread width, block
// root depth, and block count. frontier_cap reserves trailing levels that the
// caller's kernel resolves outside the block frontier (HalfTree uses one).
// launch(chunk_first, depth, offset, frontier, thread_bits, root, blocks)
// starts each chunk's kernels; its return value propagates.
template <int in_bits, int frontier_depth, int threads_per_block, int frontier_cap = 0,
    typename In, typename LaunchChunk>
__host__ inline cudaError_t SplitRangeChunks(In first, In last, size_t ys_count,
    LaunchChunk &&launch) {
  static_assert(frontier_depth >= 0, "gpu frontier depth must be nonnegative");
  static_assert(threads_per_block >= 1 && threads_per_block <= 1024 &&
      (threads_per_block & (threads_per_block - 1)) == 0,
      "gpu block size must be a power of two in [1,1024]");
  size_t count;
  if (!RangeOutputCount<in_bits>(first, last, ys_count, count)) return cudaErrorInvalidValue;
  constexpr int kThreadBits = [] {
    int bits = 0;
    for (int threads = threads_per_block; threads > 1; threads >>= 1) ++bits;
    return bits;
  }();
  constexpr int kMaxDepth = [] {
    int depth = 30 + kThreadBits + frontier_cap;
    if (depth >= sizeof(size_t) * CHAR_BIT) depth = sizeof(size_t) * CHAR_BIT - 1;
    if (frontier_depth < depth && depth - frontier_depth > 8) depth = frontier_depth + 8;
    return depth;
  }();
  return ForEachAlignedRange<in_bits>(first, last, kMaxDepth,
      [&](In chunk_first, int depth, size_t offset, size_t) {
        int frontier = frontier_depth < depth - frontier_cap ? frontier_depth : depth - frontier_cap;
        if (frontier < 0) frontier = 0;
        int thread_bits = kThreadBits < frontier ? kThreadBits : frontier;
        int root = frontier - thread_bits;
        size_t blocks = size_t{1} << root;
        if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
          return cudaErrorInvalidConfiguration;
        return launch(chunk_first, depth, offset, frontier, thread_bits, root, blocks);
      });
}

// Visit aligned power-of-two chunks in input order without shifting by In's width.
// A caller may cap depth to split chunks further for its kernel's fixed limits.
template <int in_bits, typename In, typename Callback>
__host__ inline cudaError_t ForEachAlignedRange(
    In first, In last, int max_depth, Callback callback) {
  size_t offset = 0;
  for (;;) {
    int depth = 0;
    while (depth < max_depth && depth < in_bits &&
        depth < sizeof(In) * CHAR_BIT) {
      In half_mask = (In{1} << depth) - 1;
      In mask = half_mask * 2 + 1;
      if ((first & mask) != 0 || last - first < mask) break;
      ++depth;
    }
    size_t count = size_t{1} << depth;
    cudaError_t error = callback(first, depth, offset, count);
    if (error != cudaSuccess) return error;
    if (last - first == static_cast<In>(count - 1)) return cudaSuccess;
    first += static_cast<In>(count);
    offset += count;
  }
}

// Resolve a tree split depth without shifting beyond a machine word.
inline int ResolveParDepth(int par_depth, int useful_depth = sizeof(size_t) * CHAR_BIT - 1) {
  assert(par_depth >= 0);
  return par_depth < useful_depth ? par_depth : useful_depth;
}

inline size_t ResolveParPartitions(int par_depth, size_t count) {
  assert(par_depth >= 0);
  if (count == 0) return 0;
  int depth = ResolveParDepth(par_depth);
  size_t partitions = 1;
  while (depth-- > 0 && partitions < count) {
    if (partitions > count / 2) return count;
    partitions *= 2;
  }
  return partitions;
}

inline int ResolveParThreads(int par_depth, size_t count) {
  size_t partitions = ResolveParPartitions(par_depth, count);
#ifdef _OPENMP
  int threads = omp_get_max_threads();
#else
  int threads = 1;
#endif
  return partitions < static_cast<size_t>(threads) ? static_cast<int>(partitions) : threads;
}

// Each scheduled partition owns a contiguous range. A depth of zero never
// enters an OpenMP region, including when called from a shallow fallback path.
template <typename Function>
inline void ParallelFor(int par_depth, size_t count, Function function) {
  size_t partitions = ResolveParPartitions(par_depth, count);
  if (partitions <= 1) {
    for (size_t i = 0; i < count; ++i) function(i);
    return;
  }
  int threads = ResolveParThreads(par_depth, count);
#pragma omp parallel for num_threads(threads)
  for (size_t partition = 0; partition < partitions; ++partition) {
    size_t begin = partition * (count / partitions) + (partition < count % partitions ? partition : count % partitions);
    size_t end = begin + count / partitions + (partition < count % partitions);
    for (size_t i = begin; i < end; ++i) function(i);
  }
}

template <typename In>
__host__ __device__ inline int4 Pack(In val) {
  int4 buf = {0, 0, 0, 0};
  if constexpr (sizeof(In) <= 4) {
    buf.x = static_cast<int>(val);
  } else if constexpr (sizeof(In) <= 8) {
    auto v = static_cast<uint64_t>(val);
    buf.x = static_cast<int>(v & 0xFFFFFFFF);
    buf.y = static_cast<int>((v >> 32) & 0xFFFFFFFF);
  } else {
    auto v = static_cast<__uint128_t>(val);
    buf.x = static_cast<int>(v & 0xFFFFFFFF);
    buf.y = static_cast<int>((v >> 32) & 0xFFFFFFFF);
    buf.z = static_cast<int>((v >> 64) & 0xFFFFFFFF);
    buf.w = static_cast<int>((v >> 96) & 0xFFFFFFFF);
  }
  return buf;
}

}  // namespace fss::detail

namespace fss::detail {

template <typename T>
__host__ __device__ constexpr bool FitsElementCount(size_t count) {
  return count <= size_t(-1) / sizeof(T);
}

template <typename T, int bits>
__host__ __device__ constexpr bool FitsFullDomain() {
  if constexpr (bits < 0 || bits >= sizeof(size_t) * 8) {
    return false;
  } else {
    return FitsElementCount<T>(size_t{1} << bits);
  }
}

template <typename T, int bits>
__host__ __device__ constexpr size_t FullDomainSize() {
  static_assert(FitsFullDomain<T, bits>(),
      "full-domain element count and byte size must fit in size_t");
  if constexpr (FitsFullDomain<T, bits>()) return size_t{1} << bits;
  return 0;
}

template <typename T, int bits>
__host__ __device__ constexpr bool FitsBinaryTree() {
  if constexpr (!FitsFullDomain<T, bits>()) {
    return false;
  } else {
    constexpr size_t leaves = size_t{1} << bits;
    constexpr size_t max_count = size_t(-1) / sizeof(T);
    return leaves - 1 <= max_count - leaves;
  }
}

template <typename T, int bits>
__host__ __device__ constexpr size_t FullBinaryTreeSize() {
  static_assert(FitsBinaryTree<T, bits>(),
      "full binary-tree count and byte size must fit in size_t");
  if constexpr (FitsBinaryTree<T, bits>()) {
    constexpr size_t leaves = size_t{1} << bits;
    return leaves + (leaves - 1);
  }
  return 0;
}

}  // namespace fss::detail

namespace fss::gpu::detail {

// Per-call temporary storage. Release is queued after its uses on the same stream.
class DeviceScratch {
public:
  explicit DeviceScratch(cudaStream_t stream) : stream_(stream) {}
  DeviceScratch(const DeviceScratch &) = delete;
  DeviceScratch &operator=(const DeviceScratch &) = delete;
  ~DeviceScratch() { Release(); }

  template <typename T>
  cudaError_t Allocate(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    void *allocation = nullptr;
    cudaError_t status = cudaMallocAsync(&allocation, count * sizeof(T), stream_);
    if (status == cudaSuccess) data_ = allocation;
    return status;
  }

  template <typename T>
  T *Get() const { return static_cast<T *>(data_); }

  cudaError_t Release() noexcept {
    if (data_ == nullptr) return cudaSuccess;
    void *allocation = data_;
    data_ = nullptr;
    return cudaFreeAsync(allocation, stream_);
  }

private:
  cudaStream_t stream_;
  void *data_ = nullptr;
};

inline cudaError_t ReleaseScratch(cudaError_t status,
    std::initializer_list<DeviceScratch *> buffers) {
  for (auto *buffer : buffers) {
    cudaError_t release_status = buffer->Release();
    if (status == cudaSuccess) status = release_status;
  }
  return status;
}

inline void CheckCudaStatus(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("cuda operation failed: ") + cudaGetErrorString(status));
}

__device__ __host__ inline int4 Zero4() {
  return {0, 0, 0, 0};
}

template <typename T>
__device__ __host__ inline bool GetBit(T x, int i) {
  return ((x >> i) & 1) != 0;
}

// Constrains a GPU query element type to an unsigned integer wide enough for
// the scheme's input domain.
template <typename Query, int bits>
concept QueryFor = !std::is_same_v<Query, bool> &&
    (std::is_unsigned_v<Query> || std::is_same_v<Query, __uint128_t>) && sizeof(Query) * 8 >= bits;

}  // namespace fss::gpu::detail

namespace fss::detail {

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

// Temporary access for existing CUDA helpers; retire entries with their consumers.
struct SchemeAccess {
  template <typename Scheme, typename Hash>
  static __host__ __device__ void HalfTreeExpandNode(Hash &hash, int4 instance_seed,
      int4 node, int4 cw, int4 &left, int4 &right) {
    Scheme::ExpandNode(hash, instance_seed, node, cw, left, right);
  }

  template <typename Scheme, typename Hash>
  static __host__ __device__ int4 HalfTreePathStep(Hash &hash, int4 instance_seed,
      int4 node, int4 cw, bool take_right) {
    return Scheme::PathStep(hash, instance_seed, node, cw, take_right);
  }

  template <typename Scheme, typename Hash>
  static __host__ __device__ int4 PackedConvertLeaf(Hash &hash, int4 instance_seed,
      int4 node, int4 cw) {
    return Scheme::ConvertLeaf(hash, instance_seed, node, cw);
  }

  template <typename Scheme, typename Hash, typename Cw>
  static __host__ __device__ void HalfTreeConvertLeafPair(bool party, Hash &hash,
      int4 instance_seed, int4 node, const typename Scheme::Cw &last_cw,
      Cw &output_cw, int4 &left, int4 &right) {
    Scheme::ConvertLeafPair(party, hash, instance_seed, node, last_cw,
        output_cw, left, right);
  }

  template <typename Scheme, typename Prg, typename Group, typename Cw>
  static __host__ __device__ void DcfExpandNode(Prg &prg, int4 state, const Group &value,
      bool party, const Cw &cw, int4 &left, Group &left_value,
      int4 &right, Group &right_value) {
    Scheme::ExpandNode(prg, state, value, party, cw,
        left, left_value, right, right_value);
  }

  template <typename Scheme, typename Prg, typename Group, typename Cw>
  static __host__ __device__ void DcfPathStep(Prg &prg, int4 state, const Group &value,
      bool party, const Cw &cw, bool take_right,
      int4 &next_state, Group &next_value) {
    Scheme::PathStep(prg, state, value, party, cw,
        take_right, next_state, next_value);
  }

  template <typename Scheme, typename Prg>
  static __host__ __device__ void DpfExpandNode(Prg &prg, int4 state, int4 cw,
      bool right_control, int4 &left, int4 &right) {
    Scheme::ExpandNode(prg, state, cw, right_control, left, right);
  }

  template <typename Scheme, typename Prg>
  static __host__ __device__ int4 DpfPathStep(Prg &prg, int4 state, int4 cw,
      bool right_control, bool take_right) {
    return Scheme::PathStep(prg, state, cw, right_control, take_right);
  }

  template <typename Scheme>
  static __host__ __device__ int4 DpfConvertLeaf(bool party, int4 state, int4 cw) {
    return Scheme::ConvertLeaf(party, state, cw);
  }

  template <typename Scheme>
  static __host__ __device__ auto &GetHash(Scheme &scheme) { return scheme.hash_; }

  template <typename Scheme>
  static __host__ __device__ const auto &GetPrg(const Scheme &scheme) { return scheme.prg_; }

  template <typename Scheme>
  static __host__ __device__ const auto &GetHash(const Scheme &scheme) { return scheme.hash_; }

  template <typename Scheme>
  static __host__ __device__ const auto &GetXorHash(const Scheme &scheme) { return scheme.xor_hash_; }

  template <typename Scheme>
  static __host__ __device__ const auto &GetPrp(const Scheme &scheme) { return scheme.prp_; }

  template <typename Scheme>
  static __host__ __device__ auto GetInstanceSeed(const Scheme &scheme) { return scheme.instance_seed_; }

  // Cw-layout correction-word entry points for library owners that keep the Cw
  // storage contract (DMPF/VDMPF bucket keys). Public callers use the component
  // strided APIs; these reuse the schemes' private shared implementations.
  template <typename Scheme>
  static __host__ __device__ void DpfGenCw(Scheme &scheme, typename Scheme::Cw *cws,
      const int4 *s0s, auto a, int4 b_buf) {
    scheme.Generate(typename Scheme::AosWriter{cws}, s0s, a, b_buf);
  }

  template <typename Scheme>
  static __host__ __device__ int4 DpfEvalCw(Scheme &scheme, bool b, int4 s0,
      const typename Scheme::Cw *cws, auto x) {
    return scheme.EvalCws(b, s0, cws, x);
  }

  template <typename Scheme>
  static void DpfEvalAllCw(Scheme &scheme, bool b, int4 s0,
      const typename Scheme::Cw *cws, int4 *ys, int par_depth) {
    scheme.EvalAllCws(b, s0, cws, ys, par_depth);
  }

  template <int frontier_depth = 16, int threads_per_block = 256, typename Scheme>
  static __host__ cudaError_t DpfEvalAllGpuBatchCw(Scheme &scheme, bool b,
      cuda::std::span<const int4> s0s, cuda::std::span<const typename Scheme::Cw> cws,
      cuda::std::span<int4> ys, cudaStream_t stream) {
    return scheme.template EvalAllGpuBatchCw<frontier_depth, threads_per_block>(
        b, s0s, cws, ys, stream);
  }

  template <typename Scheme>
  static __host__ __device__ int VdpfGenCw(Scheme &scheme, typename Scheme::Cw *cws,
      cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s, auto a, int4 b_buf) {
    return scheme.Generate(typename Scheme::AosWriter{cws}, cs, s0s.data(), a, b_buf);
  }

  template <typename Scheme>
  static __host__ __device__ auto VdpfEvalCw(Scheme &scheme, bool b, int4 s0,
      cuda::std::span<const typename Scheme::Cw> cws, cuda::std::span<const int4, 4> cs,
      auto x, int4 &y) {
    return scheme.EvalCws(b, s0, cws.data(), cs, x, y);
  }

  template <typename Scheme>
  static __host__ __device__ auto MakeInner(Scheme &scheme) {
    if constexpr (requires { typename Scheme::InnerVdpf; }) {
      return typename Scheme::InnerVdpf{scheme.prg_, scheme.xor_hash_, scheme.hash_};
    } else {
      return typename Scheme::InnerDpf{scheme.prg_};
    }
  }

  template <typename Scheme, typename Block, typename In>
  static __host__ __device__ bool HalfTreeGrottoLeafParity(Block block, In x) {
    return Scheme::LeafParity(block, x);
  }
};

}  // namespace fss::detail
