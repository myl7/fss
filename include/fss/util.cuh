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

}  // namespace fss::util

// -----------------------------------------------------------------------------

namespace fss::detail {

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

}  // namespace fss::gpu::detail

namespace fss::detail {

// Temporary access for existing CUDA helpers; retire entries with their consumers.
struct SchemeAccess {
  template <typename Scheme, typename Hash>
  static __host__ __device__ void HalfTreeExpandNode(Hash &hash, int4 instance_seed,
      int4 node, int4 correction, int4 &left, int4 &right) {
    Scheme::ExpandNode(hash, instance_seed, node, correction, left, right);
  }

  template <typename Scheme, typename Hash>
  static __host__ __device__ int4 HalfTreePathStep(Hash &hash, int4 instance_seed,
      int4 node, int4 correction, bool take_right) {
    return Scheme::PathStep(hash, instance_seed, node, correction, take_right);
  }

  template <typename Scheme, typename Hash>
  static __host__ __device__ int4 PackedConvertLeaf(Hash &hash, int4 instance_seed,
      int4 node, int4 output_correction) {
    return Scheme::ConvertLeaf(hash, instance_seed, node, output_correction);
  }

  template <typename Scheme, typename Hash, typename Correction>
  static __host__ __device__ void HalfTreeConvertLeafPair(bool party, Hash &hash,
      int4 instance_seed, int4 node, const typename Scheme::Cw &last_correction,
      Correction &output_correction, int4 &left, int4 &right) {
    Scheme::ConvertLeafPair(party, hash, instance_seed, node, last_correction,
        output_correction, left, right);
  }

  template <typename Scheme, typename Prg, typename Group, typename Cw>
  static __host__ __device__ void DcfExpandNode(Prg &prg, int4 state, const Group &value,
      bool party, const Cw &correction, int4 &left, Group &left_value,
      int4 &right, Group &right_value) {
    Scheme::ExpandNode(prg, state, value, party, correction,
        left, left_value, right, right_value);
  }

  template <typename Scheme, typename Prg, typename Group, typename Cw>
  static __host__ __device__ void DcfPathStep(Prg &prg, int4 state, const Group &value,
      bool party, const Cw &correction, bool take_right,
      int4 &next_state, Group &next_value) {
    Scheme::PathStep(prg, state, value, party, correction,
        take_right, next_state, next_value);
  }

  template <typename Scheme, typename Prg>
  static __host__ __device__ void DpfExpandNode(Prg &prg, int4 state, int4 correction,
      bool right_control, int4 &left, int4 &right) {
    Scheme::ExpandNode(prg, state, correction, right_control, left, right);
  }

  template <typename Scheme, typename Prg>
  static __host__ __device__ int4 DpfPathStep(Prg &prg, int4 state, int4 correction,
      bool right_control, bool take_right) {
    return Scheme::PathStep(prg, state, correction, right_control, take_right);
  }

  template <typename Scheme>
  static __host__ __device__ int4 DpfConvertLeaf(bool party, int4 state, int4 correction) {
    return Scheme::ConvertLeaf(party, state, correction);
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
