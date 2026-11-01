// SPDX-License-Identifier: Apache-2.0
/**
 * @file binding_gpu_eval.cuh
 * @brief Package-private CUDA selected-query evaluation and proof folding.
 *
 * Device scheme objects and borrowed arrays must remain alive until their stream
 * work completes. Backends must support concurrent device calls. These helpers
 * do not copy host backend state into kernels. Package-private initialization constructs
 * the exact Core scheme on its execution device, including device-local tables.
 * Temporary device storage is allocated and released on the supplied stream for each call.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <cub/device/device_radix_sort.cuh>
#include "binding_gpu_core.cuh"
#include <fss/util.cuh>
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/cuckoo_hash.cuh>

namespace fss::gpu::detail {

enum class ProofGpuKind { kVdpf, kDmpf, kVdmpf };

template <typename Scheme>
struct ProofGpuTraits;

template <int n, typename In, ProofGpuKind kind>
struct ProofGpuShape {
  using Input = In;
  static constexpr int kInputBits = n;
  static constexpr ProofGpuKind kKind = kind;
};

template <int n, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
struct ProofGpuTraits<Vdpf<n, Group, Prg, XorHash, Hash, In>> : ProofGpuShape<n, In, ProofGpuKind::kVdpf> {};
template <int n, int points, int bits, typename Group, typename Prg, typename Prp, typename In, int lambda>
struct ProofGpuTraits<Dmpf<n, points, bits, Group, Prg, Prp, In, lambda>>
    : ProofGpuShape<n, In, ProofGpuKind::kDmpf> {
  using GroupType = Group;
  static constexpr int kBucketBits = bits;
  static constexpr int kBucketCount = Dmpf<n, points, bits, Group, Prg, Prp, In, lambda>::kBucketCount;
};
template <int n, int points, int bits, typename Group, typename Prg, typename XorHash,
    typename Hash, typename Prp, typename In, int lambda>
struct ProofGpuTraits<Vdmpf<n, points, bits, Group, Prg, XorHash, Hash, Prp, In, lambda>>
    : ProofGpuShape<n, In, ProofGpuKind::kVdmpf> {
  using GroupType = Group;
  static constexpr int kBucketBits = bits;
  static constexpr int kBucketCount = Vdmpf<n, points, bits, Group, Prg, XorHash, Hash, Prp, In, lambda>::kBucketCount;
};

using ProofNode = cuda::std::array<int4, 2>;

}  // namespace fss::gpu::detail

namespace fss::gpu {

/**
 * Borrowed device components of a multi-point key.
 *
 * Seeds and correction rows are contiguous in bucket order. Each correction
 * row contains bucket_bits+1 entries, including the final output correction.
 * All allocated buckets, including inactive ones, remain represented.
 * Metadata and storage must describe the same successful generation.
 */
template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kDmpf)
struct MultiComponents {
  int4 sigma;
  int m_rt;
  __uint128_t b_size_rt;
  cuda::std::span<const int4> s0s;
  cuda::std::span<const typename Scheme::Cw> cws;
};

/**
 * Borrowed device components of a verifiable multi-point key.
 *
 * The same row and lifetime rules as MultiComponents apply. Each cs row has
 * four blocks. All components and the scheme must outlive submitted work.
 */
template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kVdmpf)
struct VerifiableMultiComponents {
  int4 sigma;
  int m_rt;
  __uint128_t b_size_rt;
  cuda::std::span<const int4> s0s;
  cuda::std::span<const typename Scheme::Cw> cws;
  cuda::std::span<const int4> cs;
};


/**
 * Fold an ordered nonempty sequence of corrected hashes into Core's Merkle proof.
 *
 * No chain proof is substituted. The canonical adjacent-pair/odd-carry tree
 * is independent of launch geometry. pi and cs each contain four device blocks.
 */
template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kVdpf)
cudaError_t ProveMerkleGpu(Scheme *scheme,
    cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes,
    cuda::std::span<const int4, 4> cs, cuda::std::span<int4, 4> pi,
    cudaStream_t stream, int threads = 256);


/**
 * Evaluate DMPF query shares on the device through the actual inner Core Eval.
 *
 * Uses three PRP candidates and the full Core input/bucket widths. No compact
 * 32-bit index representation or temporary allocation is required. An empty
 * query span is a no-op and does not access the scheme or key components.
 */
template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kDmpf)
cudaError_t BatchEvalGpu(Scheme *scheme, bool b, MultiComponents<Scheme> key,
    std::type_identity_t<cuda::std::span<const typename detail::ProofGpuTraits<Scheme>::Input>> xs,
    cuda::std::span<int4> ys, cudaStream_t stream, int threads = 256);


/**
 * Evaluate VDMPF shares and the canonical per-bucket Merkle proof on the device.
 *
 * The ordered sequence is (query position, candidate index). Stable bucket
 * sorting preserves that order. Empty buckets contribute Hash(cs); nonempty
 * buckets contribute Hash(Hash(cs)||root). All allocated buckets are folded.
 * Launch geometry does not affect proof values. xs may be empty. In that case,
 * only scheme, cs rows, and pi are accessed. Seeds, corrections, sigma, and
 * runtime Cuckoo metadata are unnecessary for the empty-bucket proof.
 */
template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kVdmpf)
cudaError_t BatchEvalMerkleGpu(Scheme *scheme, bool b, VerifiableMultiComponents<Scheme> key,
    std::type_identity_t<cuda::std::span<const typename detail::ProofGpuTraits<Scheme>::Input>> xs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi,
    cudaStream_t stream, int threads = 256);

}  // namespace fss::gpu

// -----------------------------------------------------------------------------

namespace fss::gpu::detail {


template <typename Scheme>
__global__ void HashCorrectedKernel(Scheme *scheme, const cuda::std::array<int4, 4> *corrected,
    ProofNode *nodes, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) nodes[i] = fss::detail::SchemeAccess::GetHash(*scheme).Hash(cuda::std::span<const int4, 4>(corrected[i]));
}


template <typename Scheme>
__global__ void PairNodesKernel(Scheme *scheme, const ProofNode *in, ProofNode *out, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= (count + 1) / 2) return;
  auto left = in[2 * i];
  if (2 * i + 1 == count) out[i] = left;
  else {
    auto right = in[2 * i + 1];
    int4 msg[4] = {left[0], left[1], right[0], right[1]};
    out[i] = fss::detail::SchemeAccess::GetHash(*scheme).Hash(cuda::std::span<const int4, 4>(msg));
  }
}

template <typename Scheme>
inline cudaError_t FoldNodesGpu(Scheme *scheme, ProofNode *&nodes, ProofNode *scratch,
    size_t count, cudaStream_t stream, int threads) {
  while (count > 1) {
    int blocks;
    cudaError_t status = CoreLaunchBlocks((count + 1) / 2, threads, blocks);
    if (status != cudaSuccess) return status;
    PairNodesKernel<<<blocks, threads, 0, stream>>>(scheme, nodes, scratch, count);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    auto *swap = nodes;
    nodes = scratch;
    scratch = swap;
    count = (count + 1) / 2;
  }
  return cudaSuccess;
}

static __global__ void VdpfFinishKernel(const ProofNode *root, const int4 *cs, int4 *pi) {
  pi[0] = root[0][0];
  pi[1] = root[0][1];
  pi[2] = cs[2];
  pi[3] = cs[3];
}

inline bool HasRows(size_t size, size_t rows, size_t width) {
  return rows <= size / width;
}

template <typename Scheme, typename Key>
inline bool ValidMulti(const Key &key) {
  using Traits = ProofGpuTraits<Scheme>;
  constexpr size_t rows = Traits::kBucketCount;
  if (key.m_rt < 1 || key.m_rt > Traits::kBucketCount || key.b_size_rt == 0 ||
      key.b_size_rt > (__uint128_t{1} << Traits::kBucketBits) ||
      key.b_size_rt < (((__uint128_t{3} << Traits::kInputBits) + key.m_rt - 1) / key.m_rt)) return false;
  if (!HasRows(key.s0s.size(), rows, 1) ||
      !HasRows(key.cws.size(), rows, Traits::kBucketBits + 1)) return false;
  if (key.s0s.data() == nullptr || key.cws.data() == nullptr) return false;
  if constexpr (Traits::kKind == ProofGpuKind::kVdmpf) {
    if (!HasRows(key.cs.size(), rows, 4) || key.cs.data() == nullptr) return false;
  }
  return true;
}

template <typename Scheme>
__global__ void DmpfBatchKernel(Scheme *scheme, bool b, MultiComponents<Scheme> key,
    const typename ProofGpuTraits<Scheme>::Input *xs, int4 *ys, size_t count) {
  using Traits = ProofGpuTraits<Scheme>;
  using Group = typename Traits::GroupType;
  size_t omega = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (omega >= count) return;
  auto inner = fss::detail::SchemeAccess::MakeInner(*scheme);
  auto prp = fss::detail::SchemeAccess::GetPrp(*scheme);
  cuckoo_hash::detail::PrpHash<decltype(prp), typename Traits::Input> hasher{prp};
  auto sum = Group::From(int4{0, 0, 0, 0});
  for (int k = 0; k < 3; ++k) {
    auto location = hasher.Locate(key.sigma, xs[omega], k, __uint128_t{1} << Traits::kInputBits, key.b_size_rt);
    int4 value = inner.Eval(b, key.s0s[location.bucket],
        key.cws.data() + static_cast<size_t>(location.bucket) * (Traits::kBucketBits + 1), location.index);
    sum = sum + Group::From(value);
  }
  ys[omega] = sum.Into();
}

template <typename Scheme>
__global__ void VdmpfBatchLeavesKernel(Scheme *scheme, bool b, VerifiableMultiComponents<Scheme> key,
    const typename ProofGpuTraits<Scheme>::Input *xs, int4 *ys, uint32_t *buckets,
    ProofNode *nodes, uint32_t *counts, size_t count) {
  using Traits = ProofGpuTraits<Scheme>;
  using Group = typename Traits::GroupType;
  size_t omega = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (omega >= count) return;
  auto inner = fss::detail::SchemeAccess::MakeInner(*scheme);
  auto prp = fss::detail::SchemeAccess::GetPrp(*scheme);
  cuckoo_hash::detail::PrpHash<decltype(prp), typename Traits::Input> hasher{prp};
  auto sum = Group::From(int4{0, 0, 0, 0});
  for (int k = 0; k < 3; ++k) {
    auto location = hasher.Locate(key.sigma, xs[omega], k, __uint128_t{1} << Traits::kInputBits, key.b_size_rt);
    size_t bucket = static_cast<size_t>(location.bucket);
    int4 value;
    auto corrected = inner.Eval(b, key.s0s[bucket],
        cuda::std::span<const typename Scheme::Cw>(key.cws.data() + bucket * (Traits::kBucketBits + 1), Traits::kBucketBits + 1),
        cuda::std::span<const int4, 4>(key.cs.data() + bucket * 4, 4), location.index, value);
    size_t record = 3 * omega + k;
    buckets[record] = static_cast<uint32_t>(bucket);
    nodes[record] = fss::detail::SchemeAccess::GetHash(*scheme).Hash(cuda::std::span<const int4, 4>(corrected));
    atomicAdd(counts + bucket, uint32_t{1});
    sum = sum + Group::From(value);
  }
  ys[omega] = sum.Into();
}

static __global__ void ClearCountsKernel(uint32_t *counts, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) counts[i] = 0;
}

static __global__ void BucketOffsetsKernel(const uint32_t *counts, uint32_t *offsets, size_t count) {
  uint32_t offset = 0;
  for (size_t i = 0; i < count; ++i) {
    offsets[i] = offset;
    offset += counts[i];
  }
  offsets[count] = offset;
}

template <typename Scheme>
__global__ void BucketPairNodesKernel(Scheme *scheme, const ProofNode *in, ProofNode *out,
    const uint32_t *counts, const uint32_t *offsets, size_t buckets, uint32_t stride, size_t total) {
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
    out[slot] = fss::detail::SchemeAccess::GetHash(*scheme).Hash(cuda::std::span<const int4, 4>(msg));
  }
}

template <typename Scheme>
__global__ void BindBucketRootsKernel(Scheme *scheme, VerifiableMultiComponents<Scheme> key,
    const ProofNode *roots, const uint32_t *counts, const uint32_t *offsets, ProofNode *buckets) {
  constexpr size_t count = ProofGpuTraits<Scheme>::kBucketCount;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto hash = fss::detail::SchemeAccess::GetHash(*scheme);
  auto node = hash.Hash(cuda::std::span<const int4, 4>(key.cs.data() + i * 4, 4));
  if (counts != nullptr && counts[i] != 0) {
    auto root = roots[offsets[i]];
    int4 msg[4] = {node[0], node[1], root[0], root[1]};
    node = hash.Hash(cuda::std::span<const int4, 4>(msg));
  }
  buckets[i] = node;
}

static __global__ void MultiFinishKernel(const ProofNode *root, int4 *pi) {
  pi[0] = root[0][0];
  pi[1] = root[0][1];
  pi[2] = {0, 0, 0, 0};
  pi[3] = {0, 0, 0, 0};
}

}  // namespace fss::gpu::detail

namespace fss::gpu {


template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kVdpf)
inline cudaError_t ProveMerkleGpu(Scheme *scheme,
    cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes,
    cuda::std::span<const int4, 4> cs, cuda::std::span<int4, 4> pi,
    cudaStream_t stream, int threads) {
  detail::DeviceScratch nodes_storage(stream);
  detail::DeviceScratch scratch_storage(stream);

  if (pi_tildes.empty() || pi_tildes.data() == nullptr || cs.data() == nullptr || pi.data() == nullptr || scheme == nullptr)
    return cudaErrorInvalidValue;
  int blocks;
  cudaError_t status = detail::CoreLaunchBlocks(pi_tildes.size(), threads, blocks);
  if (status != cudaSuccess) return status;
  status = nodes_storage.Allocate<detail::ProofNode>(pi_tildes.size());
  if (status != cudaSuccess) return status;
  auto *nodes = nodes_storage.Get<detail::ProofNode>();
  status = scratch_storage.Allocate<detail::ProofNode>(pi_tildes.size());
  if (status != cudaSuccess) return status;
  auto *scratch = scratch_storage.Get<detail::ProofNode>();
  detail::HashCorrectedKernel<<<blocks, threads, 0, stream>>>(scheme, pi_tildes.data(), nodes, pi_tildes.size());
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = detail::FoldNodesGpu(scheme, nodes, scratch, pi_tildes.size(), stream, threads);
  if (status != cudaSuccess) return status;
  detail::VdpfFinishKernel<<<1, 1, 0, stream>>>(nodes, cs.data(), pi.data());
  status = cudaGetLastError();
  return detail::ReleaseScratch(status, {&nodes_storage, &scratch_storage});
}


template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kDmpf)
inline cudaError_t BatchEvalGpu(Scheme *scheme, bool b, MultiComponents<Scheme> key,
    std::type_identity_t<cuda::std::span<const typename detail::ProofGpuTraits<Scheme>::Input>> xs,
    cuda::std::span<int4> ys, cudaStream_t stream, int threads) {
  if (xs.size() != ys.size()) return cudaErrorInvalidValue;
  int blocks;
  cudaError_t status = detail::CoreLaunchBlocks(xs.size(), threads, blocks);
  if (status != cudaSuccess || blocks == 0) return status;
  if (!detail::ValidMulti<Scheme>(key) || scheme == nullptr || xs.data() == nullptr || ys.data() == nullptr)
    return cudaErrorInvalidValue;
  detail::DmpfBatchKernel<<<blocks, threads, 0, stream>>>(scheme, b, key, xs.data(), ys.data(), xs.size());
  return cudaGetLastError();
}


template <typename Scheme>
  requires(detail::ProofGpuTraits<Scheme>::kKind == detail::ProofGpuKind::kVdmpf)
inline cudaError_t BatchEvalMerkleGpu(Scheme *scheme, bool b, VerifiableMultiComponents<Scheme> key,
    std::type_identity_t<cuda::std::span<const typename detail::ProofGpuTraits<Scheme>::Input>> xs,
    cuda::std::span<int4> ys, cuda::std::span<int4, 4> pi,
    cudaStream_t stream, int threads) {
  detail::DeviceScratch bucket_nodes_storage(stream);
  detail::DeviceScratch bucket_scratch_storage(stream);
  detail::DeviceScratch keys_storage(stream);
  detail::DeviceScratch sorted_keys_storage(stream);
  detail::DeviceScratch nodes_storage(stream);
  detail::DeviceScratch sorted_nodes_storage(stream);
  detail::DeviceScratch counts_storage(stream);
  detail::DeviceScratch offsets_storage(stream);
  detail::DeviceScratch sort_scratch_storage(stream);

  using Traits = detail::ProofGpuTraits<Scheme>;
  constexpr size_t buckets = Traits::kBucketCount;
  if (xs.size() != ys.size() || xs.size() > static_cast<size_t>(std::numeric_limits<int>::max() / 3) ||
      scheme == nullptr || pi.data() == nullptr || key.cs.data() == nullptr ||
      !detail::HasRows(key.cs.size(), buckets, 4)) return cudaErrorInvalidValue;
  if (!xs.empty() && !detail::ValidMulti<Scheme>(key)) return cudaErrorInvalidValue;
  int grid;
  cudaError_t status = detail::CoreLaunchBlocks(xs.size(), threads, grid);
  if (status != cudaSuccess) return status;
  if (!xs.empty() && (xs.data() == nullptr || ys.data() == nullptr)) return cudaErrorInvalidValue;
  int bucket_grid;
  status = detail::CoreLaunchBlocks(buckets, threads, bucket_grid);
  if (status != cudaSuccess) return status;
  status = bucket_nodes_storage.Allocate<detail::ProofNode>(buckets);
  if (status != cudaSuccess) return status;
  auto *bucket_nodes = bucket_nodes_storage.Get<detail::ProofNode>();
  status = bucket_scratch_storage.Allocate<detail::ProofNode>(buckets);
  if (status != cudaSuccess) return status;
  auto *bucket_scratch = bucket_scratch_storage.Get<detail::ProofNode>();
  if (xs.empty()) {
    detail::BindBucketRootsKernel<<<bucket_grid, threads, 0, stream>>>(scheme, key, nullptr, nullptr, nullptr, bucket_nodes);
  } else {
    size_t total = xs.size() * 3;
    status = keys_storage.Allocate<uint32_t>(total);
    if (status != cudaSuccess) return status;
    auto *keys = keys_storage.Get<uint32_t>();
    status = sorted_keys_storage.Allocate<uint32_t>(total);
    if (status != cudaSuccess) return status;
    auto *sorted_keys = sorted_keys_storage.Get<uint32_t>();
    status = nodes_storage.Allocate<detail::ProofNode>(total);
    if (status != cudaSuccess) return status;
    auto *nodes = nodes_storage.Get<detail::ProofNode>();
    status = sorted_nodes_storage.Allocate<detail::ProofNode>(total);
    if (status != cudaSuccess) return status;
    auto *sorted_nodes = sorted_nodes_storage.Get<detail::ProofNode>();
    status = counts_storage.Allocate<uint32_t>(buckets);
    if (status != cudaSuccess) return status;
    auto *counts = counts_storage.Get<uint32_t>();
    status = offsets_storage.Allocate<uint32_t>(buckets + 1);
    if (status != cudaSuccess) return status;
    auto *offsets = offsets_storage.Get<uint32_t>();
    detail::ClearCountsKernel<<<bucket_grid, threads, 0, stream>>>(counts, buckets);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    detail::VdmpfBatchLeavesKernel<<<grid, threads, 0, stream>>>(scheme, b, key, xs.data(), ys.data(), keys, nodes, counts, xs.size());
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    size_t bytes = 0;
    status = cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, sorted_keys, nodes, sorted_nodes,
        static_cast<int>(total), 0, 32, stream);
    if (status != cudaSuccess) return status;
    status = sort_scratch_storage.Allocate<unsigned char>(bytes == 0 ? 1 : bytes);
    if (status != cudaSuccess) return status;
    auto *sort_scratch = sort_scratch_storage.Get<unsigned char>();
    status = cub::DeviceRadixSort::SortPairs(sort_scratch, bytes, keys, sorted_keys, nodes, sorted_nodes,
        static_cast<int>(total), 0, 32, stream);
    if (status != cudaSuccess) return status;
    detail::BucketOffsetsKernel<<<1, 1, 0, stream>>>(counts, offsets, buckets);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    auto *roots = sorted_nodes;
    auto *scratch = nodes;
    size_t max_count = total;
    for (uint32_t stride = 1; max_count > 1; stride *= 2) {
      detail::BucketPairNodesKernel<<<grid * 3, threads, 0, stream>>>(
          scheme, roots, scratch, counts, offsets, buckets, stride, total);
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      auto *swap = roots;
      roots = scratch;
      scratch = swap;
      max_count = (max_count + 1) / 2;
    }
    detail::BindBucketRootsKernel<<<bucket_grid, threads, 0, stream>>>(scheme, key, roots, counts, offsets, bucket_nodes);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = detail::FoldNodesGpu(scheme, bucket_nodes, bucket_scratch, buckets, stream, threads);
  if (status != cudaSuccess) return status;
  detail::MultiFinishKernel<<<1, 1, 0, stream>>>(bucket_nodes, pi.data());
  status = cudaGetLastError();
  return detail::ReleaseScratch(status, {&bucket_nodes_storage, &bucket_scratch_storage, &keys_storage, &sorted_keys_storage, &nodes_storage, &sorted_nodes_storage, &counts_storage, &offsets_storage, &sort_scratch_storage});
}

}  // namespace fss::gpu
