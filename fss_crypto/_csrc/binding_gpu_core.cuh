// SPDX-License-Identifier: Apache-2.0
/**
 * @file binding_gpu_core.cuh
 * @brief Package-private launches of device-callable Core generation operations.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/span>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/vdpf.cuh>

namespace fss::gpu::detail {

enum class CoreGenKind { kBlock, kPacked, kComparison, kVerifiable };

template <typename Scheme>
struct CoreGpuTraits;

}  // namespace fss::gpu::detail

namespace fss::gpu {

/**
 * Generate one DPF/DCF/HalfTree key's corrections on the device.
 *
 * Calls the actual Core Gen once. Scheme and seed storage must be device-accessible.
 * The configured backend must support device execution. Inputs and outputs retain
 * the same meaning and preconditions as Core Gen(). No memory is allocated.
 * @param scheme Initialized device scheme object.
 * @param cws Device output including the terminal correction.
 * @param s0s Device storage of the two initial seeds, preserved by generation.
 * @param a Secret point or threshold, passed by value.
 * @param b_buf Payload block, passed by value as in Core Gen().
 * @param stream CUDA stream.
 * @return Immediate CUDA launch status.
 */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kBlock)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    int4 b_buf, cudaStream_t stream);

/** Generate packed corrections once, using the Core uint64_t payload. */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kPacked)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    uint64_t beta, cudaStream_t stream);

/** Generate comparison corrections once without a payload argument. */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kComparison)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    cudaStream_t stream);

/**
 * Attempt verifiable point generation once on the device.
 *
 * @param scheme Initialized device Vdpf scheme.
 * @param cws Device correction output including the terminal correction.
 * @param cs Device four-block verification correction output.
 * @param s0s Device initial seeds. They are not modified.
 * @param a Secret point, with the Core domain precondition.
 * @param b_buf Core payload block.
 * @param status Device output for the Core status: zero on success, one on failure.
 * @param stream CUDA stream.
 * @return Immediate CUDA launch status, independently of the asynchronous Core status.
 */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kVerifiable)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    typename detail::CoreGpuTraits<Scheme>::Input a, int4 b_buf, int *status, cudaStream_t stream);

}  // namespace fss::gpu

// -----------------------------------------------------------------------------

namespace fss::gpu::detail {

template <typename In, size_t count, CoreGenKind kind>
struct CoreGpuShape {
  using Input = In;
  static constexpr size_t kCorrectionCount = count;
  static constexpr CoreGenKind kGenKind = kind;
};

template <int n, typename Group, typename Prg, typename In>
struct CoreGpuTraits<Dpf<n, Group, Prg, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kBlock> {};

template <int n, typename Group, typename Prg, typename In, DcfPred pred>
struct CoreGpuTraits<Dcf<n, Group, Prg, In, pred>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kBlock> {};

template <int n, typename Group, typename Hash, typename In>
struct CoreGpuTraits<HalfTreeDpf<n, Group, Hash, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kBlock> {};

template <int n, int bits, typename Hash, typename In>
struct CoreGpuTraits<PackedHalfTreeDpf<n, bits, Hash, In>>
    : CoreGpuShape<In, PackedHalfTreeDpf<n, bits, Hash, In>::kDepth + 1,
          CoreGenKind::kPacked> {};

template <int n, typename Prg, typename In>
struct CoreGpuTraits<GrottoDcf<n, Prg, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kComparison> {};

template <int n, typename Hash, typename In>
struct CoreGpuTraits<HalfTreeGrottoDcf<n, Hash, In>>
    : CoreGpuShape<In, HalfTreeGrottoDcf<n, Hash, In>::kDepth + 1,
          CoreGenKind::kComparison> {};

template <int n, typename Group, typename Prg, typename XorHash, typename Hash,
    typename In>
struct CoreGpuTraits<Vdpf<n, Group, Prg, XorHash, Hash, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kVerifiable> {};

template <typename Scheme, typename Payload>
__global__ void CoreGenKernel(Scheme *scheme, typename Scheme::Cw *cws,
    const int4 *s0s, typename CoreGpuTraits<Scheme>::Input a, Payload payload) {
  scheme->Gen(cws, s0s, a, payload);
}

template <typename Scheme>
__global__ void CoreComparisonGenKernel(Scheme *scheme, typename Scheme::Cw *cws,
    const int4 *s0s, typename CoreGpuTraits<Scheme>::Input a) {
  scheme->Gen(cws, s0s, a);
}

template <typename Scheme>
__global__ void CoreVerifiableGenKernel(Scheme *scheme, typename Scheme::Cw *cws,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    typename CoreGpuTraits<Scheme>::Input a, int4 b_buf, int *status) {
  *status = scheme->Gen(cws, cs, s0s, a, b_buf);
}

inline cudaError_t CoreLaunchBlocks(size_t count, int threads, int &blocks) {
  if (threads < 1 || threads > 1024) return cudaErrorInvalidConfiguration;
  size_t value = count / static_cast<size_t>(threads) + (count % static_cast<size_t>(threads) != 0);
  if (value > static_cast<size_t>(std::numeric_limits<int>::max())) return cudaErrorInvalidConfiguration;
  blocks = static_cast<int>(value);
  return cudaSuccess;
}

}  // namespace fss::gpu::detail

namespace fss::gpu {

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kBlock)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    int4 b_buf, cudaStream_t stream) {
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr) return cudaErrorInvalidValue;
  detail::CoreGenKernel<<<1, 1, 0, stream>>>(scheme, cws.data(), s0s.data(), a, b_buf);
  return cudaGetLastError();
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kPacked)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    uint64_t beta, cudaStream_t stream) {
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr) return cudaErrorInvalidValue;
  detail::CoreGenKernel<<<1, 1, 0, stream>>>(scheme, cws.data(), s0s.data(), a, beta);
  return cudaGetLastError();
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kComparison)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    cudaStream_t stream) {
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr) return cudaErrorInvalidValue;
  detail::CoreComparisonGenKernel<<<1, 1, 0, stream>>>(scheme, cws.data(), s0s.data(), a);
  return cudaGetLastError();
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kVerifiable)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    typename detail::CoreGpuTraits<Scheme>::Input a, int4 b_buf, int *status, cudaStream_t stream) {
  if (scheme == nullptr || cws.data() == nullptr || cs.data() == nullptr || s0s.data() == nullptr || status == nullptr)
    return cudaErrorInvalidValue;
  detail::CoreVerifiableGenKernel<<<1, 1, 0, stream>>>(scheme, cws.data(), cs, s0s, a, b_buf, status);
  return cudaGetLastError();
}

}  // namespace fss::gpu
