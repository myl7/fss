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
#include "binding_launch_descriptor.cuh"

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

template <typename Descriptor, typename Input, typename Payload>
__global__ void CoreGenKernel(void *scheme_storage, void *cws_storage,
    const int4 *s0s, Input a, Payload payload) {
  using Scheme = typename Descriptor::Scheme;
  Scheme *scheme = static_cast<Scheme *>(scheme_storage);
  typename Scheme::Cw *cws = static_cast<typename Scheme::Cw *>(cws_storage);
  scheme->Gen(cws, s0s, a, payload);
}

// Host-only launch helpers. The DPF, DCF, HalfTreeDpf, and Vdpf overloads
// build the descriptor directly from the scheme's template parameters, so a
// wide Group argument never reaches the generated host stub.
template <typename Scheme, typename Payload>
inline cudaError_t LaunchCoreGen(Scheme *scheme, typename Scheme::Cw *cws,
    const int4 *s0s, typename CoreGpuTraits<Scheme>::Input a, Payload payload,
    cudaStream_t stream) {
  CoreGenKernel<CoreLaunchIdentityDescriptor<Scheme>, typename CoreGpuTraits<Scheme>::Input, Payload>
      <<<1, 1, 0, stream>>>(scheme, cws, s0s, a, payload);
  return cudaGetLastError();
}

template <int n, typename Group, typename Prg, typename In, typename Payload>
inline cudaError_t LaunchCoreGen(Dpf<n, Group, Prg, In> *scheme,
    typename Dpf<n, Group, Prg, In>::Cw *cws, const int4 *s0s, In a,
    Payload payload, cudaStream_t stream) {
  using Descriptor = DpfGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type, Prg, In>;
  CoreGenKernel<Descriptor, In, Payload><<<1, 1, 0, stream>>>(scheme, cws, s0s, a, payload);
  return cudaGetLastError();
}

template <int n, typename Group, typename Prg, typename In, DcfPred pred, typename Payload>
inline cudaError_t LaunchCoreGen(Dcf<n, Group, Prg, In, pred> *scheme,
    typename Dcf<n, Group, Prg, In, pred>::Cw *cws, const int4 *s0s, In a,
    Payload payload, cudaStream_t stream) {
  using Descriptor = DcfGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type,
      Prg, In, pred>;
  CoreGenKernel<Descriptor, In, Payload><<<1, 1, 0, stream>>>(scheme, cws, s0s, a, payload);
  return cudaGetLastError();
}

template <int n, typename Group, typename Hash, typename In, typename Payload>
inline cudaError_t LaunchCoreGen(HalfTreeDpf<n, Group, Hash, In> *scheme,
    typename HalfTreeDpf<n, Group, Hash, In>::Cw *cws, const int4 *s0s, In a,
    Payload payload, cudaStream_t stream) {
  using Descriptor = HalfTreeGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type,
      Hash, In>;
  CoreGenKernel<Descriptor, In, Payload><<<1, 1, 0, stream>>>(scheme, cws, s0s, a, payload);
  return cudaGetLastError();
}

template <typename Scheme>
__global__ void CoreComparisonGenKernel(Scheme *scheme, typename Scheme::Cw *cws,
    const int4 *s0s, typename CoreGpuTraits<Scheme>::Input a) {
  scheme->Gen(cws, s0s, a);
}

template <typename Descriptor, typename Input>
__global__ void CoreVerifiableGenKernel(void *scheme_storage, void *cws_storage,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    Input a, int4 b_buf, int *status) {
  using Scheme = typename Descriptor::Scheme;
  Scheme *scheme = static_cast<Scheme *>(scheme_storage);
  typename Scheme::Cw *cws = static_cast<typename Scheme::Cw *>(cws_storage);
  *status = scheme->Gen(cws, cs, s0s, a, b_buf);
}

template <typename Scheme>
inline cudaError_t LaunchCoreGen(Scheme *scheme, typename Scheme::Cw *cws,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    typename CoreGpuTraits<Scheme>::Input a, int4 b_buf, int *status, cudaStream_t stream) {
  using Descriptor = CoreLaunchIdentityDescriptor<Scheme>;
  CoreVerifiableGenKernel<Descriptor, typename CoreGpuTraits<Scheme>::Input>
      <<<1, 1, 0, stream>>>(scheme, cws, cs, s0s, a, b_buf, status);
  return cudaGetLastError();
}

template <int n, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
inline cudaError_t LaunchCoreGen(Vdpf<n, Group, Prg, XorHash, Hash, In> *scheme,
    typename Vdpf<n, Group, Prg, XorHash, Hash, In>::Cw *cws,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s, In a,
    int4 b_buf, int *status, cudaStream_t stream) {
  using Descriptor = VdpfGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type,
      Prg, XorHash, Hash, In>;
  CoreVerifiableGenKernel<Descriptor, In><<<1, 1, 0, stream>>>(
      scheme, cws, cs, s0s, a, b_buf, status);
  return cudaGetLastError();
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
  return detail::LaunchCoreGen(scheme, cws.data(), s0s.data(), a, b_buf, stream);
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kPacked)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<typename Scheme::Cw,
        detail::CoreGpuTraits<Scheme>::kCorrectionCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    uint64_t beta, cudaStream_t stream) {
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr) return cudaErrorInvalidValue;
  return detail::LaunchCoreGen(scheme, cws.data(), s0s.data(), a, beta, stream);
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
  return detail::LaunchCoreGen(scheme, cws.data(), cs, s0s, a, b_buf, status, stream);
}

}  // namespace fss::gpu
