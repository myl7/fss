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
 * Generate one DPF/DCF/HalfTree key's split correction components on the device.
 *
 * Calls the actual Core Gen once. Scheme and seed storage must be device-accessible.
 * The configured backend must support device execution. Inputs and outputs retain
 * the same meaning and preconditions as Core Gen(). No memory is allocated.
 * The caller's correction arrays and control slot are device storage; the kernel
 * writes the packed control value through tr.
 * @param scheme Initialized device scheme object.
 * @param cws Device seed components including the terminal correction word.
 * @param vws Device value components (DCF only).
 * @param tr Device slot receiving the packed control value, or null when the
 * scheme carries no control.
 * @param s0s Device storage of the two initial seeds, preserved by generation.
 * @param a Secret point or threshold, passed by value.
 * @param b_buf Payload block, passed by value as in Core Gen().
 * @param stream CUDA stream.
 * @return Immediate CUDA launch status.
 */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kBlock)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    std::conditional_t<detail::CoreGpuTraits<Scheme>::kHasValues,
        cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>, std::nullptr_t> vws,
    typename detail::CoreGpuTraits<Scheme>::Input *tr,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    int4 b_buf, cudaStream_t stream);

/** Generate packed correction words once, using the Core uint64_t payload. */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kPacked)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    uint64_t beta, cudaStream_t stream);

/** Generate comparison correction components once without a payload argument. */
template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kComparison)
cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    typename detail::CoreGpuTraits<Scheme>::Input *tr,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    cudaStream_t stream);

/**
 * Attempt verifiable point generation once on the device.
 *
 * @param scheme Initialized device Vdpf scheme.
 * @param cws Device seed components including the terminal correction word.
 * @param tr Device slot receiving the packed control value.
 * @param cs Device four-block correction seed output.
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
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    typename detail::CoreGpuTraits<Scheme>::Input *tr,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    typename detail::CoreGpuTraits<Scheme>::Input a, int4 b_buf, int *status, cudaStream_t stream);

}  // namespace fss::gpu

// -----------------------------------------------------------------------------

namespace fss::gpu::detail {

template <typename In, size_t count, CoreGenKind kind, bool has_control, bool has_values = false>
struct CoreGpuShape {
  using Input = In;
  static constexpr size_t kCwCount = count;
  static constexpr CoreGenKind kGenKind = kind;
  static constexpr bool kHasControl = has_control;
  static constexpr bool kHasValues = has_values;
};

template <int n, typename Group, typename Prg, typename In>
struct CoreGpuTraits<Dpf<n, Group, Prg, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kBlock, true> {};

template <int n, typename Group, typename Prg, typename In, DcfPred pred>
struct CoreGpuTraits<Dcf<n, Group, Prg, In, pred>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kBlock, true, true> {};

template <int n, typename Group, typename Hash, typename In>
struct CoreGpuTraits<HalfTreeDpf<n, Group, Hash, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kBlock, true> {};

template <int n, int bits, typename Hash, typename In>
struct CoreGpuTraits<PackedHalfTreeDpf<n, bits, Hash, In>>
    : CoreGpuShape<In, PackedHalfTreeDpf<n, bits, Hash, In>::kDepth + 1,
          CoreGenKind::kPacked, false> {};

template <int n, typename Prg, typename In>
struct CoreGpuTraits<GrottoDcf<n, Prg, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kComparison, true> {};

template <int n, typename Hash, typename In>
struct CoreGpuTraits<HalfTreeGrottoDcf<n, Hash, In>>
    : CoreGpuShape<In, HalfTreeGrottoDcf<n, Hash, In>::kDepth + 1,
          CoreGenKind::kComparison, false> {};

template <int n, typename Group, typename Prg, typename XorHash, typename Hash,
    typename In>
struct CoreGpuTraits<Vdpf<n, Group, Prg, XorHash, Hash, In>>
    : CoreGpuShape<In, n + 1, CoreGenKind::kVerifiable, true> {};

template <typename Descriptor, typename Input, typename Payload>
__global__ void CoreGenKernel(void *scheme_storage, void *cws_storage, void *vws_storage,
    Input *tr, const int4 *s0s, Input a, Payload payload) {
  using Scheme = typename Descriptor::Scheme;
  using Traits = fss::gpu::detail::CoreGpuTraits<Scheme>;
  Scheme *scheme = static_cast<Scheme *>(scheme_storage);
  int4 *cws = static_cast<int4 *>(cws_storage);
  constexpr size_t count = Traits::kCwCount;
  if constexpr (Traits::kHasValues) {
    int4 *vws = static_cast<int4 *>(vws_storage);
    scheme->Gen(cuda::std::span<int4>(cws, count), cuda::std::span<int4>(vws, count), *tr,
        cuda::std::span<const int4, 2>(s0s, 2), a, payload);
  } else if constexpr (Traits::kHasControl) {
    scheme->Gen(cuda::std::span<int4>(cws, count), *tr,
        cuda::std::span<const int4, 2>(s0s, 2), a, payload);
  } else {
    scheme->Gen(cuda::std::span<int4>(cws, count),
        cuda::std::span<const int4, 2>(s0s, 2), a, payload);
  }
}

// Host-only launch helpers. The DPF, DCF, HalfTreeDpf, and Vdpf overloads
// build the descriptor directly from the scheme's template parameters, so a
// wide Group argument never reaches the generated host stub.
template <typename Scheme, typename Payload>
inline cudaError_t LaunchCoreGen(Scheme *scheme, int4 *cws, int4 *vws,
    typename CoreGpuTraits<Scheme>::Input *tr, const int4 *s0s,
    typename CoreGpuTraits<Scheme>::Input a, Payload payload, cudaStream_t stream) {
  CoreGenKernel<CoreLaunchIdentityDescriptor<Scheme>, typename CoreGpuTraits<Scheme>::Input, Payload>
      <<<1, 1, 0, stream>>>(scheme, cws, vws, tr, s0s, a, payload);
  return cudaGetLastError();
}

template <int n, typename Group, typename Prg, typename In, typename Payload>
inline cudaError_t LaunchCoreGen(Dpf<n, Group, Prg, In> *scheme,
    int4 *cws, int4 *vws, In *tr, const int4 *s0s, In a,
    Payload payload, cudaStream_t stream) {
  using Descriptor = DpfGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type, Prg, In>;
  CoreGenKernel<Descriptor, In, Payload><<<1, 1, 0, stream>>>(scheme, cws, vws, tr, s0s, a, payload);
  return cudaGetLastError();
}

template <int n, typename Group, typename Prg, typename In, DcfPred pred, typename Payload>
inline cudaError_t LaunchCoreGen(Dcf<n, Group, Prg, In, pred> *scheme,
    int4 *cws, int4 *vws, In *tr, const int4 *s0s, In a,
    Payload payload, cudaStream_t stream) {
  using Descriptor = DcfGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type,
      Prg, In, pred>;
  CoreGenKernel<Descriptor, In, Payload><<<1, 1, 0, stream>>>(scheme, cws, vws, tr, s0s, a, payload);
  return cudaGetLastError();
}

template <int n, typename Group, typename Hash, typename In, typename Payload>
inline cudaError_t LaunchCoreGen(HalfTreeDpf<n, Group, Hash, In> *scheme,
    int4 *cws, int4 *vws, In *tr, const int4 *s0s, In a,
    Payload payload, cudaStream_t stream) {
  using Descriptor = HalfTreeGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type,
      Hash, In>;
  CoreGenKernel<Descriptor, In, Payload><<<1, 1, 0, stream>>>(scheme, cws, vws, tr, s0s, a, payload);
  return cudaGetLastError();
}

template <typename Descriptor, typename Input>
__global__ void CoreComparisonGenKernel(void *scheme_storage, void *cws_storage,
    Input *tr, const int4 *s0s, Input a) {
  using Scheme = typename Descriptor::Scheme;
  using Traits = fss::gpu::detail::CoreGpuTraits<Scheme>;
  Scheme *scheme = static_cast<Scheme *>(scheme_storage);
  int4 *cws = static_cast<int4 *>(cws_storage);
  constexpr size_t count = Traits::kCwCount;
  if constexpr (Traits::kHasControl)
    scheme->Gen(cuda::std::span<int4>(cws, count), *tr, cuda::std::span<const int4, 2>(s0s, 2), a);
  else
    scheme->Gen(cuda::std::span<int4>(cws, count), cuda::std::span<const int4, 2>(s0s, 2), a);
}

template <typename Scheme>
inline cudaError_t LaunchComparisonGen(Scheme *scheme, int4 *cws,
    typename CoreGpuTraits<Scheme>::Input *tr, const int4 *s0s,
    typename CoreGpuTraits<Scheme>::Input a, cudaStream_t stream) {
  CoreComparisonGenKernel<CoreLaunchIdentityDescriptor<Scheme>,
      typename CoreGpuTraits<Scheme>::Input>
      <<<1, 1, 0, stream>>>(scheme, cws, tr, s0s, a);
  return cudaGetLastError();
}

template <typename Descriptor, typename Input>
__global__ void CoreVerifiableGenKernel(void *scheme_storage, void *cws_storage,
    Input *tr, cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    Input a, int4 b_buf, int *status) {
  using Scheme = typename Descriptor::Scheme;
  using Traits = fss::gpu::detail::CoreGpuTraits<Scheme>;
  Scheme *scheme = static_cast<Scheme *>(scheme_storage);
  int4 *cws = static_cast<int4 *>(cws_storage);
  constexpr size_t count = Traits::kCwCount;
  *status = scheme->Gen(cuda::std::span<int4>(cws, count), *tr, cs, s0s, a, b_buf);
}

template <int n, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
inline cudaError_t LaunchCoreGen(Vdpf<n, Group, Prg, XorHash, Hash, In> *scheme,
    int4 *cws, In *tr,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s, In a,
    int4 b_buf, int *status, cudaStream_t stream) {
  using Descriptor = VdpfGpuKernelDescriptor<n, typename fss::detail::GpuGroupTag<Group>::Type,
      Prg, XorHash, Hash, In>;
  CoreVerifiableGenKernel<Descriptor, In><<<1, 1, 0, stream>>>(
      scheme, cws, tr, cs, s0s, a, b_buf, status);
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
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    std::conditional_t<detail::CoreGpuTraits<Scheme>::kHasValues,
        cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>, std::nullptr_t> vws,
    typename detail::CoreGpuTraits<Scheme>::Input *tr,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    int4 b_buf, cudaStream_t stream) {
  constexpr bool kControl = detail::CoreGpuTraits<Scheme>::kHasControl;
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr ||
      (kControl && tr == nullptr)) return cudaErrorInvalidValue;
  int4 *vws_data = nullptr;
  if constexpr (detail::CoreGpuTraits<Scheme>::kHasValues) {
    if (vws.data() == nullptr) return cudaErrorInvalidValue;
    vws_data = vws.data();
  }
  return detail::LaunchCoreGen(scheme, cws.data(), vws_data, tr,
      s0s.data(), a, b_buf, stream);
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kPacked)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    uint64_t beta, cudaStream_t stream) {
  static_assert(!detail::CoreGpuTraits<Scheme>::kHasControl, "packed generation carries no control");
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr) return cudaErrorInvalidValue;
  return detail::LaunchCoreGen(scheme, cws.data(), nullptr, nullptr, s0s.data(), a, beta, stream);
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kComparison)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    typename detail::CoreGpuTraits<Scheme>::Input *tr,
    cuda::std::span<const int4, 2> s0s, typename detail::CoreGpuTraits<Scheme>::Input a,
    cudaStream_t stream) {
  constexpr bool kControl = detail::CoreGpuTraits<Scheme>::kHasControl;
  if (scheme == nullptr || cws.data() == nullptr || s0s.data() == nullptr ||
      (kControl && tr == nullptr)) return cudaErrorInvalidValue;
  detail::CoreComparisonGenKernel<detail::CoreLaunchIdentityDescriptor<Scheme>,
      typename detail::CoreGpuTraits<Scheme>::Input>
      <<<1, 1, 0, stream>>>(scheme, cws.data(), tr, s0s.data(), a);
  return cudaGetLastError();
}

template <typename Scheme>
  requires(detail::CoreGpuTraits<Scheme>::kGenKind == detail::CoreGenKind::kVerifiable)
inline cudaError_t GenGpu(Scheme *scheme,
    std::type_identity_t<cuda::std::span<int4, detail::CoreGpuTraits<Scheme>::kCwCount>> cws,
    typename detail::CoreGpuTraits<Scheme>::Input *tr,
    cuda::std::span<int4, 4> cs, cuda::std::span<const int4, 2> s0s,
    typename detail::CoreGpuTraits<Scheme>::Input a, int4 b_buf, int *status, cudaStream_t stream) {
  if (scheme == nullptr || cws.data() == nullptr || cs.data() == nullptr || s0s.data() == nullptr ||
      tr == nullptr || status == nullptr) return cudaErrorInvalidValue;
  return detail::LaunchCoreGen(scheme, cws.data(), tr, cs, s0s, a, b_buf, status, stream);
}

}  // namespace fss::gpu
