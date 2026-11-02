// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <new>
#include <type_traits>
#include "binding_launch_descriptor.cuh"

namespace fss::python::detail {

template <typename Factory>
concept HasLaunchDescriptor = requires { typename Factory::Descriptor; };

template <typename Descriptor, typename Factory>
__global__ void InitializeObjectKernel(void *storage, Factory factory) {
  using T = typename Descriptor::Scheme;
  new (static_cast<T *>(storage)) T(factory());
}

// Factories that carry their own launch descriptor, e.g. NativeDeviceFactory.
template <typename T, typename Factory>
  requires(std::is_trivially_copyable_v<Factory> && std::is_trivially_destructible_v<T> &&
      HasLaunchDescriptor<Factory>)
inline cudaError_t InitializeObjectGpu(T *storage, Factory factory, cudaStream_t stream) {
  if (storage == nullptr) return cudaErrorInvalidValue;
  InitializeObjectKernel<typename Factory::Descriptor, Factory><<<1, 1, 0, stream>>>(storage, factory);
  return cudaGetLastError();
}

// Factories without a descriptor keep their object type as the launch tag.
template <typename T, typename Factory>
  requires(std::is_trivially_copyable_v<Factory> && std::is_trivially_destructible_v<T> &&
      !HasLaunchDescriptor<Factory>)
inline cudaError_t InitializeObjectGpu(T *storage, Factory factory, cudaStream_t stream) {
  if (storage == nullptr) return cudaErrorInvalidValue;
  InitializeObjectKernel<fss::gpu::detail::CoreLaunchIdentityDescriptor<T>, Factory>
      <<<1, 1, 0, stream>>>(storage, factory);
  return cudaGetLastError();
}

}  // namespace fss::python::detail
