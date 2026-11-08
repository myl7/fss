// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <new>
#include <type_traits>

namespace fss::python::detail {

template <typename T, typename Factory>
__global__ void InitializeObjectKernel(void *storage, Factory factory) {
  new (static_cast<T *>(storage)) T(factory());
}

// The factory constructs the exact Core scheme on its execution device.
template <typename T, typename Factory>
  requires(std::is_trivially_copyable_v<Factory> && std::is_trivially_destructible_v<T>)
inline cudaError_t InitializeObjectGpu(T *storage, Factory factory, cudaStream_t stream) {
  if (storage == nullptr) return cudaErrorInvalidValue;
  InitializeObjectKernel<T, Factory><<<1, 1, 0, stream>>>(storage, factory);
  return cudaGetLastError();
}

}  // namespace fss::python::detail
