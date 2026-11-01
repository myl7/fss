// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cuda/std/span>
#include <cstddef>
#include <cstdint>
#include <fss/prg/aes128_mmo_soft.cuh>

namespace fss::python::detail {

struct Aes128SoftTables {
  cuda::std::span<uint32_t, 256> te0;
  cuda::std::span<uint8_t, 256> sbox;
};

template <typename = void>
__global__ void InitializeAes128SoftTablesKernel(Aes128SoftTables tables) {
  unsigned int i = threadIdx.x;
  if (i < 256) {
    tables.te0[i] = prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
    tables.sbox[i] = prg::aes_detail::Sbox(static_cast<uint8_t>(i));
  }
}

template <int mul>
  requires(mul > 0)
inline cudaError_t InitializeAes128SoftMaterialGpu(cuda::std::span<const uint8_t, static_cast<size_t>(mul) * 16> host_keys,
    uint8_t (*device_keys)[16], Aes128SoftTables tables, cudaStream_t stream) {
  if (host_keys.data() == nullptr || device_keys == nullptr || tables.te0.data() == nullptr || tables.sbox.data() == nullptr)
    return cudaErrorInvalidValue;
  cudaError_t error = cudaMemcpyAsync(device_keys, host_keys.data(), static_cast<size_t>(mul) * 16,
      cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess) return error;
  InitializeAes128SoftTablesKernel<><<<1, 256, 0, stream>>>(tables);
  return cudaGetLastError();
}

}  // namespace fss::python::detail
