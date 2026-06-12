// SPDX-License-Identifier: Apache-2.0
// Shared helpers for generated PyTorch extension bindings.

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <random>

namespace {

int g_host_nonce[2] = {0, 0};
int *g_device_nonce = nullptr;
bool g_host_nonce_initialized = false;
bool g_device_nonce_initialized = false;

void EnsureHostNonce() {
  if (g_host_nonce_initialized) return;
  std::random_device rd;
  g_host_nonce[0] = static_cast<int>(rd());
  g_host_nonce[1] = static_cast<int>(rd());
  g_host_nonce_initialized = true;
}

void EnsureNonce() {
  EnsureHostNonce();
  if (g_device_nonce_initialized) return;

  auto err = cudaMalloc(&g_device_nonce, 2 * sizeof(int));
  TORCH_CHECK(err == cudaSuccess, "cudaMalloc for nonce failed: ", cudaGetErrorString(err));

  err = cudaMemcpy(g_device_nonce, g_host_nonce, 2 * sizeof(int), cudaMemcpyHostToDevice);
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpy for nonce failed: ", cudaGetErrorString(err));

  g_device_nonce_initialized = true;
}

template <typename In>
__host__ __device__ In BuildInput(int64_t lo, int64_t hi) {
  if constexpr (sizeof(In) <= 8) {
    return static_cast<In>(static_cast<uint64_t>(lo));
  } else {
    return static_cast<In>(static_cast<uint64_t>(lo)) | (static_cast<In>(static_cast<uint64_t>(hi)) << 64);
  }
}

}  // namespace
