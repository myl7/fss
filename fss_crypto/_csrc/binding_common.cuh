// SPDX-License-Identifier: Apache-2.0
// Shared helpers for generated PyTorch extension bindings.

#pragma once

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <random>
#include <type_traits>

namespace {

// Function-local initialization is serialized by the C++ runtime.
const auto &BindingNonce() {
  static const cuda::std::array<int, 2> nonce = [] {
    std::random_device random;
    return cuda::std::array<int, 2>{static_cast<int>(random()), static_cast<int>(random())};
  }();
  return nonce;
}

void EnsureNonce() {
  (void)BindingNonce();
}

template <typename Prg = PrgType>
Prg MakePrg() {
  if constexpr (std::is_constructible_v<Prg, int, int>) {
    const auto &nonce = BindingNonce();
    return Prg(nonce[0], nonce[1]);
  } else {
    return Prg{};
  }
}

void CheckCudaPrg(const torch::Tensor &tensor) {
  TORCH_CHECK(!tensor.is_cuda() || kGpuPrg, "aes128_mmo is CPU-only");
}

void CheckCudaHashes(const torch::Tensor &tensor) {
  // The generated source supplies these capability flags without requiring a
  // host-only hash header to instantiate any device operations.
  TORCH_CHECK(!tensor.is_cuda() || kGpuXorHash, "xor_hash='sha256' is CPU-only");
  TORCH_CHECK(!tensor.is_cuda() || kGpuHash, "hash='sha256' is CPU-only");
}

template <int bits = kInBits>
void CheckQueryDomain(const torch::Tensor &xs) {
  if (xs.numel() == 0) return;
  TORCH_CHECK(xs.min().item<int64_t>() >= 0, "xs must be nonnegative");
  if constexpr (bits < 63) {
    TORCH_CHECK(xs.max().item<int64_t>() < (int64_t{1} << bits), "xs must be in [0, 2^", bits, ")");
  }
}

// Fixed IV for the verifiable schemes' hash instances (Blake3 copies it by
// value at construction).
constexpr int4 kHashIv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};

cudaStream_t CurrentStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

void CopyDeviceToHost(void *host, const void *device, size_t bytes, const char *name) {
  // Host launch parameters must observe producers on PyTorch's current stream.
  auto stream = CurrentStream();
  auto err = cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream);
  TORCH_CHECK(err == cudaSuccess, "device-to-host ", name, " copy failed: ", cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream);
  TORCH_CHECK(err == cudaSuccess, "device-to-host ", name, " synchronization failed: ", cudaGetErrorString(err));
}

// Fetch 16B/64B scalars from possibly-device memory to the host; the GPU
// wrappers take seeds and check values by value.
inline int4 FetchInt4(const int4 *p, bool is_cuda) {
  int4 v{};
  if (is_cuda) {
    CopyDeviceToHost(&v, p, sizeof(int4), "scalar");
  } else {
    v = p[0];
  }
  return v;
}

inline cuda::std::array<int4, 4> FetchCs(const int4 *p, bool is_cuda) {
  cuda::std::array<int4, 4> v{};
  if (is_cuda) {
    CopyDeviceToHost(v.data(), p, 4 * sizeof(int4), "cs");
  } else {
    for (int i = 0; i < 4; ++i) v[i] = p[i];
  }
  return v;
}

// Frontier tiling shared by the GPU full-domain wrappers: the frontier depth
// is capped at 16 levels (and the DFS tail at 8), one block walks z - log2(bs)
// root levels, and small domains shrink the block to their whole frontier.
constexpr int EvalAllZ(int bits) {
  return bits < 16 ? bits : 16;
}
constexpr int EvalAllB1(int bits) {
  int z = EvalAllZ(bits);
  return z >= 7 ? z - 7 : 0;
}
constexpr int EvalAllBS(int bits) {
  return 1 << (EvalAllZ(bits) - EvalAllB1(bits));
}

// Frontier tiling for the wrappers whose kernels take the correction-word
// type generically (Dpf and Dcf): domains above 16 bits deepen the frontier
// so the per-thread DFS tail stays at <= 4 levels and the thread-local node
// arrays keep the measured register footprint; the shared-memory frontier is
// unchanged because z - b1 stays at 7.
constexpr int WideDomainEvalAllZ(int bits) {
  return bits <= 16 ? bits : (bits - 4 > 16 ? bits - 4 : 16);
}
constexpr int WideDomainEvalAllB1(int bits) {
  int z = WideDomainEvalAllZ(bits);
  return z >= 7 ? z - 7 : 0;
}
constexpr int WideDomainEvalAllBS(int bits) {
  return 1 << (WideDomainEvalAllZ(bits) - WideDomainEvalAllB1(bits));
}

// The grid.x dimension of the wide-domain launch is 2^(z-7) blocks, so the
// 2^31-1 dimension limit caps in_bits at 41.
constexpr int kMaxCudaEvalAllBits = 41;

// The full-domain output tensor holds 2^(in_bits+2) int32 elements, which
// stays inside the int64 element count up to in_bits = 60.
constexpr int kMaxEvalAllBits = 60;

template <typename In>
__host__ __device__ In BuildInput(int64_t lo, int64_t hi) {
  if constexpr (sizeof(In) <= 8) {
    return static_cast<In>(static_cast<uint64_t>(lo));
  } else {
    return static_cast<In>(static_cast<uint64_t>(lo)) | (static_cast<In>(static_cast<uint64_t>(hi)) << 64);
  }
}

}  // namespace
