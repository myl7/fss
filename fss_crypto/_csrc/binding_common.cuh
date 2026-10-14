// SPDX-License-Identifier: Apache-2.0
// Shared helpers for generated PyTorch extension bindings.

#pragma once

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <random>

namespace {

// The ChaCha PRG stores its nonce by value, so one pair of ints feeds both
// host-side instances and kernel-side constructions.
int g_nonce0 = 0, g_nonce1 = 0;
bool g_nonce_initialized = false;

void EnsureNonce() {
  if (g_nonce_initialized) return;
  std::random_device rd;
  g_nonce0 = static_cast<int>(rd());
  g_nonce1 = static_cast<int>(rd());
  g_nonce_initialized = true;
}

// Fixed IV for the verifiable schemes' hash instances (Blake3 copies it by
// value at construction).
constexpr int4 kHashIv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};

cudaStream_t CurrentStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

// Frontier tiling shared by the GPU full-domain wrappers: the frontier depth
// is capped at 16 levels (and the DFS tail at 8), one block walks z - log2(bs)
// root levels, and small domains shrink the block to their whole frontier.
constexpr int EvalAllZ(int bits) { return bits < 16 ? bits : 16; }
constexpr int EvalAllB1(int bits) { int z = EvalAllZ(bits); return z >= 7 ? z - 7 : 0; }
constexpr int EvalAllBS(int bits) { return 1 << (EvalAllZ(bits) - EvalAllB1(bits)); }

template <typename In>
__host__ __device__ In BuildInput(int64_t lo, int64_t hi) {
  if constexpr (sizeof(In) <= 8) {
    return static_cast<In>(static_cast<uint64_t>(lo));
  } else {
    return static_cast<In>(static_cast<uint64_t>(lo)) | (static_cast<In>(static_cast<uint64_t>(hi)) << 64);
  }
}

}  // namespace
