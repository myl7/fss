// SPDX-License-Identifier: Apache-2.0
// Binding-owned AES PRG contexts. Keys are shared within one extension while
// mutable cipher contexts belong to each host thread, including OpenMP workers.

#pragma once

#include <fss/prg/aes128_mmo.cuh>
#include <random>

namespace fss::python {

template <int mul>
class ThreadLocalAes128Mmo {
public:
  using Keys = cuda::std::array<cuda::std::array<unsigned char, 16>, mul>;

  static const Keys &PublicKeys() {
    static const Keys keys = [] {
      Keys result{};
      std::random_device random;
      for (auto &key : result) {
        for (auto &byte : key) byte = static_cast<unsigned char>(random());
      }
      return result;
    }();
    return keys;
  }

  class ContextOwner {
  public:
    explicit ContextOwner(const Keys &keys) : ctxs_(Create(keys)) {}
    ~ContextOwner() {
      prg::Aes128Mmo<mul>::FreeCtxs(ctxs_);
    }
    ContextOwner(const ContextOwner &) = delete;
    ContextOwner &operator=(const ContextOwner &) = delete;

    prg::Aes128Mmo<mul> Get() {
      return prg::Aes128Mmo<mul>(ctxs_);
    }

  private:
    static cuda::std::array<EVP_CIPHER_CTX *, mul> Create(const Keys &keys) {
      const unsigned char *pointers[mul];
      for (int i = 0; i < mul; ++i) pointers[i] = keys[i].data();
      return prg::Aes128Mmo<mul>::CreateCtxs(pointers);
    }
    cuda::std::array<EVP_CIPHER_CTX *, mul> ctxs_;
  };

  static ContextOwner &ThreadContexts() {
    static thread_local ContextOwner owner(PublicKeys());
    return owner;
  }

  __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
#ifdef __CUDA_ARCH__
    assert(false && "aes128_mmo is CPU-only");
    __trap();
    return {};
#else
    return ThreadContexts().Get().Gen(seed);
#endif
  }
};

}  // namespace fss::python
