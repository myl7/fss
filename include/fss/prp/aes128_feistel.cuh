// SPDX-License-Identifier: Apache-2.0
/**
 * @file prp/aes128_feistel.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Small-domain PRP via 4-round Feistel network with AES-128 round function and
 *   cycle-walking.
 *
 * Given a target domain [0, domain), the construction:
 * 1. Computes b = ceil(log2(domain)) and half = ceil(b / 2).
 * 2. Runs a 4-round balanced Feistel on (2 * half)-bit values, using AES-128 as the PRF
 *    for each round (with per-round key tweaking).
 * 3. Cycle-walks: if the Feistel output >= domain, re-applies the Feistel until it lands
 *    in [0, domain). Expected iterations < 4 because 2^(2*half) < 4 * domain.
 *
 * Security: 4-round Luby-Rackoff gives CCA-secure PRP. Cycle-walking preserves the
 * permutation property (Black & Rogaway, 2002).
 */

#pragma once
#include <fss/prp.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <openssl/evp.h>
#include <openssl/aes.h>

namespace fss::prp {

/**
 * Declared and never defined. Device code that reaches it fails to link with an
 * unresolved-extern error that names this symbol.
 */
__device__ void fss_host_only_Aes128Feistel_used_in_device_code();

/**
 * Small-domain AES-128 Feistel PRP for Cuckoo hashing.
 *
 * This backend is host-only because it uses OpenSSL. It satisfies Permutable
 * and is intended for VDMPF setup paths that need a permutation over a compact
 * integer domain. All special members and Permu() are `__host__ __device__`, so
 * pure host use of schemes that hold a Prp by value builds without
 * execution-space warnings. Device use of any of them, e.g. calling Permu() or
 * copying the object in a kernel, fails to build with an unresolved-extern error
 * that names `fss_host_only_Aes128Feistel_used_in_device_code`.
 */
class Aes128Feistel {
public:
  __host__ __device__ Aes128Feistel();
  __host__ __device__ ~Aes128Feistel();
  __host__ __device__ Aes128Feistel(const Aes128Feistel &);
  __host__ __device__ Aes128Feistel &operator=(const Aes128Feistel &);
  __host__ __device__ Aes128Feistel(Aes128Feistel &&o) noexcept;
  __host__ __device__ Aes128Feistel &operator=(Aes128Feistel &&o) noexcept;

  /**
     * Small-domain PRP on [0, domain) using 4-round Feistel + cycle-walking.
     *
     * Host-only. Device use fails to build with an unresolved-extern error naming
     * `fss_host_only_Aes128Feistel_used_in_device_code`.
     *
     * @param seed 16-byte PRP key.
     * @param x Input in [0, domain).
     * @param domain Domain size. Must be >= 2.
     * @return Output in [0, domain).
     */
  __host__ __device__ __uint128_t Permu(int4 seed, __uint128_t x, __uint128_t domain);

private:
  EVP_CIPHER_CTX *ctx_;

  void InitCtx();

  static int CeilLog2(__uint128_t x);

  int4 RawAes(int4 key, int4 plaintext);

  static __uint128_t Unpack(int4 v);

  static int4 Pack(__uint128_t v);
};
// -----------------------------------------------------------------------------

inline void Aes128Feistel::InitCtx() {
  ctx_ = EVP_CIPHER_CTX_new();
  assert(ctx_ != NULL);
  unsigned char dummy[16] = {};
  int ret = EVP_EncryptInit_ex2(ctx_, EVP_aes_128_ecb(), dummy, NULL, NULL);
  assert(ret == 1);
  ret = EVP_CIPHER_CTX_set_padding(ctx_, 0);
  assert(ret == 1);
}

inline int Aes128Feistel::CeilLog2(__uint128_t x) {
  if (x <= 1) return 0;
  int bits = 0;
  __uint128_t v = x - 1;
  while (v > 0) {
    v >>= 1;
    ++bits;
  }
  return bits;
}

inline int4 Aes128Feistel::RawAes(int4 key, int4 plaintext) {
  int4 out{};
  auto key_ptr = reinterpret_cast<const unsigned char *>(&key);
  auto in_ptr = reinterpret_cast<const unsigned char *>(&plaintext);
  auto out_ptr = reinterpret_cast<unsigned char *>(&out);

  int ret = EVP_EncryptInit_ex2(ctx_, NULL, key_ptr, NULL, NULL);
  assert(ret == 1);

  int cipher_len = 0;
  ret = EVP_EncryptUpdate(ctx_, out_ptr, &cipher_len, in_ptr, AES_BLOCK_SIZE);
  assert(ret == 1);
  assert(cipher_len == AES_BLOCK_SIZE);

  return out;
}

inline __uint128_t Aes128Feistel::Unpack(int4 v) {
  __uint128_t r = 0;
  r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.x));
  r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.y)) << 32;
  r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.z)) << 64;
  r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.w)) << 96;
  return r;
}

inline int4 Aes128Feistel::Pack(__uint128_t v) {
  return {static_cast<int>(v & 0xFFFFFFFF), static_cast<int>((v >> 32) & 0xFFFFFFFF),
      static_cast<int>((v >> 64) & 0xFFFFFFFF), static_cast<int>((v >> 96) & 0xFFFFFFFF)};
}

__host__ __device__ inline Aes128Feistel::Aes128Feistel() {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
  ctx_ = nullptr;
#else
  InitCtx();
#endif
}

__host__ __device__ inline Aes128Feistel::~Aes128Feistel() {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
#else
  if (ctx_) EVP_CIPHER_CTX_free(ctx_);
#endif
}

__host__ __device__ inline Aes128Feistel::Aes128Feistel(const Aes128Feistel &) {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
  ctx_ = nullptr;
#else
  InitCtx();
#endif
}

__host__ __device__ inline Aes128Feistel &Aes128Feistel::operator=(const Aes128Feistel &) {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
#endif
  return *this;
}

__host__ __device__ inline Aes128Feistel::Aes128Feistel(Aes128Feistel &&o) noexcept : ctx_(o.ctx_) {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
#else
  o.ctx_ = nullptr;
#endif
}

__host__ __device__ inline Aes128Feistel &Aes128Feistel::operator=(Aes128Feistel &&o) noexcept {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
#else
  if (this != &o) {
    EVP_CIPHER_CTX_free(ctx_);
    ctx_ = o.ctx_;
    o.ctx_ = nullptr;
  }
#endif
  return *this;
}

__host__ __device__ inline __uint128_t Aes128Feistel::Permu(int4 seed, __uint128_t x, __uint128_t domain) {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Feistel_used_in_device_code();
  return 0;
#else
  assert(x < domain);
  if (domain <= 1) return 0;

  int b = CeilLog2(domain);
  int half = (b + 1) / 2;
  __uint128_t mask = (__uint128_t(1) << half) - 1;

  __uint128_t val = x;
  do {
    __uint128_t left = (val >> half) & mask;
    __uint128_t right = val & mask;

    for (int round = 0; round < 4; ++round) {
      // Derive per-round key by XORing round number into the seed.
      int4 round_key = seed;
      round_key.x ^= round;

      // PRF: AES(round_key, pad128(right)) -> truncate to half bits.
      int4 prf_out = RawAes(round_key, Pack(right));
      __uint128_t f = Unpack(prf_out) & mask;

      left = left ^ f;
      __uint128_t tmp = left;
      left = right;
      right = tmp;
    }

    val = (left << half) | right;
  } while (val >= domain);

  return val;
#endif
}

static_assert(Permutable<Aes128Feistel>);

}  // namespace fss::prp
