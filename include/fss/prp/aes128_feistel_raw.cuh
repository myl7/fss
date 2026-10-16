// SPDX-License-Identifier: Apache-2.0
/**
 * @file prp/aes128_feistel_raw.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Small-domain PRP via 4-round Feistel network with AES-128 round function and
 *   cycle-walking, computed with AES-NI intrinsics.
 *
 * Bit-exact with Aes128Feistel: same construction (4-round Feistel, per-round key
 * seed ^ round, cycle-walking), same AES-128 block cipher, but every AES block
 * operation runs as ten aesenc intrinsics instead of an OpenSSL EVP call. The four
 * per-round key schedules are expanded once per seed and cached in the instance, so
 * a mapping build with a fixed sigma expands them a single time. The Permu seed
 * parameter keeps its concept signature; passing a different seed than the cached
 * one re-expands the schedules, so every call uses the seed it was given.
 *
 * Only for host side (x86-64 with AES-NI). Instances are trivially copyable; the
 * cache makes concurrent Permu calls with differing seeds on one instance a data
 * race, like Aes128Feistel's shared OpenSSL context.
 */

#pragma once
#include <fss/prp.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

#ifndef __CUDA_ARCH__
  #include <wmmintrin.h>
  #include <emmintrin.h>
  #include <smmintrin.h>
#endif

namespace fss::prp {

/**
 * Small-domain AES-128 Feistel PRP for Cuckoo hashing, computed with AES-NI
 * intrinsics.
 *
 * Bit-exact with Aes128Feistel and host-only like it, but free of OpenSSL and of
 * per-call key schedules: intended for DMPF/VDMPF setup and evaluation paths whose
 * PRP seed stays fixed for the duration of one mapping build.
 */
class Aes128FeistelRaw {
  static constexpr int kAesRoundKeys = 11;
  static constexpr int kAesRoundKeySize = kAesRoundKeys * 16;  // 176 bytes
  static constexpr int kFeistelRounds = 4;

  alignas(16) uint8_t round_keys_[kFeistelRounds][kAesRoundKeySize] = {};
  int4 cached_seed_{};
  bool has_seed_ = false;

#ifndef __CUDA_ARCH__
  static __m128i KeyExpStep(__m128i key, __m128i kg) {
    kg = _mm_shuffle_epi32(kg, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, kg);
  }

  static void ExpandKey(const uint8_t *key, uint8_t *rk_bytes) {
    __m128i rk[kAesRoundKeys];
    rk[0] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key));
    rk[1] = KeyExpStep(rk[0], _mm_aeskeygenassist_si128(rk[0], 0x01));
    rk[2] = KeyExpStep(rk[1], _mm_aeskeygenassist_si128(rk[1], 0x02));
    rk[3] = KeyExpStep(rk[2], _mm_aeskeygenassist_si128(rk[2], 0x04));
    rk[4] = KeyExpStep(rk[3], _mm_aeskeygenassist_si128(rk[3], 0x08));
    rk[5] = KeyExpStep(rk[4], _mm_aeskeygenassist_si128(rk[4], 0x10));
    rk[6] = KeyExpStep(rk[5], _mm_aeskeygenassist_si128(rk[5], 0x20));
    rk[7] = KeyExpStep(rk[6], _mm_aeskeygenassist_si128(rk[6], 0x40));
    rk[8] = KeyExpStep(rk[7], _mm_aeskeygenassist_si128(rk[7], 0x80));
    rk[9] = KeyExpStep(rk[8], _mm_aeskeygenassist_si128(rk[8], 0x1b));
    rk[10] = KeyExpStep(rk[9], _mm_aeskeygenassist_si128(rk[9], 0x36));
    for (int i = 0; i < kAesRoundKeys; ++i) _mm_store_si128(reinterpret_cast<__m128i *>(rk_bytes + i * 16), rk[i]);
  }
#endif

  // Expand the per-round key schedules for seed unless already cached.
  // Host-only like the whole backend; the device-pass guard only keeps the
  // intrinsics helpers out of unannotated-function parsing.
  void UpdateKeys(int4 seed) {
    if (has_seed_ && seed.x == cached_seed_.x && seed.y == cached_seed_.y && seed.z == cached_seed_.z &&
        seed.w == cached_seed_.w) {
      return;
    }
    cached_seed_ = seed;
    has_seed_ = true;
#ifndef __CUDA_ARCH__
    for (int round = 0; round < kFeistelRounds; ++round) {
      // Derive per-round key by XORing round number into the seed.
      int4 round_key = seed;
      round_key.x ^= round;
      ExpandKey(reinterpret_cast<const uint8_t *>(&round_key), round_keys_[round]);
    }
#endif
  }

  static int CeilLog2(__uint128_t x) {
    if (x <= 1) return 0;
    int bits = 0;
    __uint128_t v = x - 1;
    while (v > 0) {
      v >>= 1;
      ++bits;
    }
    return bits;
  }

  static int4 RawAes(const uint8_t *rk_bytes, int4 plaintext) {
    int4 out{};
#ifndef __CUDA_ARCH__
    const auto *rk = reinterpret_cast<const __m128i *>(rk_bytes);
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&plaintext));
    b = _mm_xor_si128(b, _mm_load_si128(rk));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 1));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 2));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 3));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 4));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 5));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 6));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 7));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 8));
    b = _mm_aesenc_si128(b, _mm_load_si128(rk + 9));
    b = _mm_aesenclast_si128(b, _mm_load_si128(rk + 10));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(&out), b);
#endif
    return out;
  }

  static __uint128_t Unpack(int4 v) {
    __uint128_t r = 0;
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.x));
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.y)) << 32;
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.z)) << 64;
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.w)) << 96;
    return r;
  }

  static int4 Pack(__uint128_t v) {
    return {static_cast<int>(v & 0xFFFFFFFF), static_cast<int>((v >> 32) & 0xFFFFFFFF),
        static_cast<int>((v >> 64) & 0xFFFFFFFF), static_cast<int>((v >> 96) & 0xFFFFFFFF)};
  }

public:
  Aes128FeistelRaw() = default;

  /**
     * Constructor pre-expanding the round keys for seed.
     *
     * @param seed 16-byte PRP key the mapping build will use.
     */
  explicit Aes128FeistelRaw(int4 seed) {
    UpdateKeys(seed);
  }

  /**
     * Small-domain PRP on [0, domain) using 4-round Feistel + cycle-walking.
     *
     * Bit-exact with Aes128Feistel::Permu. Host-only.
     *
     * @param seed 16-byte PRP key. Round keys are expanded on first use per seed
     *   and cached, so repeated calls with one seed skip the expansion.
     * @param x Input in [0, domain).
     * @param domain Domain size. Must be >= 2.
     * @return Output in [0, domain).
     */
  __uint128_t Permu(int4 seed, __uint128_t x, __uint128_t domain) {
    assert(x < domain);
    if (domain <= 1) return 0;

    UpdateKeys(seed);

    int b = CeilLog2(domain);
    int half = (b + 1) / 2;
    __uint128_t mask = (half >= 128) ? ~__uint128_t(0) : (__uint128_t(1) << half) - 1;

    __uint128_t val = x;
    do {
      __uint128_t left = (val >> half) & mask;
      __uint128_t right = val & mask;

      for (int round = 0; round < kFeistelRounds; ++round) {
        // PRF: AES(round_key, pad128(right)) -> truncate to half bits.
        int4 prf_out = RawAes(round_keys_[round], Pack(right));
        __uint128_t f = Unpack(prf_out) & mask;

        left = left ^ f;
        __uint128_t tmp = left;
        left = right;
        right = tmp;
      }

      val = (left << half) | right;
    } while (val >= domain);

    return val;
  }
};
static_assert(Permutable<Aes128FeistelRaw>);

}  // namespace fss::prp
