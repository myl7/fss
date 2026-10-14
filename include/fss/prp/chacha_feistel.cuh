// SPDX-License-Identifier: Apache-2.0
/**
 * @file prp/chacha_feistel.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Small-domain PRP via 4-round Feistel network with a ChaCha20-based round
 *   function and cycle-walking.
 *
 * Given a target domain [0, domain), the construction:
 * 1. Computes b = ceil(log2(domain)) and half = ceil(b / 2).
 * 2. Runs a 4-round balanced Feistel on (2 * half)-bit values. The round PRF is the
 *    keyed ChaCha block function already used as the library PRG: the per-round key
 *    (the seed, tweaked by the round number) fills the ChaCha key and the Feistel
 *    half-word (<= 64 bits) rides in the 64-bit nonce slot with counter 0.
 * 3. Cycle-walks: if the Feistel output >= domain, re-applies the Feistel until it
 *    lands in [0, domain). Expected iterations <= 2 because 2^(2*half) < 4 * domain.
 *
 * Security: 4-round Luby-Rackoff with PRF round functions gives a CCA-secure PRP and
 * cycle-walking preserves the permutation property (Black & Rogaway, 2002). The
 * round PRF relies on the same keyed-ChaCha pseudorandomness assumption the library
 * PRG already makes, so no new assumption is introduced.
 *
 * Unlike Aes128Feistel, this backend is stateless and free of OpenSSL, so the same
 * code path serves host and device: Permu is __host__ __device__ and instances are
 * trivially copyable.
 */

#pragma once
#include <fss/prp.cuh>
#include <fss/prg/chacha.cuh>
#include <cuda_runtime.h>
#include <cassert>

namespace fss::prp {

/**
 * Small-domain ChaCha20-Feistel PRP for Cuckoo hashing.
 *
 * This backend satisfies Permutable, is usable from both host and device, and is
 * intended for DMPF/VDMPF paths that need a permutation over a compact integer
 * domain.
 */
class ChaChaFeistel {
  static __host__ __device__ int CeilLog2(__uint128_t x) {
    if (x <= 1) return 0;
    int bits = 0;
    __uint128_t v = x - 1;
    while (v > 0) {
      v >>= 1;
      ++bits;
    }
    return bits;
  }

  static __host__ __device__ int4 Round(int4 key, __uint128_t right) {
    prg::ChaCha<1> chacha(static_cast<int>(right), static_cast<int>(right >> 32));
    return chacha.Gen(key)[0];
  }

  static __host__ __device__ __uint128_t Unpack(int4 v) {
    __uint128_t r = 0;
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.x));
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.y)) << 32;
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.z)) << 64;
    r |= static_cast<__uint128_t>(static_cast<unsigned int>(v.w)) << 96;
    return r;
  }

public:
  /**
   * Small-domain PRP on [0, domain) using 4-round Feistel + cycle-walking.
   *
   * @param seed 16-byte PRP key.
   * @param x Input in [0, domain).
   * @param domain Domain size. Must be >= 2.
   * @return Output in [0, domain).
   */
  __host__ __device__ __uint128_t Permu(int4 seed, __uint128_t x, __uint128_t domain) {
    assert(x < domain);
    if (domain <= 1) return 0;

    int b = CeilLog2(domain);
    int half = (b + 1) / 2;
    __uint128_t mask = (half >= 128) ? ~__uint128_t(0) : (__uint128_t(1) << half) - 1;

    __uint128_t val = x;
    do {
      __uint128_t left = (val >> half) & mask;
      __uint128_t right = val & mask;

      for (int r = 0; r < 4; ++r) {
        // Derive per-round key by XORing round number into the seed.
        int4 round_key = seed;
        round_key.x ^= r;

        // PRF: ChaCha keyed by round_key, right in the 64-bit nonce slot -> truncate to half bits.
        __uint128_t f = Unpack(Round(round_key, right)) & mask;

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
static_assert(Permutable<ChaChaFeistel>);

}  // namespace fss::prp
