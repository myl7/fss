// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg/aes128_mmo_raw.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief AES-128 MMO PRG using direct AES-NI intrinsics without OpenSSL overhead.
 *
 * Round keys are pre-expanded at construction time.
 * Each Gen() call issues one AES-NI encrypt sequence per output block
 * with no per-call context overhead.
 *
 * Only for host side (x86-64 with AES-NI).
 */

#pragma once
#include <fss/prg.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cstdint>
#include <cassert>

#ifndef __CUDA_ARCH__
    #include <wmmintrin.h>
    #include <emmintrin.h>
    #include <smmintrin.h>
#endif

namespace fss::prg {

/**
 * AES-128 with Matyas-Meyer-Oseas and AES-NI intrinsics as a PRG.
 *
 * Only for host side.
 *
 * @tparam mul See Prgable mul.
 */
template <int mul>
class Aes128MmoRaw {
private:
    static constexpr int kRoundKeys = 11;
    static constexpr int kRoundKeySize = kRoundKeys * 16;  // 176 bytes
    alignas(16) uint8_t round_keys_[mul][kRoundKeySize];

#ifndef __CUDA_ARCH__
    static __m128i KeyExpStep(__m128i key, __m128i kg) {
        kg = _mm_shuffle_epi32(kg, _MM_SHUFFLE(3, 3, 3, 3));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        return _mm_xor_si128(key, kg);
    }

    static void ExpandKey(const uint8_t *key, uint8_t *rk_bytes) {
        __m128i rk[kRoundKeys];
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
        for (int i = 0; i < kRoundKeys; ++i)
            _mm_store_si128(reinterpret_cast<__m128i *>(rk_bytes + i * 16), rk[i]);
    }
#endif

public:
    /**
     * Constructor.
     *
     * @param keys mul 16-byte AES-128 keys.
     */
    Aes128MmoRaw(const uint8_t keys[][16]) {
#ifndef __CUDA_ARCH__
        for (int i = 0; i < mul; ++i) ExpandKey(keys[i], round_keys_[i]);
#endif
    }

    __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
        cuda::std::array<int4, mul> out{};

#ifdef __CUDA_ARCH__
        assert(false && "Aes128MmoRaw is not supported on device side");
        __trap();
#else
        __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&seed));
        for (int i = 0; i < mul; ++i) {
            const auto *rk = reinterpret_cast<const __m128i *>(round_keys_[i]);
            __m128i b = _mm_xor_si128(s, _mm_load_si128(rk));
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
            b = _mm_xor_si128(b, s);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(&out[i]), b);
        }
#endif

        return out;
    }
};
static_assert(
    Prgable<Aes128MmoRaw<1>, 1> && Prgable<Aes128MmoRaw<2>, 2> && Prgable<Aes128MmoRaw<4>, 4>);

}  // namespace fss::prg
