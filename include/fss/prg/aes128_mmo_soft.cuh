// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg/aes128_mmo_soft.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Software AES-128 with Matyas-Meyer-Oseas as a PRG.
 *
 * Works on both host and device.
 * The AES core is based on tiny-AES-c by kokke et al. (public domain).
 */

#pragma once
#include <fss/prg.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <fss/util.cuh>

namespace fss::prg {

namespace aes_detail {

constexpr int kNb = 4;
constexpr int kNk = 4;
constexpr int kNr = 10;
constexpr int kRoundKeySize = kNb * (kNr + 1) * 4;  // 176 bytes

__host__ __device__ inline uint8_t Sbox(uint8_t idx) {
    constexpr uint8_t table[256] = {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01,
        0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad,
        0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
        0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05,
        0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e,
        0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20,
        0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb,
        0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40,
        0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c,
        0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60,
        0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4,
        0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a,
        0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b,
        0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9,
        0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87,
        0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99,
        0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};
    return table[idx];
}

__host__ __device__ inline uint8_t Rcon(int idx) {
    constexpr uint8_t table[11] = {
        0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
    return table[idx];
}

__host__ __device__ inline uint32_t ComputeTe0(uint8_t idx) {
    uint8_t s = Sbox(idx);
    uint8_t x2 = (s << 1) ^ (((s >> 7) & 1) * 0x1b);
    uint8_t x3 = s ^ x2;
    return (uint32_t(x2) << 24) | (uint32_t(s) << 16) | (uint32_t(s) << 8) | uint32_t(x3);
}

__host__ __device__ inline void InitTe0(uint32_t *dst) {
    for (int i = 0; i < 256; ++i) dst[i] = ComputeTe0(static_cast<uint8_t>(i));
}

__host__ __device__ inline void InitSbox(uint8_t *dst) {
    for (int i = 0; i < 256; ++i) dst[i] = Sbox(static_cast<uint8_t>(i));
}

__host__ __device__ inline uint32_t RotWord8(uint32_t x) {
    return (x << 24) | (x >> 8);
}

__host__ __device__ inline uint32_t RotWord16(uint32_t x) {
    return (x << 16) | (x >> 16);
}

__host__ __device__ inline uint32_t RotWord24(uint32_t x) {
    return (x << 8) | (x >> 24);
}

__host__ __device__ inline void KeyExpansion(
    uint8_t *round_key, const uint8_t *key, const uint8_t *sbox) {
    for (int i = 0; i < kNk; ++i) {
        round_key[i * 4 + 0] = key[i * 4 + 0];
        round_key[i * 4 + 1] = key[i * 4 + 1];
        round_key[i * 4 + 2] = key[i * 4 + 2];
        round_key[i * 4 + 3] = key[i * 4 + 3];
    }
    for (int i = kNk; i < kNb * (kNr + 1); ++i) {
        uint8_t tempa[4];
        int k = (i - 1) * 4;
        tempa[0] = round_key[k + 0];
        tempa[1] = round_key[k + 1];
        tempa[2] = round_key[k + 2];
        tempa[3] = round_key[k + 3];
        if (i % kNk == 0) {
            uint8_t u8tmp = tempa[0];
            tempa[0] = sbox[tempa[1]];
            tempa[1] = sbox[tempa[2]];
            tempa[2] = sbox[tempa[3]];
            tempa[3] = sbox[u8tmp];
            tempa[0] ^= Rcon(i / kNk);
        }
        int j = i * 4;
        k = (i - kNk) * 4;
        round_key[j + 0] = round_key[k + 0] ^ tempa[0];
        round_key[j + 1] = round_key[k + 1] ^ tempa[1];
        round_key[j + 2] = round_key[k + 2] ^ tempa[2];
        round_key[j + 3] = round_key[k + 3] ^ tempa[3];
    }
}

__host__ __device__ inline uint32_t LoadBE32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

__host__ __device__ inline void StoreBE32(uint8_t *p, uint32_t v) {
    p[0] = uint8_t(v >> 24);
    p[1] = uint8_t(v >> 16);
    p[2] = uint8_t(v >> 8);
    p[3] = uint8_t(v);
}

__host__ __device__ inline void Encrypt(
    uint8_t *buf, const uint8_t *round_key, const uint32_t *te0, const uint8_t *sbox) {
    uint32_t s0 = LoadBE32(buf) ^ LoadBE32(round_key);
    uint32_t s1 = LoadBE32(buf + 4) ^ LoadBE32(round_key + 4);
    uint32_t s2 = LoadBE32(buf + 8) ^ LoadBE32(round_key + 8);
    uint32_t s3 = LoadBE32(buf + 12) ^ LoadBE32(round_key + 12);

#pragma unroll 1
    for (int r = 1; r <= 9; ++r) {
        const uint8_t *rk = round_key + r * 16;
        uint32_t rk0 = LoadBE32(rk);
        uint32_t rk1 = LoadBE32(rk + 4);
        uint32_t rk2 = LoadBE32(rk + 8);
        uint32_t rk3 = LoadBE32(rk + 12);

        uint32_t t0 = te0[s0 >> 24] ^ RotWord8(te0[(s1 >> 16) & 0xff]) ^
            RotWord16(te0[(s2 >> 8) & 0xff]) ^ RotWord24(te0[s3 & 0xff]) ^ rk0;
        uint32_t t1 = te0[s1 >> 24] ^ RotWord8(te0[(s2 >> 16) & 0xff]) ^
            RotWord16(te0[(s3 >> 8) & 0xff]) ^ RotWord24(te0[s0 & 0xff]) ^ rk1;
        uint32_t t2 = te0[s2 >> 24] ^ RotWord8(te0[(s3 >> 16) & 0xff]) ^
            RotWord16(te0[(s0 >> 8) & 0xff]) ^ RotWord24(te0[s1 & 0xff]) ^ rk2;
        uint32_t t3 = te0[s3 >> 24] ^ RotWord8(te0[(s0 >> 16) & 0xff]) ^
            RotWord16(te0[(s1 >> 8) & 0xff]) ^ RotWord24(te0[s2 & 0xff]) ^ rk3;

        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
    }

    // Last round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    const uint8_t *rk = round_key + 160;
    uint32_t rk0 = LoadBE32(rk);
    uint32_t rk1 = LoadBE32(rk + 4);
    uint32_t rk2 = LoadBE32(rk + 8);
    uint32_t rk3 = LoadBE32(rk + 12);

    uint32_t o0 = (uint32_t(sbox[s0 >> 24]) << 24) | (uint32_t(sbox[(s1 >> 16) & 0xff]) << 16) |
        (uint32_t(sbox[(s2 >> 8) & 0xff]) << 8) | uint32_t(sbox[s3 & 0xff]);
    uint32_t o1 = (uint32_t(sbox[s1 >> 24]) << 24) | (uint32_t(sbox[(s2 >> 16) & 0xff]) << 16) |
        (uint32_t(sbox[(s3 >> 8) & 0xff]) << 8) | uint32_t(sbox[s0 & 0xff]);
    uint32_t o2 = (uint32_t(sbox[s2 >> 24]) << 24) | (uint32_t(sbox[(s3 >> 16) & 0xff]) << 16) |
        (uint32_t(sbox[(s0 >> 8) & 0xff]) << 8) | uint32_t(sbox[s1 & 0xff]);
    uint32_t o3 = (uint32_t(sbox[s3 >> 24]) << 24) | (uint32_t(sbox[(s0 >> 16) & 0xff]) << 16) |
        (uint32_t(sbox[(s1 >> 8) & 0xff]) << 8) | uint32_t(sbox[s2 & 0xff]);

    StoreBE32(buf, o0 ^ rk0);
    StoreBE32(buf + 4, o1 ^ rk1);
    StoreBE32(buf + 8, o2 ^ rk2);
    StoreBE32(buf + 12, o3 ^ rk3);
}

}  // namespace aes_detail

/**
 * Software AES-128 with Matyas-Meyer-Oseas as a PRG.
 *
 * Works on both host and device side.
 *
 * @tparam mul See Prgable mul.
 */
template <int mul>
class Aes128Soft {
private:
    uint8_t round_keys_[mul][aes_detail::kRoundKeySize];
    const uint32_t *te0_;
    const uint8_t *sbox_;

public:
    /**
     * Constructor.
     *
     * @param keys mul 16-byte AES-128 keys.
     * @param te0 Pointer to the 256-entry AES T-table (uint32_t[256]).
     * @param sbox Pointer to the AES S-box (uint8_t[256]).
     *
     * On device, both should point to __shared__ memory initialized via
     * InitTe0() and InitSbox(). On host, call InitTe0()/InitSbox() on
     * stack arrays and pass them.
     */
    __host__ __device__ Aes128Soft(
        const uint8_t keys[][16], const uint32_t *te0, const uint8_t *sbox)
        : te0_(te0), sbox_(sbox) {
        for (int i = 0; i < mul; ++i) aes_detail::KeyExpansion(round_keys_[i], keys[i], sbox);
    }

    __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
        cuda::std::array<int4, mul> out{};
        for (int i = 0; i < mul; ++i) {
            out[i] = seed;
            aes_detail::Encrypt(reinterpret_cast<uint8_t *>(&out[i]), round_keys_[i], te0_, sbox_);
            out[i] = fss::util::Xor(out[i], seed);
        }
        return out;
    }
};
static_assert(Prgable<Aes128Soft<1>, 1> && Prgable<Aes128Soft<2>, 2>);

}  // namespace fss::prg
