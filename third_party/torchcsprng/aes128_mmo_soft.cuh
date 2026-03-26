// SPDX-License-Identifier: BSD-3-Clause
/*
 * Copyright (c) 2020, pytorch
 * All rights reserved.
 *
 * See LICENSE in this directory for the full license text.
 *
 * Textbook AES-128 with Matyas-Meyer-Oseas, ported from Meta torchcsprng.
 * Byte-by-byte AES (no T-table optimization).
 * The AES core is from tiny-AES-c by kokke et al. (public domain),
 * adapted for CUDA by Pavel Belevich (Meta).
 *
 * Source: https://github.com/meta-pytorch/csprng
 * File:   torchcsprng/csrc/aes.inc
 */

#pragma once
#include <fss/prg.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <fss/util.cuh>

namespace torchcsprng {

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

__host__ __device__ inline void KeyExpansion(uint8_t *round_key, const uint8_t *key) {
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
            tempa[0] = Sbox(tempa[1]);
            tempa[1] = Sbox(tempa[2]);
            tempa[2] = Sbox(tempa[3]);
            tempa[3] = Sbox(u8tmp);
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

// Textbook AES-128 encrypt: SubBytes + ShiftRows + MixColumns per round.
// State is uint8_t[16] in column-major order (AES standard).
__host__ __device__ inline void Encrypt(uint8_t *state, const uint8_t *round_key) {
    for (int i = 0; i < 16; ++i) state[i] ^= round_key[i];

#pragma unroll 1
    for (int round = 1; round <= kNr; ++round) {
        // SubBytes
        for (int i = 0; i < 16; ++i) state[i] = Sbox(state[i]);

        // ShiftRows
        uint8_t tmp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = tmp;

        tmp = state[2];
        state[2] = state[10];
        state[10] = tmp;
        tmp = state[6];
        state[6] = state[14];
        state[14] = tmp;

        tmp = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = state[3];
        state[3] = tmp;

        // MixColumns (skip on last round)
        if (round < kNr) {
            for (int col = 0; col < 4; ++col) {
                int c = col * 4;
                uint8_t a0 = state[c + 0], a1 = state[c + 1];
                uint8_t a2 = state[c + 2], a3 = state[c + 3];
                uint8_t sum = a0 ^ a1 ^ a2 ^ a3;

                auto xtime = [](uint8_t x) -> uint8_t {
                    return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
                };

                state[c + 0] = a0 ^ xtime(a0 ^ a1) ^ sum;
                state[c + 1] = a1 ^ xtime(a1 ^ a2) ^ sum;
                state[c + 2] = a2 ^ xtime(a2 ^ a3) ^ sum;
                state[c + 3] = a3 ^ xtime(a3 ^ a0) ^ sum;
            }
        }

        // AddRoundKey
        const uint8_t *rk = round_key + round * 16;
        for (int i = 0; i < 16; ++i) state[i] ^= rk[i];
    }
}

}  // namespace aes_detail

/**
 * Textbook AES-128 with Matyas-Meyer-Oseas as a PRG.
 *
 * Byte-by-byte SubBytes/ShiftRows/MixColumns (no T-table).
 * Implements the fss Prgable concept for comparison benchmarking.
 *
 * @tparam mul See Prgable mul.
 */
template <int mul>
class Aes128Mmo {
private:
    uint8_t round_keys_[mul][aes_detail::kRoundKeySize];

public:
    __host__ __device__ Aes128Mmo(const uint8_t keys[][16]) {
        for (int i = 0; i < mul; ++i) aes_detail::KeyExpansion(round_keys_[i], keys[i]);
    }

    __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
        cuda::std::array<int4, mul> out{};
        for (int i = 0; i < mul; ++i) {
            out[i] = seed;
            aes_detail::Encrypt(reinterpret_cast<uint8_t *>(&out[i]), round_keys_[i]);
            out[i] = fss::util::Xor(out[i], seed);
        }
        return out;
    }
};
static_assert(Prgable<Aes128Mmo<1>, 1> && Prgable<Aes128Mmo<2>, 2>);

}  // namespace torchcsprng
