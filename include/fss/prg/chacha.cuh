// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg/chacha.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * ChaCha as a PRG.
 */

#pragma once
#include <fss/prg.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <fss/util.cuh>

namespace fss::prg {

template <int mul, int rounds = 20>
    requires(rounds % 2 == 0 && (mul == 2 || mul == 4))
class ChaCha {
private:
    const int *nonce_;

    template <int n>
    __host__ __device__ static int RotateLeft(int a) {
        return (a << n) | (a >> (32 - n));
    }

    __host__ __device__ static void QuarterRound(int &a, int &b, int &c, int &d) {
        a += b;
        d ^= a;
        d = RotateLeft<16>(d);
        c += d;
        b ^= c;
        b = RotateLeft<12>(b);
        a += b;
        d ^= a;
        d = RotateLeft<8>(d);
        c += d;
        b ^= c;
        b = RotateLeft<7>(b);
    }

    __host__ __device__ static void Rounds(int4 x[4]) {
        for (int i = 0; i < rounds; i += 2) {
            // Column rounds
            QuarterRound(x[0].x, x[1].x, x[2].x, x[3].x);
            QuarterRound(x[0].y, x[1].y, x[2].y, x[3].y);
            QuarterRound(x[0].z, x[1].z, x[2].z, x[3].z);
            QuarterRound(x[0].w, x[1].w, x[2].w, x[3].w);
            // Diagonal rounds
            QuarterRound(x[0].x, x[1].y, x[2].z, x[3].w);
            QuarterRound(x[0].y, x[1].z, x[2].w, x[3].x);
            QuarterRound(x[0].z, x[1].w, x[2].x, x[3].y);
            QuarterRound(x[0].w, x[1].x, x[2].y, x[3].z);
        }
    }

    __host__ __device__ static void ExpandKey(int4 x[4]) {
        x[2] = x[1];
    }

    // The nothing-up-my-sleeve constant for 32B key: "expand 32-byte k"
    constexpr static int4 kConstant32 = {
        0x61707865,  // "expa"
        0x3320646e,  // "nd 3"
        0x79622d32,  // "2-by"
        0x6b206574,  // "te k"
    };
    // The nothing-up-my-sleeve constant for 16B key: "expand 16-byte k"
    constexpr static int4 kConstant16 = {
        0x61707865,  // "expa"
        0x3120646e,  // "nd 1"
        0x79622d36,  // "6-by"
        0x6b206574,  // "te k"
    };

public:
    __host__ __device__ ChaCha(const int *nonce) : nonce_(nonce) {}

    __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
        int4 buf[4];

        // Constant
        if constexpr (mul == 2) buf[0] = kConstant16;
        else buf[0] = kConstant32;

        // Key from the seed
        buf[1] = seed;
        ExpandKey(buf);

        // Counter is always 0 because we only gen one block
        buf[3].x = 0;
        buf[3].y = 0;
        // Nonce
        buf[3].z = nonce_[0];
        buf[3].w = nonce_[1];

        Rounds(buf);

        buf[1] = util::Xor(buf[1], seed);
        if constexpr (mul == 2) {
            buf[0] = util::Xor(buf[0], kConstant16);
            return {buf[0], buf[1]};
        } else {
            buf[0] = util::Xor(buf[0], kConstant32);
            buf[2] = util::Xor(buf[2], seed);
            buf[3] = util::Xor(buf[3], {0, 0, nonce_[0], nonce_[1]});
            return {buf[0], buf[1], buf[2], buf[3]};
        }
    }
};
static_assert(Prgable<ChaCha<2>, 2> && Prgable<ChaCha<4>, 4>);

}  // namespace fss::prg
