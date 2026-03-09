// SPDX-License-Identifier: Apache-2.0
/**
 * @file hash/blake3.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/hash.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <fss/util.cuh>

namespace fss::hash {

/**
 * BLAKE3 keyed hash (CPU+GPU).
 *
 * The IV replaces BLAKE3's standard IV in keyed-hash mode.
 * The KEYED_HASH flag is set on all compression calls.
 */
class Blake3 {
private:
    int4 iv_[2];

    template <int n>
    __host__ __device__ static int RotateRight(int a) {
        auto u = static_cast<unsigned>(a);
        return static_cast<int>((u >> n) | (u << (32 - n)));
    }

    __host__ __device__ static void G(int &a, int &b, int &c, int &d, int x, int y) {
        a += b + x;
        d = RotateRight<16>(d ^ a);
        c += d;
        b = RotateRight<12>(b ^ c);
        a += b + y;
        d = RotateRight<8>(d ^ a);
        c += d;
        b = RotateRight<7>(b ^ c);
    }

    __host__ __device__ static int GetWord(const int4 m[4], int i) {
        return reinterpret_cast<const int *>(m)[i];
    }

    __host__ __device__ static void SetWord(int4 m[4], int i, int val) {
        reinterpret_cast<int *>(m)[i] = val;
    }

    __host__ __device__ static void Permute(int4 m[4]) {
        constexpr int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
        int tmp[16];
        for (int i = 0; i < 16; ++i) {
            tmp[i] = GetWord(m, perm[i]);
        }
        for (int i = 0; i < 16; ++i) {
            SetWord(m, i, tmp[i]);
        }
    }

    __host__ __device__ static void Round(int4 v[4], const int4 m[4]) {
        // Column rounds: G calls use message words (0,1), (2,3), (4,5), (6,7)
        G(v[0].x, v[1].x, v[2].x, v[3].x, GetWord(m, 0), GetWord(m, 1));
        G(v[0].y, v[1].y, v[2].y, v[3].y, GetWord(m, 2), GetWord(m, 3));
        G(v[0].z, v[1].z, v[2].z, v[3].z, GetWord(m, 4), GetWord(m, 5));
        G(v[0].w, v[1].w, v[2].w, v[3].w, GetWord(m, 6), GetWord(m, 7));
        // Diagonal rounds: G calls use message words (8,9), (10,11), (12,13), (14,15)
        G(v[0].x, v[1].y, v[2].z, v[3].w, GetWord(m, 8), GetWord(m, 9));
        G(v[0].y, v[1].z, v[2].w, v[3].x, GetWord(m, 10), GetWord(m, 11));
        G(v[0].z, v[1].w, v[2].x, v[3].y, GetWord(m, 12), GetWord(m, 13));
        G(v[0].w, v[1].x, v[2].y, v[3].z, GetWord(m, 14), GetWord(m, 15));
    }

    // First 4 standard BLAKE3 IV words
    constexpr static int4 kIv0 = {
        static_cast<int>(0x6A09E667),
        static_cast<int>(0xBB67AE85),
        static_cast<int>(0x3C6EF372),
        static_cast<int>(0xA54FF53A),
    };

    constexpr static int kChunkStart = 1;
    constexpr static int kChunkEnd = 2;
    constexpr static int kRoot = 8;
    constexpr static int kKeyedHash = 16;

    /**
     * BLAKE3 compression function.
     *
     * @param h Chaining value (2 int4s = 8 words).
     * @param msg Message block (4 int4s = 16 words).
     * @param counter 64-bit block counter.
     * @param block_len Number of input bytes in the block.
     * @param flags Domain separation flags.
     * @return 4 int4s (16 words) of output.
     */
    __host__ __device__ static cuda::std::array<int4, 4> Compress(
        const int4 h[2], const int4 msg[4], unsigned long long counter, int block_len, int flags) {
        int4 v[4];
        v[0] = h[0];
        v[1] = h[1];
        v[2] = kIv0;
        v[3] = {static_cast<int>(counter & 0xFFFFFFFF),
            static_cast<int>((counter >> 32) & 0xFFFFFFFF), block_len, flags};

        int4 m[4] = {msg[0], msg[1], msg[2], msg[3]};

        // 7 rounds with permutation between rounds
        for (int i = 0; i < 7; ++i) {
            Round(v, m);
            if (i < 6) {
                Permute(m);
            }
        }

        // Finalization
        v[0] = util::Xor(v[0], v[2]);
        v[1] = util::Xor(v[1], v[3]);
        v[2] = util::Xor(v[2], h[0]);
        v[3] = util::Xor(v[3], h[1]);

        return {v[0], v[1], v[2], v[3]};
    }

public:
    /**
     * Constructor.
     *
     * @param iv Custom IV (2 int4s = 32B) replacing BLAKE3's standard IV in keyed-hash mode.
     * Stored by value.
     */
    __host__ __device__ explicit Blake3(cuda::std::span<const int4, 2> iv) : iv_{iv[0], iv[1]} {}

    /**
     * Hash a 64B message.
     *
     * Single BLAKE3 compression: h=iv_, m=msg, counter=0, block_len=64.
     * flags = CHUNK_START | CHUNK_END | ROOT | KEYED_HASH (0x1B).
     *
     * @param msg 4 int4 blocks (64B).
     * @return 2 int4 blocks (32B), the first 32B of the compression output.
     */
    __host__ __device__ cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4> msg) {
        constexpr int flags = kChunkStart | kChunkEnd | kRoot | kKeyedHash;
        auto out = Compress(iv_, msg.data(), 0, 64, flags);
        return {out[0], out[1]};
    }

    /**
     * XOR-collision-resistant hash.
     *
     * Produces two 32B digests (one with LSB cleared, one with LSB set)
     * and concatenates them into 64B.
     *
     * @param msg Tuple of (a, b) where a's LSB is used for domain separation.
     * @return 4 int4 blocks (64B).
     */
    __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4> msg) {
        constexpr int flags = kChunkStart | kChunkEnd | kRoot | kKeyedHash;
        auto [a, b] = msg;
        int4 padded[4] = {util::SetLsb(a, false), b, {0, 0, 0, 0}, {0, 0, 0, 0}};

        auto out0 = Compress(iv_, padded, 0, 32, flags);

        padded[0] = util::SetLsb(a, true);
        auto out1 = Compress(iv_, padded, 0, 32, flags);

        return {out0[0], out0[1], out1[0], out1[1]};
    }
};
static_assert(Hashable<Blake3> && XorHashable<Blake3>);

}  // namespace fss::hash
