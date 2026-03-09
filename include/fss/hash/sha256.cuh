// SPDX-License-Identifier: Apache-2.0
/**
 * @file hash/sha256.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/hash.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cassert>
#include <openssl/evp.h>
#include <fss/util.cuh>

namespace fss::hash {

/**
 * SHA-256 keyed hash.
 *
 * Only for host side.
 */
class Sha256 {
private:
    int4 key_;

public:
    /**
     * Constructor.
     *
     * @param key 16B key stored by value.
     */
    explicit Sha256(int4 key) : key_(key) {}

    /**
     * Hash a 64B message with the key.
     *
     * @param msg 4 int4 blocks (64B).
     * @return 2 int4 blocks (32B) SHA-256 digest.
     */
    __host__ __device__ cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4> msg) {
        cuda::std::array<int4, 2> out{};

#ifdef __CUDA_ARCH__
        assert(false && "Sha256 is not supported on device side");
        __trap();
#else
        int4 buf[5] = {key_, msg[0], msg[1], msg[2], msg[3]};
        auto buf_ptr = reinterpret_cast<const unsigned char *>(buf);
        auto out_ptr = reinterpret_cast<unsigned char *>(out.data());
        int ret = EVP_Digest(buf_ptr, 80, out_ptr, NULL, EVP_sha256(), NULL);
        assert(ret == 1);
#endif

        return out;
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
        cuda::std::array<int4, 4> out{};

#ifdef __CUDA_ARCH__
        assert(false && "Sha256 is not supported on device side");
        __trap();
#else
        auto [a, b] = msg;
        int4 buf[3] = {key_, fss::util::SetLsb(a, false), b};
        auto buf_ptr = reinterpret_cast<const unsigned char *>(buf);
        auto out_ptr = reinterpret_cast<unsigned char *>(out.data());
        int ret = EVP_Digest(buf_ptr, 48, out_ptr, NULL, EVP_sha256(), NULL);
        assert(ret == 1);

        buf[1] = fss::util::SetLsb(a, true);
        ret = EVP_Digest(buf_ptr, 48, out_ptr + 32, NULL, EVP_sha256(), NULL);
        assert(ret == 1);
#endif

        return out;
    }
};
static_assert(Hashable<Sha256> && XorHashable<Sha256>);

}  // namespace fss::hash
