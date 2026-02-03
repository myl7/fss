// SPDX-License-Identifier: Apache-2.0
/**
 * @file prg/aes128_mmo.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * AES-128 with Matyas-Meyer-Oseas and pre-initialized cipher context as a PRG.
 */

#pragma once
#include <fss/prg.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cassert>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <fss/util.cuh>

namespace fss::prg {

template <int mul>
class Aes128Mmo {
private:
    EVP_CIPHER_CTX *ctxs_[mul];

public:
    __host__ Aes128Mmo(EVP_CIPHER_CTX *ctxs[mul]) {
        for (int i = 0; i < mul; i++) {
            ctxs_[i] = ctxs[i];
        }
    }
    __host__ static cuda::std::array<EVP_CIPHER_CTX *, mul> InitCtxs(
        const unsigned char *keys[mul]) {
        int ret;
        cuda::std::array<EVP_CIPHER_CTX *, mul> ctxs;

        for (int i = 0; i < mul; ++i) {
            EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
            assert(ctx != NULL);

            ret = EVP_EncryptInit_ex2(ctx, EVP_aes_128_ecb(), keys[i], NULL, NULL);
            assert(ret == 1);

            ret = EVP_CIPHER_CTX_set_padding(ctx, 0);
            assert(ret == 1);

            ctxs[i] = ctx;
        }
        return ctxs;
    }

    __host__ static void FreeCtxs(EVP_CIPHER_CTX *ctxs[mul]) {
        for (int i = 0; i < mul; ++i) {
            EVP_CIPHER_CTX_free(ctxs[i]);
        }
    }

    __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
        cuda::std::array<int4, mul> out{};

#ifdef __CUDA_ARCH__
        assert(false && "Aes128Mmo is not supported on device side.");
        __trap();
#else
        for (int i = 0; i < mul; ++i) {
            int ret;

            auto out_ptr = reinterpret_cast<unsigned char *>(&out[i]);
            auto seed_ptr = reinterpret_cast<const unsigned char *>(&seed);
            int cipher_len = 0;
            // Ctx does not change after block encryption because we use ECB, no padding, and AES_BLOCK_SIZE input size.
            ret = EVP_EncryptUpdate(ctxs_[i], out_ptr, &cipher_len, seed_ptr, AES_BLOCK_SIZE);
            assert(ret == 1);
            assert(cipher_len == AES_BLOCK_SIZE);

            out[i] = fss::util::Xor(out[i], seed);
        }
#endif

        return out;
    }
};
static_assert(Prgable<Aes128Mmo<2>, 2> && Prgable<Aes128Mmo<4>, 4>);

}  // namespace fss::prg
