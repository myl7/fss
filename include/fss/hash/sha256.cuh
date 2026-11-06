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
#include <stdexcept>
#include <memory>

namespace fss::hash {

/**
 * Declared and never defined. Device code that reaches it fails to link with an
 * unresolved-extern error that names this symbol.
 */
__device__ void fss_host_only_Sha256_used_in_device_code();

/**
 * SHA-256 keyed hash.
 *
 * Only for host side. Device use fails to build with an unresolved-extern error
 * naming `fss_host_only_Sha256_used_in_device_code`.
 */
class Sha256 {
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
     * @throws std::runtime_error if EVP_Digest fails.
     */
  __host__ __device__ cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4> msg);

  /**
     * Hash a 128+127-bit logical message into two 32B digests.
     *
     * All 128 bits of a are message data. Bit 0 of b.w is reserved for
     * domain separation; its supplied value is ignored. The other 127 bits
     * of b are message data. Hash a followed by a copy of b with its reserved
     * bit set to 0, then to 1, and concatenate the digests in that order.
     * The input blocks are not modified.
     * Each digest hashes the configured 16B key prefix followed by the
     * two message blocks, for 48B total. This is a prefixed SHA-256 hash.
     *
     * @param msg Tuple of (a, b), with the reserved bit in the second block.
     * @return 4 int4 blocks (64B).
     * @throws std::runtime_error if context allocation or a digest operation fails or returns an invalid length.
     */
  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4> msg);

private:
  int4 key_;
};
// -----------------------------------------------------------------------------

__host__ __device__ inline cuda::std::array<int4, 2> Sha256::Hash(cuda::std::span<const int4, 4> msg) {
  cuda::std::array<int4, 2> out{};

#ifdef __CUDA_ARCH__
  fss_host_only_Sha256_used_in_device_code();
#else
  int4 buf[5] = {key_, msg[0], msg[1], msg[2], msg[3]};
  auto buf_ptr = reinterpret_cast<const unsigned char *>(buf);
  auto out_ptr = reinterpret_cast<unsigned char *>(out.data());
  int ret = EVP_Digest(buf_ptr, 80, out_ptr, NULL, EVP_sha256(), NULL);
  if (ret != 1)
    throw std::runtime_error("Sha256::Hash: EVP_Digest failed");
#endif

  return out;
}

__host__ __device__ inline cuda::std::array<int4, 4> Sha256::Hash(cuda::std::tuple<int4, const int4> msg) {
  cuda::std::array<int4, 4> out{};

#ifdef __CUDA_ARCH__
  fss_host_only_Sha256_used_in_device_code();
#else
  auto [a, b] = msg;
  int4 buf[3] = {key_, a, fss::util::SetLsb(b, false)};
  auto buf_ptr = reinterpret_cast<const unsigned char *>(buf);
  auto out_ptr = reinterpret_cast<unsigned char *>(out.data());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx)
    throw std::runtime_error("Sha256::Hash: EVP_MD_CTX_new failed");
  for (int i = 0; i < 2; ++i) {
    EVP_MD_CTX_set_flags(ctx.get(), EVP_MD_CTX_FLAG_ONESHOT);
    unsigned int digest_length = 0;
    if (EVP_DigestInit_ex(ctx.get(), i == 0 ? EVP_sha256() : NULL, NULL) != 1 ||
        EVP_DigestUpdate(ctx.get(), buf_ptr, 48) != 1 ||
        EVP_DigestFinal_ex(ctx.get(), out_ptr + i * 32, &digest_length) != 1)
      throw std::runtime_error("Sha256::Hash: EVP_Digest failed");
    if (digest_length != 32)
      throw std::runtime_error("Sha256::Hash: EVP_Digest returned a partial digest");
    if (i == 0) buf[2].w |= 1;
  }
#endif

  return out;
}

static_assert(Hashable<Sha256> && XorHashable<Sha256>);

}  // namespace fss::hash
