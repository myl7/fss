// SPDX-License-Identifier: Apache-2.0
/**
 * @file hash/aes128_ccr.cuh
 * @brief AES-128 circular-correlation-robust hashing.
 */

#pragma once
#include <fss/hash.cuh>
#include <fss/prg.cuh>
#include <cuda_runtime.h>
#include <cassert>

namespace fss::prg {
template <int mul>
class Aes128Mmo;
template <int mul>
class Aes128MmoRaw;
template <int mul>
class Aes128Soft;
}  // namespace fss::prg

namespace fss::hash {

/**
 * Declared and never defined. Device code that reaches it fails to link with an
 * unresolved-extern error that names this symbol.
 */
__device__ void fss_host_only_Aes128Ccr_used_in_device_code();

/**
 * AES-128 CCR hash H(z) = AES_K(sigma(z)) xor sigma(z).
 * @ingroup fss_api
 *
 * For 64-bit halves z = (L, R), sigma(z) = (L xor R, L).
 * L is stored in int4::x and int4::y, and R in int4::z and int4::w.
 * The construction uses the linear orthomorphism from Section 7.3 of
 * https://eprint.iacr.org/2019/074.
 * The AES key K belongs to the backend and differs from HalfTree's seed S.
 *
 * @tparam AesMmo Aes128Mmo<1>, Aes128MmoRaw<1>, or Aes128Soft<1>.
 * Execution-space support, context concurrency, and table/context lifetimes
 * are those of the selected backend. Hash() is host/device for Aes128Soft<1>
 * and host-only for Aes128Mmo<1> and Aes128MmoRaw<1>, whose device use fails to
 * build with an unresolved-extern error naming
 * `fss_host_only_Aes128Ccr_used_in_device_code`.
 */
template <typename AesMmo>
  requires(std::same_as<AesMmo, prg::Aes128Mmo<1>> ||
      std::same_as<AesMmo, prg::Aes128MmoRaw<1>> || std::same_as<AesMmo, prg::Aes128Soft<1>>)
class Aes128Ccr {
public:
  /** Initialize the hash with an AES-MMO backend configured with the fixed key K. */
  __host__ __device__ explicit Aes128Ccr(AesMmo aes_mmo);

  /**
   * Return the 128-bit CCR hash of block with an OpenSSL or AES-NI backend.
   * Only for host side; device use fails to build with an unresolved-extern error
   * naming `fss_host_only_Aes128Ccr_used_in_device_code`.
   */
  __host__ __device__ int4 Hash(int4 block)
    requires(!std::same_as<AesMmo, prg::Aes128Soft<1>>);

  /** Return the 128-bit CCR hash of block on the host or device with a software AES backend. */
  __host__ __device__ int4 Hash(int4 block)
    requires std::same_as<AesMmo, prg::Aes128Soft<1>>;

private:
  AesMmo aes_mmo_;
};

// -----------------------------------------------------------------------------

template <typename AesMmo>
  requires(std::same_as<AesMmo, prg::Aes128Mmo<1>> ||
      std::same_as<AesMmo, prg::Aes128MmoRaw<1>> || std::same_as<AesMmo, prg::Aes128Soft<1>>)
__host__ __device__ inline Aes128Ccr<AesMmo>::Aes128Ccr(AesMmo aes_mmo) : aes_mmo_(aes_mmo) {}

template <typename AesMmo>
  requires(std::same_as<AesMmo, prg::Aes128Mmo<1>> ||
      std::same_as<AesMmo, prg::Aes128MmoRaw<1>> || std::same_as<AesMmo, prg::Aes128Soft<1>>)
__host__ __device__ inline int4 Aes128Ccr<AesMmo>::Hash(int4 block)
  requires(!std::same_as<AesMmo, prg::Aes128Soft<1>>) {
#ifdef __CUDA_ARCH__
  fss_host_only_Aes128Ccr_used_in_device_code();
  return {};
#else
  int4 sigma = {block.x ^ block.z, block.y ^ block.w, block.x, block.y};
  return aes_mmo_.Gen(sigma)[0];
#endif
}

template <typename AesMmo>
  requires(std::same_as<AesMmo, prg::Aes128Mmo<1>> ||
      std::same_as<AesMmo, prg::Aes128MmoRaw<1>> || std::same_as<AesMmo, prg::Aes128Soft<1>>)
__host__ __device__ inline int4 Aes128Ccr<AesMmo>::Hash(int4 block)
  requires std::same_as<AesMmo, prg::Aes128Soft<1>> {
  int4 sigma = {block.x ^ block.z, block.y ^ block.w, block.x, block.y};
  return aes_mmo_.Gen(sigma)[0];
}

}  // namespace fss::hash
