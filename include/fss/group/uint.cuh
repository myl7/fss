// SPDX-License-Identifier: Apache-2.0
/**
 * @file group/uint.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/group.cuh>
#include <fss/compat.cuh>
#include <fss/group/uint128.cuh>
#include <cuda_runtime.h>
#include <type_traits>
#include <cassert>
#include <cstdint>

namespace fss::group {

/**
 * Unsigned integers with arithmetic addition and optional modulo as a group.
 * @ingroup fss_api
 *
 * @tparam T Type to store the value. From uint8_t to uint64_t.
 * @tparam mod Modulus, making the group addition defined as $a + b \mod mod$.
 * If 0, use $a + b$ as the group addition, i.e., same as $mod = 2^{sizeof(T) * 8}$.
 *
 * Note that if you will use these groups as fields, i.e., perform multiplication, you must set mod to a prime number so that for random $a, b$, $a \cdot b$ is uniformly distributed in the field.
 */
#if FSS_ENABLE_V1_COMPAT
// Under the v1 compatibility switch the primary also admits __uint128_t with
// a uint64-range positive modulus so the deprecated partial specialization
// below is reachable: naming Uint<__uint128_t, ...> checks this requires-
// clause even when the specialization wins partial ordering. Every other
// __uint128_t modulus still fails the clause, and the specialization covers
// the whole admitted __uint128_t domain, so the primary body below never
// instantiates for __uint128_t and keeps no wide arithmetic.
template <typename T, T mod = 0>
  requires((std::is_unsigned_v<T> && sizeof(T) <= 8) ||
      (std::is_same_v<T, __uint128_t> && mod > 0 && mod <= static_cast<__uint128_t>(UINT64_MAX)))
class Uint {
#else
template <typename T, T mod = 0>
  requires(std::is_unsigned_v<T> && sizeof(T) <= 8)
class Uint {
#endif
public:
  // Groupable interface.

  /** Return the group sum. */
  __host__ __device__ Uint operator+(Uint rhs) const {
    if constexpr (mod == 0) return {static_cast<T>(val_ + rhs.val_)};

    if (val_ >= mod - rhs.val_) return {static_cast<T>(val_ + rhs.val_ - mod)};
    else return {static_cast<T>(val_ + rhs.val_)};
  }

  /** Return the additive inverse. */
  __host__ __device__ Uint operator-() const {
    if constexpr (mod == 0) return {static_cast<T>(-val_)};

    if (val_ == 0) return {0};
    else return {static_cast<T>(mod - val_)};
  }

  /** Construct the additive identity. */
  __host__ __device__ Uint() : val_(0) {}

  /** Convert a clamped block to a group element, reducing modulo mod when set. */
  __host__ __device__ static Uint From(int4 buf) {
    assert((buf.w & 1) == 0);

    T value = 0;
    if constexpr (sizeof(T) < 4) value = buf.x & ((1 << 8 * sizeof(T)) - 1);
    // Cast to unsigned int first to prevent sign extension when promoting to larger types
    else if constexpr (sizeof(T) == 4) value = static_cast<unsigned int>(buf.x);
    else if constexpr (sizeof(T) == 8)
      value = static_cast<T>(static_cast<unsigned int>(buf.x)) | static_cast<T>(static_cast<unsigned int>(buf.y)) << 32;
    else __builtin_unreachable();

    if constexpr (mod > 0) value %= mod;

    return {value};
  }

  /** Encode this group element as a clamped block. */
  __host__ __device__ int4 Into() const {
    int4 buf = {0, 0, 0, 0};
    if constexpr (sizeof(T) <= 4) buf.x = static_cast<int>(val_);
    else if constexpr (sizeof(T) == 8) {
      buf.x = static_cast<int>(val_ & 0xffffffff);
      buf.y = static_cast<int>(val_ >> 32);
    } else __builtin_unreachable();
    return buf;
  }

  // Additional public API.
  // Temporary access for the host DCF full-domain fast path. Revisit after
  // comparing it with the generic Groupable implementation.
  /** Return a copy of the underlying unsigned integer value. */
  __host__ __device__ T Value() const { return val_; }

private:
  T val_;

  __host__ __device__ Uint(T v) : val_(v) {}
};
static_assert(Groupable<Uint<uint8_t>>);
static_assert(Groupable<Uint<uint16_t>>);
static_assert(Groupable<Uint<uint32_t>>);
static_assert(Groupable<Uint<uint64_t>>);

#if FSS_ENABLE_V1_COMPAT
/**
 * The v1.3.0 wide-storage Uint, restored as a deprecated thin wrapper.
 * @ingroup fss_api
 *
 * Only $0 < mod \leq 2^{64} - 1$ is restored, forwarding every operation to
 * @ref Uint128 with the same modulus. Wider moduli stay rejected: on the
 * probed CUDA 12.6/GCC 11.4 and 13.4/GCC 13.3 toolchains the host-stub
 * spelling of the two probed wider moduli misparsed as zero, other wider
 * values could silently misparse differently, and wide-form kernel launches
 * are therefore not compile-safe above the uint64 range.
 *
 * @tparam mod The modulus. Must satisfy $0 < mod \leq 2^{64} - 1$; there is
 * no wrap default, unlike the small-type primary.
 */
// Directly naming this class with a modulus at or above 2^63 makes the nvcc
// host pass spell the template argument as an unsuffixed decimal. The probed
// CUDA 12.6/GCC 11.4 host pass warns about that literal while still parsing
// the value correctly. The warning is informational; do not suppress it.
template <__uint128_t mod>
  requires(mod > 0 && mod <= static_cast<__uint128_t>(UINT64_MAX))
class [[deprecated("use fss::group::Uint128<fss::group::Uint128Modulus{.low = mod, .high = 0}> instead")]]
    Uint<__uint128_t, mod> {
public:
  // Groupable interface.

  /** Return the group sum. */
  __host__ __device__ Uint operator+(Uint rhs) const { return Uint(impl_ + rhs.impl_); }

  /** Return the additive inverse. */
  __host__ __device__ Uint operator-() const { return Uint(-impl_); }

  /** Construct the additive identity. */
  __host__ __device__ Uint() : impl_() {}

  /** Convert a clamped block to a group element, reducing modulo mod when set. */
  __host__ __device__ static Uint From(int4 buf) { return Uint(Underlying::From(buf)); }

  /** Encode this group element as a clamped block. */
  __host__ __device__ int4 Into() const { return impl_.Into(); }

  // Additional public API.
  /** Return a copy of the underlying unsigned 128-bit integer value. */
  __host__ __device__ __uint128_t Value() const { return impl_.Value(); }

private:
  using Underlying = Uint128<Uint128Modulus{static_cast<uint64_t>(mod), 0}>;

  Underlying impl_;

  __host__ __device__ explicit Uint(Underlying impl) : impl_(impl) {}
};
#endif  // FSS_ENABLE_V1_COMPAT

}  // namespace fss::group
