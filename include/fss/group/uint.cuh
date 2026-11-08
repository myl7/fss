// SPDX-License-Identifier: Apache-2.0
/**
 * @file group/uint.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/group.cuh>
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
template <typename T, T mod = 0>
  requires(std::is_unsigned_v<T> && sizeof(T) <= 8)
class Uint {
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

}  // namespace fss::group
