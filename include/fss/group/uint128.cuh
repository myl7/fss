// SPDX-License-Identifier: Apache-2.0
/**
 * @file group/uint128.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/group.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>

namespace fss::group {

/**
 * The modulus of a `Uint128` group, split into two 64-bit halves: the
 * structural form keeps host stubs from having to spell a wide integral
 * non-type template parameter.
 */
struct Uint128Modulus {
  uint64_t low;  ///< Low 64 bits of the modulus.
  uint64_t high;  ///< High 64 bits of the modulus, i.e., the modulus is $(high \ll 64) | low$.
};

/**
 * Clamped 127-bit modular integers in a 128-bit storage as a group.
 * @ingroup fss_api
 *
 * The storage is a full unsigned __int128, but elements are clamped like every
 * other group (see @ref fss_api "Groupable"), so every element is < 2^127 and
 * one bit of storage stays unused. The class is named Uint128 after the
 * storage, not the 127 usable bits.
 *
 * @tparam mod The modulus as its two 64-bit halves. It defaults to the full
 * usable range, i.e., $2^{127}$, and must satisfy $0 < mod \leq 2^{127}$.
 * There is no wrap default: unlike Uint, the modulo group addition is the only
 * mode.
 *
 * Note that if you will use these groups as fields, i.e., perform multiplication, you must set mod to a prime number so that for random $a, b$, $a \cdot b$ is uniformly distributed in the field.
 */
template <Uint128Modulus mod = Uint128Modulus{0, uint64_t{1} << 63}>
  requires(((static_cast<__uint128_t>(mod.high) << 64) | mod.low) > 0 &&
      ((static_cast<__uint128_t>(mod.high) << 64) | mod.low) <= static_cast<__uint128_t>(1) << 127)
class Uint128 {
public:
  // Groupable interface.

  /** Return the group sum. */
  __host__ __device__ Uint128 operator+(Uint128 rhs) const {
    if (val_ >= kMod - rhs.val_) return {val_ + rhs.val_ - kMod};
    else return {val_ + rhs.val_};
  }

  /** Return the additive inverse. */
  __host__ __device__ Uint128 operator-() const {
    if (val_ == 0) return {0};
    else return {kMod - val_};
  }

  /** Construct the additive identity. */
  __host__ __device__ Uint128() : val_(0) {}

  /** Convert a clamped block to a group element, reducing modulo mod when set. */
  __host__ __device__ static Uint128 From(int4 buf) {
    assert((buf.w & 1) == 0);

    __uint128_t value = static_cast<__uint128_t>(static_cast<unsigned int>(buf.x)) |
        static_cast<__uint128_t>(static_cast<unsigned int>(buf.y)) << 32 |
        static_cast<__uint128_t>(static_cast<unsigned int>(buf.z)) << 64 |
        // The LSB of buf.w is the clamped bit
        static_cast<__uint128_t>(static_cast<unsigned int>(buf.w) >> 1) << 96;

    return {value % kMod};
  }

  /** Encode this group element as a clamped block. */
  __host__ __device__ int4 Into() const {
    int4 buf = {0, 0, 0, 0};
    buf.x = static_cast<int>(val_ & 0xffffffff);
    buf.y = static_cast<int>((val_ >> 32) & 0xffffffff);
    buf.z = static_cast<int>((val_ >> 64) & 0xffffffff);
    // The LSB of buf.w is the clamped bit
    buf.w = static_cast<int>((val_ >> 96) << 1);
    return buf;
  }

  // Additional public API.
  /** Return a copy of the underlying unsigned 128-bit integer value. */
  __host__ __device__ __uint128_t Value() const { return val_; }

private:
  static constexpr __uint128_t kMod = (static_cast<__uint128_t>(mod.high) << 64) | mod.low;

  __uint128_t val_;

  __host__ __device__ Uint128(__uint128_t v) : val_(v) {}
};
static_assert(Groupable<Uint128<>>);
static_assert(Groupable<Uint128<Uint128Modulus{.low = 0xffffffffffffffff, .high = 0x3fffffff}>>);

}  // namespace fss::group
