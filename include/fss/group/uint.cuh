// SPDX-License-Identifier: Apache-2.0
/**
 * @file group/uint.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 */

#pragma once
#include <fss/group.cuh>
#include <cuda_runtime.h>
#include <type_traits>
#include <cassert>

namespace fss::group {

/**
 * Unsigned integers with arithmetic addition and optional modulo as a group.
 *
 * @tparam T Type to store the value. From uint8_t to __uint128_t.
 * @tparam mod Modulus, making the group addition defined as $a + b \mod mod$.
 * If 0, use $a + b$ as the group addition, i.e., same as $mod = 2^{sizeof(T) * 8}$.
 *
 * When `T = __uint128_t`, because elements are clamped, any element has < 2^127 and 0 < `mod` <= 2^127.
 *
 * When `T = __uint128_t` and `mod` >= 2^64, the compiler would output warnings like "warning: integer constant is so large that it is unsigned".
 * These warnings cannot be easily suppressed (I tried any in-code method I can find out), but they are harmless and can be ignored.
 *
 * Note that if you will use these groups as fields, i.e., perform multiplication, you must set `mod` to a prime number so that for random $a, b$, $a \cdot b$ is uniformly distributed in the field.
 */
template <typename T, T mod = 0>
    requires((std::is_unsigned_v<T> || std::is_same_v<T, __uint128_t>) && sizeof(T) <= 16 &&
        (sizeof(T) < 16 || (mod > 0 && mod <= static_cast<T>(1) << 127)))
struct Uint {
    T val;

    __host__ __device__ Uint operator+(Uint rhs) const {
        if constexpr (mod == 0) return {static_cast<T>(val + rhs.val)};

        if (val >= mod - rhs.val) return {static_cast<T>(val + rhs.val - mod)};
        else return {static_cast<T>(val + rhs.val)};
    }

    __host__ __device__ Uint operator-() const {
        if constexpr (mod == 0) return {static_cast<T>(-val)};

        if (val == 0) return {0};
        else return {static_cast<T>(mod - val)};
    }

    __host__ __device__ Uint() : val(0) {}

    __host__ __device__ static Uint From(int4 buf) {
        assert((buf.w & 1) == 0);

        T val = 0;
        if constexpr (sizeof(T) < 4) val = buf.x & ((1 << 8 * sizeof(T)) - 1);
        // Cast to unsigned int first to prevent sign extension when promoting to larger types
        else if constexpr (sizeof(T) == 4) val = static_cast<unsigned int>(buf.x);
        else if constexpr (sizeof(T) == 8)
            val = static_cast<T>(static_cast<unsigned int>(buf.x)) |
                static_cast<T>(static_cast<unsigned int>(buf.y)) << 32;
        else if constexpr (sizeof(T) == 16)
            val = static_cast<T>(static_cast<unsigned int>(buf.x)) |
                static_cast<T>(static_cast<unsigned int>(buf.y)) << 32 |
                static_cast<T>(static_cast<unsigned int>(buf.z)) << 64 |
                // For uint128, LSB of buf.w is the clamped bit
                static_cast<T>(static_cast<unsigned int>(buf.w) >> 1) << 96;
        else __builtin_unreachable();

        if constexpr (mod > 0) val %= mod;

        return {val};
    }

    __host__ __device__ int4 Into() const {
        int4 buf = {0, 0, 0, 0};
        if constexpr (sizeof(T) <= 4) buf.x = static_cast<int>(val);
        else if constexpr (sizeof(T) == 8) {
            buf.x = static_cast<int>(val & 0xffffffff);
            buf.y = static_cast<int>(val >> 32);
        } else if constexpr (sizeof(T) == 16) {
            buf.x = static_cast<int>(val & 0xffffffff);
            buf.y = static_cast<int>((val >> 32) & 0xffffffff);
            buf.z = static_cast<int>((val >> 64) & 0xffffffff);
            // For uint128, its LSB is the clamped bit
            buf.w = static_cast<int>((val >> 96) << 1);
        } else __builtin_unreachable();
        return buf;
    }

private:
    __host__ __device__ Uint(T v) : val(v) {}
};
static_assert(Groupable<Uint<uint8_t>>);
static_assert(Groupable<Uint<uint16_t>>);
static_assert(Groupable<Uint<uint32_t>>);
static_assert(Groupable<Uint<uint64_t>>);
static_assert(Groupable<Uint<__uint128_t, static_cast<__uint128_t>(1) << 127>>);

}  // namespace fss::group
