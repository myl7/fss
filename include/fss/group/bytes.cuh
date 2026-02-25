// SPDX-License-Identifier: Apache-2.0
/**
 * @file group/bytes.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * @brief Bytes with XOR as a group
 */

#pragma once
#include <fss/group.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <fss/util.cuh>

namespace fss::group {

struct Bytes {
    int4 val;

    __host__ __device__ Bytes operator+(Bytes rhs) const {
        return util::Xor(val, rhs.val);
    }

    __host__ __device__ Bytes operator-() const {
        return *this;
    }

    __host__ __device__ Bytes() : val({0, 0, 0, 0}) {}

    __host__ __device__ static Bytes From(int4 buf) {
        assert((buf.w & 1) == 0);
        return Bytes(buf);
    }

    __host__ __device__ int4 Into() const {
        return val;
    }

private:
    __host__ __device__ Bytes(int4 buf) : val(buf) {}
};
static_assert(Groupable<Bytes>);

}  // namespace fss::group
