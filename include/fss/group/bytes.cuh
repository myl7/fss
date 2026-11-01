// SPDX-License-Identifier: Apache-2.0
/**
 * @file group/bytes.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/group.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <fss/util.cuh>

namespace fss::group {

/**
 * Bytes with XOR as a group.
 * @ingroup fss_api
 */
class Bytes {
public:
  // Groupable interface.

  /** Return the group sum, the bitwise XOR of both values. */
  __host__ __device__ Bytes operator+(Bytes rhs) const {
    return util::Xor(val_, rhs.val_);
  }

  /** Return the additive inverse, which is this value in the XOR group. */
  __host__ __device__ Bytes operator-() const {
    return *this;
  }

  /** Construct the additive identity. */
  __host__ __device__ Bytes() : val_({0, 0, 0, 0}) {}

  /** Wrap a clamped block as a group element. */
  __host__ __device__ static Bytes From(int4 buf) {
    assert((buf.w & 1) == 0);
    return Bytes(buf);
  }

  /** Return this group element as a clamped block. */
  __host__ __device__ int4 Into() const {
    return val_;
  }

  // Additional public API.
  // Temporary access for the host DCF full-domain fast path. Revisit after
  // comparing it with the generic Groupable implementation.
  /** Return a copy of the underlying block. */
  __host__ __device__ int4 Value() const { return val_; }

private:
  int4 val_;

  __host__ __device__ Bytes(int4 buf) : val_(buf) {}
};
static_assert(Groupable<Bytes>);

}  // namespace fss::group
