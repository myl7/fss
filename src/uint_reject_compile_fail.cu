// SPDX-License-Identifier: Apache-2.0
// Negative compile coverage for the Uint storage and Uint128 modulus
// contracts. This file must fail to compile: each case forms one rejected
// group type and requires its completeness, so the unsatisfied requires-
// clause is the only failure source. The accepted forms compile in
// group_test.cu and group/uint128.cuh, so a failure here cannot come from a
// probe mistake on a legal type. FSS_UINT_REJECT_CASE selects one rejection
// per translation unit, matching the direct-nvcc harness.
#include <cstddef>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>

// Case: __uint128_t storage is not part of Uint.
#if defined(FSS_UINT_REJECT_UINT128_STORAGE)
using Rejected = fss::group::Uint<__uint128_t>;
// Case: the same rejection with the old common127 modulus.
#elif defined(FSS_UINT_REJECT_UINT128_STORAGE_MOD)
using Rejected = fss::group::Uint<__uint128_t, static_cast<__uint128_t>(1) << 127>;
// Case: a zero Uint128 modulus is the identity, not a group.
#elif defined(FSS_UINT_REJECT_MOD_ZERO)
using Rejected = fss::group::Uint128<fss::group::Uint128Modulus{.low = 0, .high = 0}>;
// Case: one above 2^127 exceeds the clamped element range.
#elif defined(FSS_UINT_REJECT_MOD_ABOVE_2_127)
using Rejected = fss::group::Uint128<fss::group::Uint128Modulus{.low = 0, .high = (uint64_t{1} << 63) | 1}>;
#else
#error "no FSS_UINT_REJECT_* case selected"
#endif

static_assert(sizeof(Rejected) > 0, "the rejected group type must not form");
