// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace fss::gpu::detail {

// Identity launch descriptor for schemes without a safe family descriptor.
// It exposes `S` unchanged, so it does not hide a wide Group non-type
// template parameter.
template <typename S>
struct CoreLaunchIdentityDescriptor {
  using Scheme = S;
};

}  // namespace fss::gpu::detail
