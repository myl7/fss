// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace fss::python {

template <int depth, int frontier_limit = depth, bool merkle = false>
constexpr bool ValidCoreGpuLaunch() {
  constexpr int frontier = kFrontierDepth < frontier_limit ? kFrontierDepth : frontier_limit;
  constexpr int requested_log2 = [] {
    int result = 0;
    for (int threads = kThreadsPerBlock; threads > 1; threads >>= 1) ++result;
    return result;
  }();
  constexpr int block_log2 = requested_log2 < frontier ? requested_log2 : frontier;
  constexpr int roots_log2 = frontier - block_log2;
  return depth >= 0 && frontier_limit >= 0 && kFrontierDepth >= 0 &&
      kThreadsPerBlock >= 1 && kThreadsPerBlock <= 1024 &&
      (kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0 &&
      depth - frontier <= 8 && roots_log2 <= 30 &&
      (!merkle || roots_log2 <= block_log2);
}

}  // namespace fss::python
