// SPDX-License-Identifier: Apache-2.0
/**
 * @file host_only_device_use_test.cu
 *
 * @brief Build-time check that device use of a host-only backend fails to link.
 *
 * The host_only_* CMake tests compile this file with nvcc through
 * cmake/HostOnlyDeviceUseCheck.cmake. It is never run. Define at most one of:
 * - FSS_HOST_ONLY_DEVICE_PERMU: a kernel calls Aes128Feistel::Permu.
 * - FSS_HOST_ONLY_DEVICE_COPY: a kernel copy-constructs and destroys an Aes128Feistel.
 * With neither, the file is the positive control: the same host use, no device use.
 * It must build without nvcc #20011-D or #20014-D.
 */

#include <fss/prp/aes128_feistel.cuh>
#include <cstdio>

namespace host_only_probe {

// Stands for a scheme method: `__host__ __device__`, copies the backend, and calls Permu.
template <typename Prp>
__host__ __device__ __uint128_t SchemeLikeEval(const Prp &prp, int4 seed, __uint128_t x) {
  Prp copy = prp;
  return copy.Permu(seed, x, 100);
}

__global__ void DeviceOkKernel(__uint128_t *out) {
  out[0] = 1;
}

#if defined(FSS_HOST_ONLY_DEVICE_PERMU)
__global__ void HostOnlyKernel(fss::prp::Aes128Feistel *prp, __uint128_t *out) {
  out[0] = prp->Permu(int4{}, 1, 100);
}
#elif defined(FSS_HOST_ONLY_DEVICE_COPY)
__global__ void HostOnlyKernel(fss::prp::Aes128Feistel *prp, __uint128_t *out) {
  fss::prp::Aes128Feistel copy = *prp;
  out[0] = 1;
}
#endif

}  // namespace host_only_probe

int main() {
  fss::prp::Aes128Feistel prp;
  auto y = host_only_probe::SchemeLikeEval(prp, int4{}, 1);
  std::printf("%d\n", static_cast<int>(y < 100));
  (void)&host_only_probe::DeviceOkKernel;
#if defined(FSS_HOST_ONLY_DEVICE_PERMU) || defined(FSS_HOST_ONLY_DEVICE_COPY)
  (void)&host_only_probe::HostOnlyKernel;
#endif
  return 0;
}
