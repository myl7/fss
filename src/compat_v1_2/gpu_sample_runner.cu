// SPDX-License-Identifier: Apache-2.0
// The included consumer is an unchanged v1.2.0 sample.
#define main FssLegacySampleMain
#include "dpf_dcf_gpu.cu"
#undef main

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  return FssLegacySampleMain();
}
