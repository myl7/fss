// SPDX-License-Identifier: Apache-2.0
// Positive compile coverage for the deprecated v1.3.0 wide-storage Uint
// compatibility entry. FSS_UINT_COMPAT_CHECK selects one mode per translation
// unit, matching the direct-nvcc harness. The DEPRECATED mode names the legacy
// type through direct spellings only: namespace-scope aliases do not surface
// the diagnostic on the nvcc host pass, so an alias would silently weaken the
// check. The MODERN mode uses only current types and must stay free of any
// deprecation diagnostic, with the host warning promoted to an error.
#include <cuda_runtime.h>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>

#if defined(FSS_UINT_COMPAT_CHECK_DEPRECATED)

// The direct spelling in a kernel template argument is what the nvcc host
// stub must spell, so it carries the widest compatibility weight here.
template <typename Group>
__global__ void CompatProbeKernel(int4 *ys, const int4 *xs, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  Group v = Group::From(xs[tid]);
  ys[tid] = (v + (-v) + v).Into();
}

void CompatProbeLaunch(int4 *ys, const int4 *xs, int n) {
  CompatProbeKernel<fss::group::Uint<__uint128_t, UINT64_MAX>><<<1, 64>>>(ys, xs, n);
  CompatProbeKernel<fss::group::Uint<__uint128_t, 17>><<<1, 64>>>(ys, xs, n);
}

int CompatProbeHost() {
  fss::group::Uint<__uint128_t, 17> g{};
  g = g + (-g);
  return static_cast<int>(g.Into().x) + static_cast<int>(g.Value());
}

#elif defined(FSS_UINT_COMPAT_CHECK_MODERN)

template <typename Group>
__global__ void ModernProbeKernel(int4 *ys, const int4 *xs, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  Group v = Group::From(xs[tid]);
  ys[tid] = (v + (-v) + v).Into();
}

void ModernProbeLaunch(int4 *ys, const int4 *xs, int n) {
  ModernProbeKernel<fss::group::Uint<uint32_t, 2147483659U>><<<1, 64>>>(ys, xs, n);
  ModernProbeKernel<fss::group::Uint128<>><<<1, 64>>>(ys, xs, n);
  ModernProbeKernel<fss::group::Uint128<fss::group::Uint128Modulus{17, 0}>><<<1, 64>>>(ys, xs, n);
}

int ModernProbeHost() {
  fss::group::Uint<uint8_t> a{};
  fss::group::Uint<uint16_t, 17> b{};
  fss::group::Uint<uint32_t, 5> c{};
  fss::group::Uint<uint64_t, 3> d{};
  fss::group::Uint128<> e{};
  fss::group::Uint128<fss::group::Uint128Modulus{17, 0}> f{};
  a = a + a;
  b = -b;
  c = c + c;
  d = -d;
  e = e + e;
  f = -f;
  return a.Into().x + b.Into().x + c.Into().x + d.Into().x + e.Into().x + f.Into().x;
}

#else
#error "no FSS_UINT_COMPAT_CHECK_* mode selected"
#endif
