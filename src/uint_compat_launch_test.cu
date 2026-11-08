// SPDX-License-Identifier: Apache-2.0
// Compile-only regression for the deprecated v1.3.0 wide-storage Uint
// compatibility Group in CUDA host stubs. The legacy group carries its wide
// non-type template parameter into the canonical typename of every kernel
// template argument below, which is exactly the spelling the stubs must
// reproduce, so a plain wrapper struct would weaken the check. Moduli span
// the restored range from small to the largest below 2^64. Each function
// instantiates a fixed public GPU entry point; no main() and no GPU work.
// Built as an object target under BUILD_TESTING when FSS_ENABLE_V1_COMPAT
// is on.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/std/span>
#include <fss/compat.cuh>

#if FSS_ENABLE_V1_COMPAT

#include <fss/dcf.cuh>
#include <fss/dpf.cuh>
#include <fss/group/uint.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/prg/chacha.cuh>

namespace probe {

using Legacy17 = fss::group::Uint<__uint128_t, 17>;
using LegacyCross = fss::group::Uint<__uint128_t, (uint64_t{1} << 32) + 15>;
using LegacyPow63m1 = fss::group::Uint<__uint128_t, (uint64_t{1} << 63) - 1>;
using LegacyMaxM58 = fss::group::Uint<__uint128_t, UINT64_MAX - 58>;

// Minimal CCR hash. Probes only need the expression shape for constraints.
struct CcrHash {
  __host__ __device__ int4 Hash(int4 block) const { return block; }
};

using DpfPrg = fss::prg::ChaCha<2>;
using DcfPrg = fss::prg::ChaCha<4>;

using Dpf17 = fss::Dpf<8, Legacy17, DpfPrg, uint32_t>;
using DpfMaxM58 = fss::Dpf<8, LegacyMaxM58, DpfPrg, uint32_t>;
using DcfCross = fss::Dcf<8, LegacyCross, DcfPrg, uint32_t, fss::DcfPred::kLt>;
using HalfTreePow63m1 = fss::HalfTreeDpf<8, LegacyPow63m1, CcrHash, uint32_t>;

}  // namespace probe

namespace {

void ProbeDpfEvalAllGpu(const probe::Dpf17 &scheme, const int4 *cws, const uint32_t *tr, int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const int4>(cws, 9), tr[0], cuda::std::span<int4>(ys, 256));
}

void ProbeDpfEvalAllGpuMaxM58(const probe::DpfMaxM58 &scheme, const int4 *cws, const uint32_t *tr, int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const int4>(cws, 9), tr[0], cuda::std::span<int4>(ys, 256));
}

void ProbeDcfEvalAllGpu(const probe::DcfCross &scheme, const int4 *cws, const int4 *vws,
    const uint32_t *tr, int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const int4>(cws, 9), cuda::std::span<const int4>(vws, 9), tr[0],
      cuda::std::span<int4>(ys, 256));
}

void ProbeHalfTreeEvalAllGpu(const probe::HalfTreePow63m1 &scheme, const int4 *cws, const uint32_t *tr,
    int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const int4>(cws, 9), tr[0], cuda::std::span<int4>(ys, 256));
}

void ProbeDpfEvalPointGpu(const probe::Dpf17 &scheme, const int4 *seeds, const int4 *cws,
    const uint32_t *tr, const uint32_t *xs, int4 *ys) {
  fss::gpu::detail::DpfEvalPointGpu<4>(false, cuda::std::span<const int4>(seeds, 1),
      cuda::std::span<const int4>(cws, 9), cuda::std::span<const uint32_t>(tr, 1),
      cuda::std::span<const uint32_t>(xs, 1), ys, 1, scheme);
}

void ProbeDcfEvalPointGpu(const probe::DcfCross &scheme, const int4 *seeds, const int4 *cws,
    const int4 *vws, const uint32_t *tr, const uint32_t *xs, int4 *ys) {
  fss::gpu::detail::DcfEvalPointGpu<4>(false, cuda::std::span<const int4>(seeds, 1),
      cuda::std::span<const int4>(cws, 9), cuda::std::span<const int4>(vws, 9),
      cuda::std::span<const uint32_t>(tr, 1), cuda::std::span<const uint32_t>(xs, 1), ys, 1, scheme);
}

}  // namespace

#endif  // FSS_ENABLE_V1_COMPAT
