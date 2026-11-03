// SPDX-License-Identifier: Apache-2.0
// Compile-only regression for the wide Uint127 Group in CUDA host stubs.
// Each function instantiates a fixed public GPU entry point; no main() and no
// GPU work. Built as an object target under BUILD_TESTING.

// SPDX-License-Identifier: Apache-2.0


#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/std/span>
#include <fss/dcf.cuh>
#include <fss/dmpf.cuh>
#include <fss/dpf.cuh>
#include <fss/group/uint.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/chacha_feistel.cuh>
#include <fss/vdpf.cuh>
#include <fss/vdmpf.cuh>

namespace probe {

// The wide Group under audit.
using Uint127 = fss::group::Uint<__uint128_t, static_cast<__uint128_t>(1) << 127>;

// Minimal CCR hash. Probes only need the expression shape for constraints.
struct CcrHash {
  __host__ __device__ int4 Hash(int4 block) const { return block; }
};

using DpfPrg = fss::prg::ChaCha<2>;
using DcfPrg = fss::prg::ChaCha<4>;
using Prp = fss::prp::ChaChaFeistel;
using Hash = fss::hash::Blake3;

using DpfScheme = fss::Dpf<8, Uint127, DpfPrg, uint32_t>;
using DcfScheme = fss::Dcf<8, Uint127, DcfPrg, uint32_t, fss::DcfPred::kLt>;
using HalfTreeScheme = fss::HalfTreeDpf<8, Uint127, CcrHash, uint32_t>;
using VdpfScheme = fss::Vdpf<8, Uint127, DpfPrg, Hash, Hash, uint32_t>;
using DmpfScheme = fss::Dmpf<8, 30, 4, Uint127, DpfPrg, Prp, uint32_t>;
using VdmpfScheme = fss::Vdmpf<8, 30, 4, Uint127, DpfPrg, Hash, Hash, Prp, uint16_t>;

}  // namespace probe

namespace {

void ProbeDpfEvalAllGpu(const probe::DpfScheme &scheme, const probe::DpfScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const probe::DpfScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeDpfEvalAllGpuBatch(const probe::DpfScheme &scheme, const int4 *s0s, const probe::DpfScheme::Cw *cws,
    int4 *ys) {
  (void)scheme.template EvalAllGpuBatch<8, 256>(false, cuda::std::span<const int4>(s0s, 1),
      cuda::std::span<const probe::DpfScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeDpfEvalRangeGpu(const probe::DpfScheme &scheme, const probe::DpfScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalRangeGpu<8, 256>(false, uint32_t{0}, uint32_t{255}, int4{0, 0, 0, 0},
      cuda::std::span<const probe::DpfScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeDpfEvalPointGpu(const probe::DpfScheme &scheme, const int4 *seeds, const int4 *cws, const uint32_t *tr,
    const uint32_t *xs, int4 *ys) {
  fss::gpu::detail::DpfEvalPointGpu<4>(false, cuda::std::span<const int4>(seeds, 1),
      cuda::std::span<const int4>(cws, 9), cuda::std::span<const uint32_t>(tr, 1),
      cuda::std::span<const uint32_t>(xs, 1), ys, 1, scheme);
}

void ProbeDcfEvalAllGpu(const probe::DcfScheme &scheme, const probe::DcfScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const probe::DcfScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeDcfEvalAllGpuBatch(const probe::DcfScheme &scheme, const int4 *s0s, const probe::DcfScheme::Cw *cws,
    int4 *ys) {
  (void)scheme.template EvalAllGpuBatch<8, 256>(false, cuda::std::span<const int4>(s0s, 1),
      cuda::std::span<const probe::DcfScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeDcfEvalRangeGpu(const probe::DcfScheme &scheme, const probe::DcfScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalRangeGpu<8, 256>(false, uint32_t{0}, uint32_t{255}, int4{0, 0, 0, 0},
      cuda::std::span<const probe::DcfScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeDcfEvalPointGpu(const probe::DcfScheme &scheme, const int4 *seeds, const int4 *cws, const int4 *vws,
    const uint32_t *tr, const uint32_t *xs, int4 *ys) {
  fss::gpu::detail::DcfEvalPointGpu<4>(false, cuda::std::span<const int4>(seeds, 1),
      cuda::std::span<const int4>(cws, 9), cuda::std::span<const int4>(vws, 9),
      cuda::std::span<const uint32_t>(tr, 1), cuda::std::span<const uint32_t>(xs, 1), ys, 1, scheme);
}

void ProbeHalfTreeEvalAllGpu(const probe::HalfTreeScheme &scheme, const probe::HalfTreeScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const probe::HalfTreeScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeHalfTreeEvalAllGpuBatch(const probe::HalfTreeScheme &scheme, const int4 *s0s,
    const probe::HalfTreeScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalAllGpuBatch<8, 256>(false, cuda::std::span<const int4>(s0s, 1),
      cuda::std::span<const probe::HalfTreeScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeHalfTreeEvalRangeGpu(const probe::HalfTreeScheme &scheme, const probe::HalfTreeScheme::Cw *cws, int4 *ys) {
  (void)scheme.template EvalRangeGpu<8, 256>(false, uint32_t{0}, uint32_t{255}, int4{0, 0, 0, 0},
      cuda::std::span<const probe::HalfTreeScheme::Cw>(cws, 9), cuda::std::span<int4>(ys, 256));
}

void ProbeHalfTreeEvalPointGpu(const probe::HalfTreeScheme &scheme, const int4 *seeds, const int4 *cws,
    const uint32_t *tr, const uint32_t *xs, int4 *ys) {
  fss::gpu::detail::HalfTreeDpfEvalPointGpu<4>(false, cuda::std::span<const int4>(seeds, 1),
      cuda::std::span<const int4>(cws, 9), cuda::std::span<const uint32_t>(tr, 1),
      cuda::std::span<const uint32_t>(xs, 1), ys, 1, scheme);
}

void ProbeVdpfEvalAllMerkleGpu(const probe::VdpfScheme &scheme, const probe::VdpfScheme::Cw *cws, const int4 *cs,
    int4 *ys, int4 *pi) {
  (void)scheme.template EvalAllMerkleGpu<8, 256>(false, int4{0, 0, 0, 0},
      cuda::std::span<const probe::VdpfScheme::Cw>(cws, 9), cuda::std::span<const int4, 4>(cs, 4),
      cuda::std::span<int4>(ys, 256), cuda::std::span<int4, 4>(pi, 4));
}

void ProbeVdpfEvalRangeMerkleGpu(const probe::VdpfScheme &scheme, const probe::VdpfScheme::Cw *cws, const int4 *cs,
    int4 *ys, int4 *pi) {
  (void)scheme.template EvalRangeMerkleGpu<8, 256>(false, uint32_t{0}, uint32_t{255}, int4{0, 0, 0, 0},
      cuda::std::span<const probe::VdpfScheme::Cw>(cws, 9), cuda::std::span<const int4, 4>(cs, 4),
      cuda::std::span<int4>(ys, 256), cuda::std::span<int4, 4>(pi, 4));
}

void ProbeVdpfEvalPointGpu(const probe::VdpfScheme &scheme, const int4 *seeds, const probe::VdpfScheme::Cw *cws,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws, const uint32_t *xs, int4 *ys,
    cuda::std::array<int4, 4> *pi) {
  fss::gpu::detail::VdpfEvalPointGpu<4>(false, cuda::std::span<const int4>(seeds, 1),
      cuda::std::span<const probe::VdpfScheme::Cw>(cws, 9),
      cuda::std::span<const cuda::std::array<int4, 4>>(cs, 1), cuda::std::span<const int4>(ocws, 1),
      cuda::std::span<const uint32_t>(xs, 1), ys, cuda::std::span<cuda::std::array<int4, 4>>(pi, 1), 1, scheme);
}

void ProbeDmpfEvalAllGpu(const probe::DmpfScheme &scheme, const int4 *s0s, const probe::DmpfScheme::Cw *cws,
    int4 *ys) {
  (void)scheme.template EvalAllGpu<8, 256>(false, int4{0, 0, 0, 0}, __uint128_t{1},
      cuda::std::span<const int4>(s0s, 1), cuda::std::span<const probe::DmpfScheme::Cw>(cws, 5),
      cuda::std::span<int4>(ys, 256));
}

void ProbeDmpfBatchEvalGpu(const probe::DmpfScheme &scheme, const uint32_t *xs, int4 *ys, const int4 *s0s,
    const probe::DmpfScheme::Cw *cws) {
  fss::gpu::detail::DmpfBatchEvalGpu<4>(false, cuda::std::span<const uint32_t>(xs, 1), ys, 1,
      int4{0, 0, 0, 0}, 1, __uint128_t{1}, cuda::std::span<const int4>(s0s, 1),
      cuda::std::span<const probe::DmpfScheme::Cw>(cws, 5), scheme);
}

void ProbeVdmpfEvalAllMerkleGpu(const probe::VdmpfScheme &scheme, const int4 *s0s, const probe::VdmpfScheme::Cw *cws,
    const int4 *cs, int4 *ys, int4 *pi) {
  (void)scheme.template EvalAllMerkleGpu<256>(false, int4{0, 0, 0, 0}, __uint128_t{1},
      cuda::std::span<const int4>(s0s, 1), cuda::std::span<const probe::VdmpfScheme::Cw>(cws, 5),
      cuda::std::span<const int4>(cs, 4), cuda::std::span<int4>(ys, 256),
      cuda::std::span<int4, 4>(pi, 4));
}

void ProbeVdmpfBatchEvalMerkleGpu(const probe::VdmpfScheme &scheme, const uint16_t *xs, int4 *ys,
    cuda::std::array<int4, 4> *pi, const int4 *s0s, const probe::VdmpfScheme::Cw *cws,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws) {
  fss::gpu::detail::VdmpfBatchEvalMerkleGpu<4>(false, cuda::std::span<const uint16_t>(xs, 1), ys, pi, 1,
      int4{0, 0, 0, 0}, 1, __uint128_t{1}, cuda::std::span<const int4>(s0s, 1),
      cuda::std::span<const probe::VdmpfScheme::Cw>(cws, 5),
      cuda::std::span<const cuda::std::array<int4, 4>>(cs, 1), cuda::std::span<const int4>(ocws, 1), scheme);
}

}  // namespace
