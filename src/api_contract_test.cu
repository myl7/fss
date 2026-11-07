// SPDX-License-Identifier: Apache-2.0
// Compile-time public API contract of the stride-only correction-word
// interfaces. The paired Cw overloads are gone, so their old argument
// sequences must not resolve; the component forms must stay callable with
// their defaults (single key stride 1, batch dense level-major stride 0),
// with an explicit non-unit stride, and with an explicit batch stride of 0.
// The multi-point bucket interfaces and the parity-tree view entries were
// never paired with a stride form and must remain callable unchanged.
#include <gtest/gtest.h>
#include <cuda/std/span>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/hash/aes128_ccr.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prp/chacha_feistel.cuh>

namespace {

using Bytes = fss::group::Bytes;
using Prg2 = fss::prg::ChaCha<2>;
using Prg4 = fss::prg::ChaCha<4>;
using Ccr = fss::hash::Aes128Ccr<fss::prg::Aes128Mmo<1>>;
using Blake3 = fss::hash::Blake3;
using Prp = fss::prp::ChaChaFeistel;

using Dpf8 = fss::Dpf<8, Bytes, Prg2, uint16_t>;
using Dcf8 = fss::Dcf<8, Bytes, Prg4, uint16_t, fss::DcfPred::kLt>;
using Half8 = fss::HalfTreeDpf<8, Bytes, Ccr, uint16_t>;
using Packed8 = fss::PackedHalfTreeDpf<8, 8, Ccr, uint16_t>;
using Grotto8 = fss::GrottoDcf<8, Prg2, uint16_t>;
using HtGrotto8 = fss::HalfTreeGrottoDcf<8, Ccr, uint16_t>;
using Vdpf8 = fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3, uint16_t>;
using Dmpf8 = fss::Dmpf<8, 30, 4, Bytes, Prg2, Prp, uint16_t>;
using Vdmpf8 = fss::Vdmpf<8, 30, 4, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>;

// The removed Cw argument sequences are checked by
// src/api_removed_cw_compile_fail.cu, which must fail to compile; the
// cmake/ApiRemovedCwCheck.cmake harness asserts that failure. Some nvcc
// versions report overload failures inside requires-expressions as hard
// errors, so the negative side cannot live in static_asserts here.
int4 g_dense[9] = {};
int4 g_values[9] = {};
int4 g_wide[65] = {};
int4 g_seed_pair[2] = {};
int4 g_check[4] = {};
uint16_t g_trs[2] = {};
int4 g_ys[256] = {};
bool g_bits[256] = {};

// DPF: the Cw forms of Gen/Eval/EvalAll/EvalRange and the key-major Cw batch
// launch are removed; the component forms keep their defaults and strides.
static_assert(requires(Dpf8 &s, uint16_t &tr, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_dense, 9), tr, cuda::std::span<const int4, 2>(g_seed_pair), in,
      int4{});
});
static_assert(requires(Dpf8 &s, uint16_t &tr, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_wide, 65), tr, cuda::std::span<const int4, 2>(g_seed_pair), in,
      int4{}, 8);
});
static_assert(requires(Dpf8 &s, bool b, int4 s0, uint16_t tr, uint16_t x) {
  s.Eval(b, s0, cuda::std::span<const int4>(g_dense, 9), tr, x);
});
static_assert(requires(Dpf8 &s, bool b, int4 s0, uint16_t tr) {
  s.EvalAll(b, s0, cuda::std::span<const int4>(g_dense, 9), tr, g_ys);
});
static_assert(requires(Dpf8 &s, bool b, uint16_t lo, uint16_t hi, int4 s0, uint16_t tr) {
  s.EvalRange(b, lo, hi, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<int4>(g_ys, 256), 0, 3);
});
static_assert(requires(const Dpf8 &s, bool b, cuda::std::span<const int4> s0s,
    cuda::std::span<int4> ys, cudaStream_t stream) {
  s.EvalAllGpuBatch(b, s0s, cuda::std::span<const int4>(g_dense, 18),
      cuda::std::span<const uint16_t>(g_trs, 2), ys, stream);
});
static_assert(requires(const Dpf8 &s, bool b, cuda::std::span<const int4> s0s,
    cuda::std::span<int4> ys, cudaStream_t stream) {
  s.EvalAllGpuBatch(b, s0s, cuda::std::span<const int4>(g_dense, 18),
      cuda::std::span<const uint16_t>(g_trs, 2), ys, stream, 0);
});

// DCF: Cw forms removed; the component forms take both arrays and the control.
static_assert(requires(Dcf8 &s, uint16_t &tr, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_dense, 9), cuda::std::span<int4>(g_values, 9), tr,
      cuda::std::span<const int4, 2>(g_seed_pair), in, int4{});
});
static_assert(requires(Dcf8 &s, bool b, int4 s0, uint16_t tr, uint16_t x) {
  s.Eval(b, s0, cuda::std::span<const int4>(g_dense, 9),
      cuda::std::span<const int4>(g_values, 9), tr, x, 2);
});

// HalfTree: Cw forms removed; the extra final control rides in packed tr.
static_assert(requires(Half8 &s, uint16_t &tr, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_dense, 9), tr, cuda::std::span<const int4, 2>(g_seed_pair), in,
      int4{}, 2);
});

// PackedHalfTree: no control argument on any form.
static_assert(requires(Packed8 &s, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_dense, Packed8::kDepth + 1),
      cuda::std::span<const int4, 2>(g_seed_pair), in, uint64_t{1}, 3);
});

// Grotto: the Cw forms are gone; the parity-tree view entries remain.
static_assert(requires(Grotto8 &s, uint16_t &tr, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_dense, 9), tr, cuda::std::span<const int4, 2>(g_seed_pair), in);
});
static_assert(requires(Grotto8 &s, bool b, int4 s0, uint16_t tr) {
  s.EvalAll(b, s0, cuda::std::span<const int4>(g_dense, 9), tr, g_bits, 0, 2);
});
static_assert(requires(Grotto8::ParityTreeView pt, uint16_t x) { Grotto8::Eval(pt, x); });
static_assert(requires(const HtGrotto8 &s, bool b, int4 s0) {
  s.PreprocessGpu(b, s0, cuda::std::span<const int4>(g_dense, HtGrotto8::kDepth + 1),
      cuda::std::span<int4>(g_dense, 1), cuda::std::span<bool>(g_bits, 256));
});

// VDPF: the Cw forms of Gen/Eval and the nine former Cw-only operations are
// replaced by their component forms; proof and verification entries remain.
static_assert(requires(Vdpf8 &s, uint16_t &tr, uint16_t in) {
  s.Gen(cuda::std::span<int4>(g_dense, 9), tr, cuda::std::span<int4, 4>(g_check),
      cuda::std::span<const int4, 2>(g_seed_pair), in, int4{});
});
static_assert(requires(Vdpf8 &s, bool b, int4 s0, uint16_t tr, uint16_t x, int4 &y) {
  s.Eval(b, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), x, y, 4);
});
static_assert(requires(Vdpf8 &s, bool b, int4 s0, uint16_t tr) {
  s.EvalAll(b, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_check), 0, 2);
});
static_assert(requires(Vdpf8 &s, bool b, uint16_t lo, uint16_t hi, int4 s0, uint16_t tr) {
  s.EvalRange(b, lo, hi, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_check), 0, 2);
});
static_assert(requires(Vdpf8 &s, bool b, int4 s0, uint16_t tr) {
  s.BatchEval(b, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<const uint16_t>(g_trs, 2),
      cuda::std::span<int4>(g_ys, 2), cuda::std::span<int4, 4>(g_check), 2);
});
static_assert(requires(Vdpf8 &s, bool b, int4 s0, uint16_t tr) {
  s.BatchEvalMerkle(b, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<const uint16_t>(g_trs, 2),
      cuda::std::span<int4>(g_ys, 2), cuda::std::span<int4, 4>(g_check));
});
static_assert(requires(Vdpf8 &s, bool b, int4 s0, uint16_t tr) {
  s.EvalAllMerkle(b, s0, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_check));
});
static_assert(requires(const Vdpf8 &s, bool b, uint16_t tr) {
  s.EvalAllMerkleGpu(b, int4{}, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_check));
});
static_assert(requires(const Vdpf8 &s, bool b, uint16_t lo, uint16_t hi, uint16_t tr) {
  s.EvalRangeMerkleGpu(b, lo, hi, int4{}, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_check));
});
static_assert(requires(const Vdpf8 &s, bool b, uint16_t tr) {
  s.BatchEvalMerkleGpu(b, int4{}, cuda::std::span<const int4>(g_dense, 9), tr,
      cuda::std::span<const int4, 4>(g_check), cuda::std::span<const uint16_t>(g_trs, 2),
      cuda::std::span<int4>(g_ys, 2), cuda::std::span<int4, 4>(g_check));
});
static_assert(requires(Vdpf8 &s, cuda::std::span<const cuda::std::array<int4, 4>> leaves,
    cuda::std::span<int4, 4> pi) {
  s.Prove(leaves, cuda::std::span<const int4, 4>(g_check), pi);
});

// DMPF/VDMPF bucket entries stay on their Cw storage contract.
static_assert(requires(Dmpf8 &s, cuda::std::span<typename Dmpf8::Cw> cws, int &m_rt,
    __uint128_t &b_size_rt, int4 sigma, cuda::std::span<const int4> seeds,
    cuda::std::span<const uint16_t> alphas, cuda::std::span<const int4> betas) {
  s.Gen(cws, m_rt, b_size_rt, sigma, seeds, alphas, betas, 30);
});
static_assert(requires(Dmpf8 &s, bool b, int4 sigma, __uint128_t b_size,
    cuda::std::span<const int4, Dmpf8::kBucketCount> s0s,
    cuda::std::span<const typename Dmpf8::Cw> cws, cuda::std::span<const uint16_t> xs,
    cuda::std::span<int4> ys) {
  s.BatchEval(b, sigma, b_size, s0s, cws, xs, ys);
});
static_assert(requires(Vdmpf8 &s, cuda::std::span<typename Vdmpf8::Cw> cws,
    cuda::std::span<int4> cs, int &m_rt, __uint128_t &b_size_rt, int4 sigma,
    cuda::std::span<const int4> seeds, cuda::std::span<const uint16_t> alphas,
    cuda::std::span<const int4> betas) {
  s.Gen(cws, cs, m_rt, b_size_rt, sigma, seeds, alphas, betas, 30);
});

TEST(ApiContract, StrideOnlyCorrectionInterfacesCompile) {
  SUCCEED();
}

}  // namespace
