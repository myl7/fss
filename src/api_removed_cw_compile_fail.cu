// SPDX-License-Identifier: Apache-2.0
// Negative compile coverage for the removed public Cw correction-word
// overloads. This file must fail to compile: every call site passes the exact
// argument list of one deleted Cw form, using a real Scheme::Cw array and the
// raw s0s array or Cw span that form accepted. The same file compiles against
// a baseline that still declares the old overloads, so a failure here comes
// from the removed overload, not from probe-argument mistakes. The component
// overloads keep the same method names, so name lookup alone would succeed;
// only the argument lists can carry the check. FSS_API_REMOVED_CW_CASE selects
// one call per translation unit, matching the direct-nvcc harness.
#include <cuda/std/span>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/hash/aes128_ccr.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>

using Bytes = fss::group::Bytes;
using Prg2 = fss::prg::ChaCha<2>;
using Prg4 = fss::prg::ChaCha<4>;
using Blake3 = fss::hash::Blake3;
// The soft AES keeps the instantiated GPU method templates device-callable;
// the AES-NI MMO backend is host-only and would fail at ptxas even on the
// baseline, for reasons unrelated to the removed overloads.
using Ccr = fss::hash::Aes128Ccr<fss::prg::Aes128Soft<1>>;

using Dpf16 = fss::Dpf<16, Bytes, Prg2, uint16_t>;
using Dcf16 = fss::Dcf<16, Bytes, Prg4, uint16_t>;
using Ht16 = fss::HalfTreeDpf<16, Bytes, Ccr, uint16_t>;
using Pht16 = fss::PackedHalfTreeDpf<16, 1, Ccr, uint16_t>;
using Gro16 = fss::GrottoDcf<16, Prg2, uint16_t>;
using Htg16 = fss::HalfTreeGrottoDcf<16, Ccr, uint16_t>;
using Vdpf16 = fss::Vdpf<16, Bytes, Prg2, Blake3, Blake3, uint16_t>;

// One Cw array per scheme type, sized for in_bits = 16 correction words plus
// the terminal entry, and the shared raw buffers the old forms consumed.
Dpf16::Cw g_dpf_cws[17] = {};
Dcf16::Cw g_dcf_cws[17] = {};
Ht16::Cw g_ht_cws[17] = {};
Pht16::Cw g_pht_cws[18] = {};
Gro16::Cw g_gro_cws[17] = {};
Htg16::Cw g_htg_cws[18] = {};
Vdpf16::Cw g_vdpf_cws[17] = {};
int4 g_ys[256] = {};
int4 g_blocks[256] = {};
int4 g_s0s[2] = {};
int4 g_cs[4] = {};
bool g_bits[256] = {};
uint16_t g_xs[2] = {};

#if defined(FSS_API_REMOVED_CW_DPF_GEN)
void Call(Dpf16 &s, uint16_t a) {
  s.Gen(g_dpf_cws, g_s0s, a, int4{});
}
#elif defined(FSS_API_REMOVED_CW_DPF_EVAL)
int4 Call(Dpf16 &s, int4 s0, uint16_t x) {
  return s.Eval(false, s0, g_dpf_cws, x);
}
#elif defined(FSS_API_REMOVED_CW_DPF_RANGE)
void Call(Dpf16 &s, int4 s0) {
  s.EvalRange(false, uint16_t{0}, uint16_t{7}, s0, g_dpf_cws,
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_DPF_EVALALL)
void Call(Dpf16 &s, int4 s0) {
  s.EvalAll(false, s0, g_dpf_cws, g_ys);
}
#elif defined(FSS_API_REMOVED_CW_DPF_RANGE_GPU)
void Call(const Dpf16 &s, int4 s0) {
  (void)s.EvalRangeGpu(false, uint16_t{0}, uint16_t{7}, s0,
      cuda::std::span<const Dpf16::Cw>(g_dpf_cws, 17),
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_DPF_EVALALL_GPU)
void Call(const Dpf16 &s, int4 s0) {
  (void)s.EvalAllGpu(false, s0,
      cuda::std::span<const Dpf16::Cw>(g_dpf_cws, 17),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_DPF_BATCH)
void Call(const Dpf16 &s) {
  (void)s.EvalAllGpuBatch(false, cuda::std::span<const int4>(g_s0s, 1),
      cuda::std::span<const Dpf16::Cw>(g_dpf_cws, 17),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_DCF_GEN)
void Call(Dcf16 &s, uint16_t a) {
  s.Gen(g_dcf_cws, g_s0s, a, int4{});
}
#elif defined(FSS_API_REMOVED_CW_DCF_EVAL)
int4 Call(Dcf16 &s, int4 s0, uint16_t x) {
  return s.Eval(false, s0, g_dcf_cws, x);
}
#elif defined(FSS_API_REMOVED_CW_DCF_RANGE)
void Call(Dcf16 &s, int4 s0) {
  s.EvalRange(false, uint16_t{0}, uint16_t{7}, s0, g_dcf_cws,
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_DCF_EVALALL)
void Call(Dcf16 &s, int4 s0) {
  s.EvalAll(false, s0, g_dcf_cws, g_ys);
}
#elif defined(FSS_API_REMOVED_CW_DCF_RANGE_GPU)
void Call(const Dcf16 &s, int4 s0) {
  (void)s.EvalRangeGpu(false, uint16_t{0}, uint16_t{7}, s0,
      cuda::std::span<const Dcf16::Cw>(g_dcf_cws, 17),
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_DCF_EVALALL_GPU)
void Call(const Dcf16 &s, int4 s0) {
  (void)s.EvalAllGpu(false, s0,
      cuda::std::span<const Dcf16::Cw>(g_dcf_cws, 17),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_DCF_BATCH)
void Call(const Dcf16 &s) {
  (void)s.EvalAllGpuBatch(false, cuda::std::span<const int4>(g_s0s, 1),
      cuda::std::span<const Dcf16::Cw>(g_dcf_cws, 17),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_HT_GEN)
void Call(Ht16 &s, uint16_t a) {
  s.Gen(g_ht_cws, g_s0s, a, int4{});
}
#elif defined(FSS_API_REMOVED_CW_HT_EVAL)
int4 Call(Ht16 &s, int4 s0, uint16_t x) {
  return s.Eval(false, s0, g_ht_cws, x);
}
#elif defined(FSS_API_REMOVED_CW_HT_EVALALL)
void Call(Ht16 &s, int4 s0) {
  s.EvalAll(false, s0, g_ht_cws, g_ys);
}
#elif defined(FSS_API_REMOVED_CW_HT_RANGE)
void Call(Ht16 &s, int4 s0) {
  s.EvalRange(false, uint16_t{0}, uint16_t{7}, s0, g_ht_cws,
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_HT_RANGE_GPU)
void Call(const Ht16 &s, int4 s0) {
  (void)s.EvalRangeGpu(false, uint16_t{0}, uint16_t{7}, s0,
      cuda::std::span<const Ht16::Cw>(g_ht_cws, 17),
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_HT_EVALALL_GPU)
void Call(const Ht16 &s, int4 s0) {
  (void)s.EvalAllGpu(false, s0,
      cuda::std::span<const Ht16::Cw>(g_ht_cws, 17),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_HT_BATCH)
void Call(const Ht16 &s) {
  (void)s.EvalAllGpuBatch(false, cuda::std::span<const int4>(g_s0s, 1),
      cuda::std::span<const Ht16::Cw>(g_ht_cws, 17),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_PHT_GEN)
void Call(Pht16 &s, uint16_t a) {
  s.Gen(g_pht_cws, g_s0s, a, uint64_t{1});
}
#elif defined(FSS_API_REMOVED_CW_PHT_EVAL)
int4 Call(Pht16 &s, int4 s0, uint16_t x) {
  return s.Eval(false, s0, g_pht_cws, x);
}
#elif defined(FSS_API_REMOVED_CW_PHT_EVALALL)
void Call(Pht16 &s, int4 s0) {
  s.EvalAll(false, s0, g_pht_cws, g_ys);
}
#elif defined(FSS_API_REMOVED_CW_PHT_RANGE)
void Call(Pht16 &s, int4 s0) {
  s.EvalRange(false, uint16_t{0}, uint16_t{7}, s0, g_pht_cws,
      cuda::std::span<int4>(g_ys, 8));
}
#elif defined(FSS_API_REMOVED_CW_PHT_EVALALL_GPU)
void Call(const Pht16 &s, int4 s0) {
  (void)s.EvalAllGpu(false, s0,
      cuda::std::span<const Pht16::Cw>(g_pht_cws, 18),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_PHT_BATCH)
void Call(const Pht16 &s) {
  (void)s.EvalAllGpuBatch(false, cuda::std::span<const int4>(g_s0s, 1),
      cuda::std::span<const Pht16::Cw>(g_pht_cws, 18),
      cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_GEN)
void Call(Gro16 &s, uint16_t a) {
  s.Gen(g_gro_cws, g_s0s, a);
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_EVAL)
bool Call(Gro16 &s, int4 s0, uint16_t x) {
  return s.Eval(false, s0, g_gro_cws, x);
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_PREPROCESS)
void Call(Gro16 &s, int4 s0) {
  s.Preprocess(false, s0, g_gro_cws, g_bits);
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_PREPROCESSOWNED)
void Call(Gro16 &s, int4 s0) {
  (void)s.PreprocessOwned(false, s0, g_gro_cws);
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_EVALALL)
void Call(Gro16 &s, int4 s0) {
  s.EvalAll(false, s0, g_gro_cws, g_bits);
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_PREPROCESS_GPU)
void Call(const Gro16 &s, int4 s0) {
  (void)s.PreprocessGpu(false, s0,
      cuda::std::span<const Gro16::Cw>(g_gro_cws, 17),
      cuda::std::span<bool>(g_bits, 256));
}
#elif defined(FSS_API_REMOVED_CW_GROTTO_EVALALL_GPU)
void Call(const Gro16 &s, int4 s0) {
  (void)s.EvalAllGpu(false, s0,
      cuda::std::span<const Gro16::Cw>(g_gro_cws, 17),
      cuda::std::span<bool>(g_bits, 256));
}
#elif defined(FSS_API_REMOVED_CW_HTG_GEN)
void Call(Htg16 &s, uint16_t a) {
  s.Gen(g_htg_cws, g_s0s, a);
}
#elif defined(FSS_API_REMOVED_CW_HTG_EVAL)
bool Call(Htg16 &s, int4 s0, uint16_t x) {
  return s.Eval(false, s0, g_htg_cws, x);
}
#elif defined(FSS_API_REMOVED_CW_HTG_PREPROCESS)
void Call(Htg16 &s, int4 s0) {
  s.Preprocess(false, s0, g_htg_cws, g_blocks, g_bits);
}
#elif defined(FSS_API_REMOVED_CW_HTG_PREPROCESSOWNED)
void Call(Htg16 &s, int4 s0) {
  (void)s.PreprocessOwned(false, s0, g_htg_cws);
}
#elif defined(FSS_API_REMOVED_CW_HTG_EVALALL)
void Call(Htg16 &s, int4 s0) {
  s.EvalAll(false, s0, g_htg_cws, g_bits);
}
#elif defined(FSS_API_REMOVED_CW_HTG_PREPROCESS_GPU)
void Call(const Htg16 &s, int4 s0) {
  (void)s.PreprocessGpu(false, s0,
      cuda::std::span<const Htg16::Cw>(g_htg_cws, 18),
      cuda::std::span<int4>(g_blocks, 256),
      cuda::std::span<bool>(g_bits, 256));
}
#elif defined(FSS_API_REMOVED_CW_HTG_EVALALL_GPU)
void Call(const Htg16 &s, int4 s0) {
  (void)s.EvalAllGpu(false, s0,
      cuda::std::span<const Htg16::Cw>(g_htg_cws, 18),
      cuda::std::span<bool>(g_bits, 256));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_GEN)
int Call(Vdpf16 &s, uint16_t a) {
  return s.Gen(g_vdpf_cws, cuda::std::span<int4, 4>(g_cs),
      cuda::std::span<const int4, 2>(g_s0s), a, int4{});
}
#elif defined(FSS_API_REMOVED_CW_VDPF_EVAL)
cuda::std::array<int4, 4> Call(Vdpf16 &s, int4 s0, uint16_t x, int4 &y) {
  return s.Eval(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), x, y);
}
#elif defined(FSS_API_REMOVED_CW_VDPF_EVALALL)
void Call(Vdpf16 &s, int4 s0) {
  s.EvalAll(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_RANGE)
void Call(Vdpf16 &s, int4 s0) {
  s.EvalRange(false, uint16_t{0}, uint16_t{7}, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<int4>(g_ys, 8),
      cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_BATCH)
void Call(Vdpf16 &s, int4 s0) {
  s.BatchEval(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<const uint16_t>(g_xs, 2),
      cuda::std::span<int4>(g_ys, 2), cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_BATCH_MERKLE)
void Call(Vdpf16 &s, int4 s0) {
  s.BatchEvalMerkle(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<const uint16_t>(g_xs, 2),
      cuda::std::span<int4>(g_ys, 2), cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_EVALALL_MERKLE)
void Call(Vdpf16 &s, int4 s0) {
  s.EvalAllMerkle(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_RANGE_MERKLE)
void Call(Vdpf16 &s, int4 s0) {
  s.EvalRangeMerkle(false, uint16_t{0}, uint16_t{7}, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<int4>(g_ys, 8),
      cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_EVALALL_MERKLE_GPU)
void Call(const Vdpf16 &s, int4 s0) {
  (void)s.EvalAllMerkleGpu(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<int4>(g_ys, 256),
      cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_RANGE_MERKLE_GPU)
void Call(const Vdpf16 &s, int4 s0) {
  (void)s.EvalRangeMerkleGpu(false, uint16_t{0}, uint16_t{7}, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<int4>(g_ys, 8),
      cuda::std::span<int4, 4>(g_cs));
}
#elif defined(FSS_API_REMOVED_CW_VDPF_BATCH_MERKLE_GPU)
void Call(const Vdpf16 &s, int4 s0) {
  (void)s.BatchEvalMerkleGpu(false, s0,
      cuda::std::span<const Vdpf16::Cw>(g_vdpf_cws, 17),
      cuda::std::span<const int4, 4>(g_cs), cuda::std::span<const uint16_t>(g_xs, 2),
      cuda::std::span<int4>(g_ys, 2), cuda::std::span<int4, 4>(g_cs));
}
#else
#error "define one FSS_API_REMOVED_CW_* case"
#endif
