// SPDX-License-Identifier: Apache-2.0
// Negative compile coverage for the deprecated v1.3.0 aggregate-Key and
// legacy-name entries. This file must fail to compile with
// FSS_ENABLE_V1_COMPAT=0: every probe uses the exact argument list or nested
// name of one restored entry, so with the switch off the legacy API is absent
// and the probe fails on lookup or overload resolution. FSS_API_REMOVED_KEY_CASE
// selects one probe per translation unit, matching the direct-nvcc harness;
// see cmake/ApiRemovedCwCheck.cmake, which drives this file as well.
#include <cuda/std/span>
#include <span>
#include <cuda/std/array>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include <fss/hash/aes128_ccr.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>

using Bytes = fss::group::Bytes;
using Prg2 = fss::prg::ChaCha<2>;
using Prg4 = fss::prg::ChaCha<4>;
using Blake3 = fss::hash::Blake3;
// The soft AES keeps the instantiated templates device-callable; the AES-NI
// MMO backend is host-only and would fail at ptxas for reasons unrelated to
// the removed entries.
using Ccr = fss::hash::Aes128Ccr<fss::prg::Aes128Soft<1>>;
using Prp = fss::prp::Aes128Feistel;

using Dpf16 = fss::Dpf<16, Bytes, Prg2, uint16_t>;
using Dcf16 = fss::Dcf<16, Bytes, Prg4, uint16_t>;
using Ht16 = fss::HalfTreeDpf<16, Bytes, Ccr, uint16_t>;
using Pht16 = fss::PackedHalfTreeDpf<16, 8, Ccr, uint16_t>;
using Gro16 = fss::GrottoDcf<16, Prg2, uint16_t>;
using Vdpf16 = fss::Vdpf<16, Bytes, Prg2, Blake3, Blake3, uint16_t>;
using Dmpf12 = fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>;
using Vdmpf12 = fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>;

int4 g_s0s[2] = {};
int4 g_ys[256] = {};
bool g_bits[256] = {};
uint16_t g_xs[4] = {};
int4 g_b_bufs[30] = {};
uint16_t g_as[30] = {};

#if defined(FSS_API_REMOVED_KEY_DPF_KEY_GEN)
void Probe(Dpf16 &s) {
  Dpf16::Key k0, k1;
  s.Gen(k0, k1, cuda::std::span<const int4, 2>(g_s0s, 2), uint16_t{3}, int4{});
}
#elif defined(FSS_API_REMOVED_KEY_DPF_KEY_EVAL)
int4 Probe(Dpf16 &s) {
  Dpf16::Key k;
  return s.Eval(false, k, uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_DPF_KEY_EVALALL)
void Probe(Dpf16 &s) {
  Dpf16::Key k;
  s.EvalAll(false, k, cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_KEY_DPF_KEY_KPARDEPTH)
int Probe() {
  Dpf16::Key key{};
  return key.kParDepth;
}
#elif defined(FSS_API_REMOVED_KEY_DCF_KEY_GEN)
void Probe(Dcf16 &s) {
  Dcf16::Key k0, k1;
  s.Gen(k0, k1, cuda::std::span<const int4, 2>(g_s0s, 2), uint16_t{3}, int4{});
}
#elif defined(FSS_API_REMOVED_KEY_DCF_KEY_EVAL)
int4 Probe(Dcf16 &s) {
  Dcf16::Key k;
  return s.Eval(false, k, uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_DCF_KEY_EVALALL)
void Probe(Dcf16 &s) {
  Dcf16::Key k;
  s.EvalAll(false, k, cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_KEY_HT_KEY_GEN)
void Probe(Ht16 &s) {
  Ht16::Key k0, k1;
  s.Gen(k0, k1, cuda::std::span<const int4, 2>(g_s0s, 2), uint16_t{3}, int4{});
}
#elif defined(FSS_API_REMOVED_KEY_HT_KEY_EVAL)
int4 Probe(Ht16 &s) {
  Ht16::Key k;
  return s.Eval(false, k, uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_HT_KEY_EVALALL)
void Probe(Ht16 &s) {
  Ht16::Key k;
  s.EvalAll(false, k, cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_KEY_HT_KEY_KPARDEPTH)
int Probe() {
  Ht16::Key key{};
  return key.kParDepth;
}
#elif defined(FSS_API_REMOVED_KEY_PHT_KEY_GEN)
void Probe(Pht16 &s) {
  Pht16::Key k0, k1;
  s.Gen(k0, k1, cuda::std::span<const int4, 2>(g_s0s, 2), uint16_t{3}, 42);
}
#elif defined(FSS_API_REMOVED_KEY_PHT_KEY_EVAL)
int4 Probe(Pht16 &s) {
  Pht16::Key k;
  return s.Eval(false, k, uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_PHT_KEY_EVALALL)
void Probe(Pht16 &s) {
  Pht16::Key k;
  s.EvalAll(false, k, cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_KEY_PHT_KBETAMAX)
uint64_t Probe() {
  return Pht16::kBetaMax;
}
#elif defined(FSS_API_REMOVED_KEY_PHT_KTOPDEPTH)
int Probe() {
  return Pht16::kTopDepth;
}
#elif defined(FSS_API_REMOVED_KEY_GROTTO_KEY_GEN)
void Probe(Gro16 &s) {
  Gro16::Key k0, k1;
  s.Gen(k0, k1, cuda::std::span<const int4, 2>(g_s0s, 2), uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_GROTTO_KEY_PREPROCESS)
void Probe(Gro16 &s) {
  Gro16::Key k;
  auto pt = s.Preprocess(false, k);
  (void)pt;
}
#elif defined(FSS_API_REMOVED_KEY_GROTTO_KEY_EVAL_PARITYTREE)
bool Probe() {
  Gro16::ParityTree pt{g_bits, false};
  return Gro16::Eval(pt, uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_GROTTO_KEY_EVAL_PREPROCESSED)
bool Probe(Gro16 &s) {
  Gro16::Key k;
  auto pt = s.Preprocess(false, k);
  return Gro16::Eval(pt, uint16_t{3});
}
#elif defined(FSS_API_REMOVED_KEY_GROTTO_KEY_EVALALL)
void Probe(Gro16 &s) {
  Gro16::Key k;
  s.EvalAll(false, k, cuda::std::span<bool>(g_bits, 256));
}
#elif defined(FSS_API_REMOVED_KEY_VDPF_KEY_EVALRESULT)
void Probe() {
  Vdpf16::EvalResult r{};
  (void)r;
}
#elif defined(FSS_API_REMOVED_KEY_VDPF_KEY_EVAL)
void Probe(Vdpf16 &s) {
  Vdpf16::Key k;
  auto r = s.Eval(false, k, uint16_t{3});
  (void)r;
}
#elif defined(FSS_API_REMOVED_KEY_VDPF_KEY_EVALALL)
void Probe(Vdpf16 &s) {
  Vdpf16::Key k;
  cuda::std::array<int4, 4> pi;
  s.EvalAll(false, k, cuda::std::span<int4>(g_ys, 256), pi);
}
#elif defined(FSS_API_REMOVED_KEY_VDPF_KEY_EVALALL_MERKLE)
void Probe(Vdpf16 &s) {
  Vdpf16::Key k;
  cuda::std::array<int4, 4> pi;
  s.EvalAllMerkle(false, k, cuda::std::span<int4>(g_ys, 256), pi);
}
#elif defined(FSS_API_REMOVED_KEY_DMPF_KEY_BATCHEVAL)
void Probe(Dmpf12 &s) {
  Dmpf12::Key k;
  s.BatchEval(false, k, cuda::std::span<const uint16_t>(g_xs, 2), cuda::std::span<int4>(g_ys, 2));
}
#elif defined(FSS_API_REMOVED_KEY_DMPF_KEY_EVALALL)
void Probe(Dmpf12 &s) {
  Dmpf12::Key k;
  s.EvalAll(false, k, cuda::std::span<int4>(g_ys, 256));
}
#elif defined(FSS_API_REMOVED_KEY_DMPF_CONST_M)
int Probe() {
  return Dmpf12::m;
}
#elif defined(FSS_API_REMOVED_KEY_DMPF_CONST_N)
__uint128_t Probe() {
  return Dmpf12::n;
}
#elif defined(FSS_API_REMOVED_KEY_DMPF_CONST_B_SIZE)
int Probe() {
  return Dmpf12::b_size;
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_KEY_GEN)
void Probe(Vdmpf12 &s) {
  Vdmpf12::Key k0, k1;
  cuda::std::array<cuda::std::array<int4, 2>, Vdmpf12::kBucketCount> s0s{};
  s.Gen(k0, k1, int4{}, cuda::std::span<const cuda::std::array<int4, 2>, Vdmpf12::kBucketCount>(s0s),
      std::span<const uint16_t>(g_as, 30), std::span<const int4>(g_b_bufs, 30), 30);
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_KEY_BATCHEVAL_STD)
void Probe(Vdmpf12 &s) {
  Vdmpf12::Key k;
  cuda::std::array<int4, 4> pi;
  s.BatchEval(false, k, std::span<const uint16_t>(g_xs, 2), std::span<int4>(g_ys, 2), pi);
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_KEY_BATCHEVAL_EXTENT)
void Probe(Vdmpf12 &s) {
  Vdmpf12::Key k;
  cuda::std::array<int4, 4> pi;
  s.BatchEval(false, k, cuda::std::span<const uint16_t, 2>(g_xs, 2),
      cuda::std::span<int4, 2>(g_ys, 2), pi);
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_KEY_BATCHEVAL_MERKLE)
void Probe(Vdmpf12 &s) {
  Vdmpf12::Key k;
  cuda::std::array<int4, 4> pi;
  s.BatchEvalMerkle(false, k, cuda::std::span<const uint16_t>(g_xs, 2),
      cuda::std::span<int4>(g_ys, 2), pi);
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_CONST_M)
int Probe() {
  return Vdmpf12::m;
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_CONST_N)
__uint128_t Probe() {
  return Vdmpf12::n;
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_CONST_B_SIZE)
int Probe() {
  return Vdmpf12::b_size;
}
#elif defined(FSS_API_REMOVED_KEY_VDMPF_CONST_MERKLE_CHUNK)
int Probe() {
  return Vdmpf12::kMerkleChunkSize;
}
#endif
