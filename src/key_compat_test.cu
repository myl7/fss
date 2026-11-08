// SPDX-License-Identifier: Apache-2.0
// Behavior coverage for the deprecated v1.3.0 aggregate-Key compatibility
// entries. Every fixture drives the same seeds through the deprecated Key
// entry and the current split-component API, then compares the packed layout,
// point outputs, full-domain outputs, owners, and proofs of the two routes.
// Legacy-version byte and wire identity is out of scope; the entries must
// match the current library's own outputs. The file builds an empty suite when
// FSS_ENABLE_V1_COMPAT is 0, so the same source compiles in both switch
// configurations.
#include <fss/compat.cuh>

#if FSS_ENABLE_V1_COMPAT

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <cstring>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>
#include <cuda/std/array>
#include <cuda/std/span>
#include <span>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include <fss/hash/aes128_ccr.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include "core_api_test_helpers.cuh"

namespace {

using Bytes = fss::group::Bytes;
using Uint64Group = fss::group::Uint<uint64_t>;
using Uint128Group = fss::group::Uint128<>;
using Prg2 = fss::prg::ChaCha<2>;
using Prg4 = fss::prg::ChaCha<4>;
using Blake3 = fss::hash::Blake3;
using TestCcrHash = fss_test::TestCcrHash;
using CcrBackend = fss_test::CcrBackend<TestCcrHash>;
using Prp = fss::prp::Aes128Feistel;

__host__ __device__ Blake3 Blake(int a, int b) {
  const int4 iv[2] = {{a, b, a, b}, {b, a, b, a}};
  return Blake3{cuda::std::span<const int4, 2>(iv, 2)};
}

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// The compat-only records and entries add no data members to the scheme
// objects; the multi-point schemes are their backends and the packed scheme is
// its backend plus the instance seed.
static_assert(sizeof(fss::Dmpf<16, 30, 14, Bytes, Prg2, Prp, uint16_t>) == sizeof(Prg2) + sizeof(Prp));
// The Vdmpf size is its four backends plus alignment padding, as on the
// switch-off build; the compat entries add no data members.
static_assert(sizeof(fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>) == 96);
static_assert(sizeof(fss::PackedHalfTreeDpf<16, 1, CcrBackend, uint16_t>) ==
    sizeof(CcrBackend) + sizeof(int4));

// The legacy nested names alias the compat records, which stay aggregates with
// the legacy field shapes, extents, and ownership.
static_assert(std::is_aggregate_v<fss::compat::detail::CwsKeyRecord<fss::Dcf<8, Bytes, Prg4>, fss::Dcf<8, Bytes, Prg4>::Cw, 9>>);
static_assert(std::is_aggregate_v<fss::compat::detail::OcwKeyRecord<fss::HalfTreeDpf<8, Bytes, CcrBackend>, fss::HalfTreeDpf<8, Bytes, CcrBackend>::Cw, 8>>);
static_assert(std::is_aggregate_v<fss::compat::detail::FcwKeyRecord<fss::PackedHalfTreeDpf<8, 8, CcrBackend>, fss::PackedHalfTreeDpf<8, 8, CcrBackend>::Cw, 4>>);
static_assert(std::is_aggregate_v<fss::compat::detail::VdpfKeyRecord<fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>, fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>::Cw, 8>>);
static_assert(std::is_same_v<fss::Dcf<8, Bytes, Prg4>::Key,
    fss::compat::detail::CwsKeyRecord<fss::Dcf<8, Bytes, Prg4>, fss::Dcf<8, Bytes, Prg4>::Cw, 9>>);
static_assert(std::is_same_v<fss::GrottoDcf<8, Prg2>::Key,
    fss::compat::detail::CwsKeyRecord<fss::GrottoDcf<8, Prg2>, fss::GrottoDcf<8, Prg2>::Cw, 9>>);
static_assert([] {
  fss::compat::detail::VdpfKeyRecord<fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>,
      fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>::Cw, 8> record;
  return sizeof(record.cws) / sizeof(record.cws[0]);
}() == 8);
static_assert(std::is_same_v<std::decay_t<decltype(fss::compat::detail::VdpfKeyRecord<fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>, fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>::Cw, 8>::cs)>, cuda::std::array<int4, 4>>);
static_assert(!std::is_copy_constructible_v<fss::compat::detail::PreprocessedTreeRecord<fss::GrottoDcf<8, Prg2>>>);
static_assert(std::is_move_constructible_v<fss::compat::detail::PreprocessedTreeRecord<fss::GrottoDcf<8, Prg2>>>);
static_assert(std::is_aggregate_v<fss::compat::detail::EvalResultRecord<fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3>>>);
// The multi-point storage types keep their current modern shape.
static_assert(std::is_same_v<std::decay_t<decltype(std::declval<fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::Key &>().b_size_rt)>, __uint128_t>);
static_assert([] {
  fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::BucketKey bucket_key;
  return sizeof(bucket_key.cws) / sizeof(bucket_key.cws[0]);
}() == 9);
template <typename T>
concept HasOcwMember = requires { T::ocw; };
static_assert(!HasOcwMember<fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::BucketKey>);

// A wide modern configuration whose kBucketSize cannot fit an int still
// instantiates the class under the default switch; only referencing its
// legacy b_size name would trip the domain check.
static_assert(sizeof(fss::Dmpf<126, 30, 127, Bytes, Prg2, Prp, __uint128_t>) > 0);
static_assert(fss::Dmpf<126, 30, 127, Bytes, Prg2, Prp, __uint128_t>::kBucketSize >
    static_cast<__uint128_t>(std::numeric_limits<int>::max()));

// Deprecated constant values alias the current spellings.
static_assert(fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::m == fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::kBucketCount);
static_assert(fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::n == fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::kDomainSize);
static_assert(fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::b_size ==
    static_cast<int>(fss::Dmpf<12, 30, 8, Bytes, Prg2, Prp, uint16_t>::kBucketSize));
static_assert(fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::m ==
    fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::kBucketCount);
static_assert(fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::n ==
    fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::kDomainSize);
static_assert(fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::b_size ==
    static_cast<int>(fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::kBucketSize));
static_assert(fss::Vdmpf<12, 30, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>::kMerkleChunkSize == 256);
static_assert(fss::PackedHalfTreeDpf<8, 8, CcrBackend>::kBetaMax == 0xFF);
static_assert(fss::PackedHalfTreeDpf<8, 8, CcrBackend>::kTopDepth == 3);
static_assert(fss::PackedHalfTreeDpf<8, 64, CcrBackend>::kBetaMax == ~0ULL);
static_assert(fss::Dpf<8, Bytes, Prg2, uint8_t>::Key::kParDepth == -1);
static_assert(fss::HalfTreeDpf<8, Bytes, CcrBackend>::Key::kParDepth == -1);

// Clamped 16B beta, as the generators require a zero LSB. Device kernels of
// this suite call it, so it must be callable in both execution spaces.
__host__ __device__ int4 BetaBuf(int tag = 604) { return {tag, 0, 0, 0}; }

void ExpectBlock(int4 actual, int4 expected, size_t note) {
  EXPECT_EQ(actual.x, expected.x) << "note " << note;
  EXPECT_EQ(actual.y, expected.y) << "note " << note;
  EXPECT_EQ(actual.z, expected.z) << "note " << note;
  EXPECT_EQ(actual.w, expected.w) << "note " << note;
}

void ExpectBlocks(const int4 *actual, const int4 *expected, size_t count) {
  for (size_t i = 0; i < count; ++i) ExpectBlock(actual[i], expected[i], i);
}

int4 BytesAdd(int4 a, int4 b) { return (Bytes::From(a) + Bytes::From(b)).Into(); }

// ---------------------------------------------------------------------------
// Dpf

TEST(DpfKeyCompat, GenEvalEvalAllMatchComponents) {
  using DpfType = fss::Dpf<8, Bytes, Prg2, uint8_t>;
  DpfType dpf{Prg2{kNonce0, kNonce1}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 0xA7;

  int4 seed[9];
  uint8_t tr;
  dpf.Gen(cuda::std::span<int4>(seed, 9), tr, cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf());

  DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf());
  // The shared correction words are byte-identical to the component route and
  // the seeds split by party.
  for (int i = 0; i < 8; ++i) {
    ExpectBlock(k0.cws[i].s, seed[i], i);
    EXPECT_EQ(k0.cws[i].tr, ((tr >> i) & 1) != 0) << "layer " << i;
    ExpectBlock(k1.cws[i].s, k0.cws[i].s, i);
    EXPECT_EQ(k1.cws[i].tr, k0.cws[i].tr);
  }
  ExpectBlock(k0.cws[8].s, seed[8], 8);
  EXPECT_FALSE(k0.cws[8].tr);
  ExpectBlock(k1.cws[8].s, seed[8], 8);
  ExpectBlock(k0.s0, s0s[0], 100);
  ExpectBlock(k1.s0, s0s[1], 101);

  int4 ys0[256], ys1[256], modern0[256];
  dpf.EvalAll(false, k0.s0, cuda::std::span<const int4>(seed, 9), tr, modern0);
  dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0, 256));
  dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1, 256));
  ExpectBlocks(ys0, modern0, 256);
  for (size_t x = 0; x < 256; ++x) {
    ExpectBlock(dpf.Eval(false, k0, static_cast<uint8_t>(x)), ys0[x], x);
    ExpectBlock(dpf.Eval(true, k1, static_cast<uint8_t>(x)), ys1[x], x);
    const int4 reconstructed = BytesAdd(ys0[x], ys1[x]);
    ExpectBlock(reconstructed, x == alpha ? BetaBuf() : int4{0, 0, 0, 0}, x);
  }
}

TEST(DpfKeyCompat, WideInputScalarAndUint128Group) {
  using DpfType = fss::Dpf<100, Uint128Group, Prg2, __uint128_t>;
  DpfType dpf{Prg2{kNonce0, kNonce1}};
  const int4 s0s[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const __uint128_t alpha = (__uint128_t{0x9} << 96) + 7;
  int4 beta = BetaBuf(0x1234);

  DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
  beta.w &= ~1;
  for (__uint128_t x = alpha - 2; x <= alpha + 2; ++x) {
    const auto y0 = dpf.Eval(false, k0, x);
    const auto y1 = dpf.Eval(true, k1, x);
    ExpectBlock((Uint128Group::From(y0) + Uint128Group::From(y1)).Into(), x == alpha ? beta : int4{0, 0, 0, 0}, 0);
  }
}

// ---------------------------------------------------------------------------
// Dcf

TEST(DcfKeyCompat, GenEvalEvalAllMatchComponents) {
  using DcfType = fss::Dcf<8, Uint64Group, Prg4, uint8_t>;
  DcfType dcf{Prg4{kNonce0, kNonce1}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 0x5A;
  int4 b_buf = {700, 0, 0, 0};

  int4 seed[9], value[9];
  uint8_t tr;
  dcf.Gen(cuda::std::span<int4>(seed, 9), cuda::std::span<int4>(value, 9), tr,
      cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

  DcfType::Key k0, k1;
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);
  for (int i = 0; i < 8; ++i) {
    ExpectBlock(k0.cws[i].s, seed[i], i);
    ExpectBlock(k1.cws[i].s, seed[i], i);
    EXPECT_EQ(fss::util::GetLsb(k0.cws[i].v), ((tr >> i) & 1) != 0) << "layer " << i;
  }
  ExpectBlock(k0.cws[8].v, value[8], 8);
  ExpectBlock(k0.s0, s0s[0], 100);

  int4 ys0[256], ys1[256], modern0[256];
  dcf.EvalAll(false, k0.s0, cuda::std::span<const int4>(seed, 9),
      cuda::std::span<const int4>(value, 9), tr, modern0);
  dcf.EvalAll(false, k0, cuda::std::span<int4>(ys0, 256));
  dcf.EvalAll(true, k1, cuda::std::span<int4>(ys1, 256));
  ExpectBlocks(ys0, modern0, 256);
  b_buf.w &= ~1;
  for (size_t x = 0; x < 256; ++x) {
    ExpectBlock(dcf.Eval(false, k0, static_cast<uint8_t>(x)), ys0[x], x);
    const auto sum = Uint64Group::From(dcf.Eval(true, k1, static_cast<uint8_t>(x))) + Uint64Group::From(ys0[x]);
    const uint64_t expected = x < alpha ? (static_cast<uint64_t>(b_buf.x) & 0xFFFFFFFFFFFFFFFEull) : 0;
    EXPECT_EQ(sum.Into().x, static_cast<int>(expected)) << "x " << x;
  }
}

// ---------------------------------------------------------------------------
// HalfTreeDpf

TEST(HalfTreeKeyCompat, GenEvalEvalAllMatchComponents) {
  using Scheme = fss::HalfTreeDpf<8, Bytes, CcrBackend, uint8_t>;
  Scheme ht{CcrBackend{TestCcrHash{kNonce0}}, int4{kNonce1, 1, 2, 3}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 0xC3;

  int4 seed[9];
  uint8_t tr;
  ht.Gen(cuda::std::span<int4>(seed, 9), tr, cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf());

  Scheme::Key k0, k1;
  ht.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf());
  for (int i = 0; i < 7; ++i) {
    ExpectBlock(k0.cws[i].s, seed[i], i);
    EXPECT_FALSE(k0.cws[i].extra);
    ExpectBlock(k1.cws[i].s, k0.cws[i].s, i);
  }
  ExpectBlock(k0.cws[7].s, seed[7], 7);
  EXPECT_EQ(k0.cws[7].extra, ((tr >> 7) & 1) != 0);
  ExpectBlock(k0.ocw, seed[8], 8);
  ExpectBlock(k1.ocw, seed[8], 8);
  ExpectBlock(k1.s0, s0s[1], 101);

  int4 ys0[256], ys1[256], modern0[256];
  ht.EvalAll(false, k0.s0, cuda::std::span<const int4>(seed, 9), tr, modern0);
  ht.EvalAll(false, k0, cuda::std::span<int4>(ys0, 256));
  ht.EvalAll(true, k1, cuda::std::span<int4>(ys1, 256));
  ExpectBlocks(ys0, modern0, 256);
  for (size_t x = 0; x < 256; ++x) {
    ExpectBlock(ht.Eval(false, k0, static_cast<uint8_t>(x)), ys0[x], x);
    ExpectBlock(BytesAdd(ys0[x], ys1[x]), x == alpha ? BetaBuf() : int4{0, 0, 0, 0}, x);
  }
}

// ---------------------------------------------------------------------------
// PackedHalfTreeDpf

TEST(PackedKeyCompat, GenEvalEvalAllMatchComponents) {
  using Scheme = fss::PackedHalfTreeDpf<8, 8, CcrBackend, uint8_t>;
  Scheme pht{CcrBackend{TestCcrHash{kNonce0}}, int4{kNonce1, 1, 2, 3}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 0x6B;
  constexpr uint64_t beta = 0xE7;

  int4 seed[Scheme::kDepth + 1];
  pht.Gen(cuda::std::span<int4>(seed, Scheme::kDepth + 1),
      cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);

  Scheme::Key k0, k1;
  pht.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
  for (int i = 0; i < Scheme::kDepth; ++i) {
    ExpectBlock(k0.cws[i].s, seed[i], i);
    ExpectBlock(k1.cws[i].s, seed[i], i);
  }
  ExpectBlock(k0.fcw, seed[Scheme::kDepth], Scheme::kDepth);
  ExpectBlock(k1.fcw, seed[Scheme::kDepth], Scheme::kDepth);

  constexpr size_t kBlocks = size_t{1} << Scheme::kDepth;
  int4 ys0[kBlocks], ys1[kBlocks], modern0[kBlocks];
  pht.EvalAll(false, k0.s0, cuda::std::span<const int4>(seed, Scheme::kDepth + 1), modern0);
  pht.EvalAll(false, k0, cuda::std::span<int4>(ys0, kBlocks));
  pht.EvalAll(true, k1, cuda::std::span<int4>(ys1, kBlocks));
  ExpectBlocks(ys0, modern0, kBlocks);
  for (size_t x = 0; x < 256; ++x) {
    const int4 block = pht.Eval(false, k0, static_cast<uint8_t>(x));
    ExpectBlock(block, ys0[x >> 4], x);
    const uint64_t reconstructed = Scheme::Extract(block, static_cast<uint8_t>(x)) ^
        Scheme::Extract(pht.Eval(true, k1, static_cast<uint8_t>(x)), static_cast<uint8_t>(x));
    EXPECT_EQ(reconstructed, x == alpha ? beta : 0) << "x " << x;
  }
}

// ---------------------------------------------------------------------------
// GrottoDcf

TEST(GrottoKeyCompat, GenOwnerViewEvalAndEvalAll) {
  using Scheme = fss::GrottoDcf<8, Prg2, uint8_t>;
  Scheme gro{Prg2{kNonce0, kNonce1}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 0x31;

  int4 seed[9];
  uint8_t tr;
  gro.Gen(cuda::std::span<int4>(seed, 9), tr, cuda::std::span<const int4, 2>(s0s, 2), alpha);

  Scheme::Key k0, k1;
  gro.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha);
  for (int i = 0; i <= 8; ++i) ExpectBlock(k0.cws[i].s, seed[i], i);
  ExpectBlock(k1.s0, s0s[1], 101);

  auto pt0 = gro.Preprocess(false, k0);
  auto pt1 = gro.Preprocess(true, k1);
  ASSERT_NE(pt0.p, nullptr);
  EXPECT_FALSE(pt0.b);
  EXPECT_TRUE(pt1.b);
  EXPECT_EQ(pt0.view().p, pt0.p.get());
  EXPECT_FALSE(pt0.view().b);

  // The owner is move-only and moving keeps the borrowed view usable.
  Scheme::ParityTree moved = pt0.view();
  auto pt0_moved = std::move(pt0);
  EXPECT_EQ(moved.p, pt0_moved.p.get());

  bool modern_ys[256], legacy_ys0[256], legacy_ys1[256];
  gro.EvalAll(false, k0.s0, cuda::std::span<const int4>(seed, 9), tr, modern_ys);
  gro.EvalAll(false, k0, cuda::std::span<bool>(legacy_ys0, 256));
  gro.EvalAll(true, k1, cuda::std::span<bool>(legacy_ys1, 256));
  for (size_t x = 0; x < 256; ++x) {
    EXPECT_EQ(legacy_ys0[x], modern_ys[x]) << "x " << x;
    EXPECT_EQ(legacy_ys0[x] ^ legacy_ys1[x], x >= alpha) << "x " << x;
    EXPECT_EQ(Scheme::Eval(pt0_moved, static_cast<uint8_t>(x)), legacy_ys0[x]) << "x " << x;
    EXPECT_EQ(Scheme::Eval(pt0_moved.view(), static_cast<uint8_t>(x)), legacy_ys0[x]) << "x " << x;
    EXPECT_EQ(Scheme::Eval(moved, static_cast<uint8_t>(x)), legacy_ys0[x]) << "x " << x;
  }
}

// ---------------------------------------------------------------------------
// Vdpf

TEST(VdpfKeyCompat, EvalResultAndFullDomainProofs) {
  using Scheme = fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3, uint8_t>;
  Scheme vdpf{Prg2{kNonce0, kNonce1}, Blake(kNonce0, kNonce1), Blake(kNonce1, kNonce0)};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 0x94;

  int4 seed[9];
  uint8_t tr;
  cuda::std::array<int4, 4> cs;
  int4 effective[2] = {s0s[0], s0s[1]};
  int rc = 1;
  for (int attempt = 0; attempt < 64 && rc != 0; ++attempt) {
    effective[0].x = s0s[0].x + attempt * 17;
    effective[1].y = s0s[1].y + attempt * 29;
    rc = vdpf.Gen(cuda::std::span<int4>(seed, 9), tr, cuda::std::span<int4, 4>(cs),
        cuda::std::span<const int4, 2>(effective, 2), alpha, BetaBuf());
  }
  ASSERT_EQ(rc, 0);

  // The aggregate-Key Gen() of the release is not restored; assemble the
  // record from the component outputs exactly as its layout prescribes.
  Scheme::Key k0, k1;
  for (int i = 0; i < 8; ++i) {
    k0.cws[i] = {seed[i], ((tr >> i) & 1) != 0};
    k1.cws[i] = k0.cws[i];
  }
  k0.cs = cs;
  k1.cs = cs;
  k0.ocw = seed[8];
  k1.ocw = seed[8];
  k0.s0 = effective[0];
  k1.s0 = effective[1];

  // Point route: the result record carries the share and the corrected hash,
  // matching the component outputs, and the parties reconstruct the point.
  for (uint8_t x = alpha - 2; x <= alpha + 2; ++x) {
    auto r0 = vdpf.Eval(false, k0, x);
    auto r1 = vdpf.Eval(true, k1, x);
    int4 y_component;
    const auto pi_component = vdpf.Eval(false, k0.s0, cuda::std::span<const int4>(seed, 9),
        tr, cuda::std::span<const int4, 4>(cs), x, y_component);
    ExpectBlock(r0.y, y_component, x);
    for (int j = 0; j < 4; ++j) ExpectBlock(r0.pi_tilde[j], pi_component[j], x * 4 + j);
    ExpectBlock(BytesAdd(r0.y, r1.y), x == alpha ? BetaBuf() : int4{0, 0, 0, 0}, x);
  }

  // Full-domain chain and Merkle routes in independent four-block buffers.
  int4 ys0[256], ys1[256], modern_ys[256];
  cuda::std::array<int4, 4> pi0, pi1, pi_merkle0, pi_merkle1;
  vdpf.EvalAll(false, k0.s0, cuda::std::span<const int4>(seed, 9), tr,
      cuda::std::span<const int4, 4>(cs), cuda::std::span<int4>(modern_ys, 256),
      cuda::std::span<int4, 4>(pi0));
  vdpf.EvalAll(false, k0, cuda::std::span<int4>(ys0, 256), pi0);
  vdpf.EvalAll(true, k1, cuda::std::span<int4>(ys1, 256), pi1);
  ExpectBlocks(ys0, modern_ys, 256);
  EXPECT_TRUE(Scheme::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  for (size_t x = 0; x < 256; ++x)
    ExpectBlock(BytesAdd(ys0[x], ys1[x]), x == alpha ? BetaBuf() : int4{0, 0, 0, 0}, x);

  vdpf.EvalAllMerkle(false, k0, cuda::std::span<int4>(ys0, 256), pi_merkle0);
  vdpf.EvalAllMerkle(true, k1, cuda::std::span<int4>(ys1, 256), pi_merkle1);
  EXPECT_TRUE(Scheme::Verify(cuda::std::span<const int4, 4>(pi_merkle0), cuda::std::span<const int4, 4>(pi_merkle1)));
  // Both accumulator routes verify while producing distinct proofs.
  EXPECT_FALSE(Scheme::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi_merkle1)));
  // Corrupting one proof block must break verification.
  pi1[2].x ^= 1;
  EXPECT_FALSE(Scheme::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
}

// ---------------------------------------------------------------------------
// Dmpf

using DmpfTestType = fss::Dmpf<12, 40, 8, Bytes, Prg2, Prp, uint16_t>;

// Fills a modern Dmpf::Key through the component API, so the deprecated
// aggregate-Key evaluations can be compared against the component route.
struct DmpfFixture {
  static constexpr int kT = 30;
  DmpfTestType dmpf{Prg2{kNonce0, kNonce1}, Prp{}};

  void Gen(typename DmpfTestType::Key &k0, typename DmpfTestType::Key &k1) {
    int4 sigma = {kNonce0, kNonce1, 7, 9};
    cuda::std::array<int4, size_t{DmpfTestType::kBucketCount} * 2> s0s{};
    for (size_t i = 0; i < s0s.size(); ++i)
      s0s[i] = {static_cast<int>(i * 3 + 1), static_cast<int>(i), static_cast<int>(i / 7), static_cast<int>(i / 11) & ~1};
    ASSERT_EQ(fss_test::GenMulti(dmpf, k0, k1, sigma, cuda::std::span<const int4>(s0s),
        cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT), 0);
  }

  uint16_t alphas[kT] = {};
  int4 betas[kT] = {};

  void FillPoints() {
    for (int i = 0; i < kT; ++i) {
      alphas[i] = static_cast<uint16_t>(i * 131 + 17);
      betas[i] = {(i + 1) * 11, 0, 0, 0};
    }
  }
};

TEST(DmpfKeyCompat, BatchEvalAndEvalAllMatchComponents) {
  DmpfFixture fixture;
  fixture.FillPoints();
  typename DmpfTestType::Key k0, k1;
  ASSERT_NO_FATAL_FAILURE(fixture.Gen(k0, k1));
  // The runtime bucket count stays below the allocation, leaving empty and
  // inactive buckets in the key.
  ASSERT_LT(k0.m_rt, DmpfTestType::kBucketCount);
  DmpfTestType &dmpf = fixture.dmpf;

  std::vector<uint16_t> xs(fixture.alphas, fixture.alphas + DmpfFixture::kT);
  for (uint16_t x = 0; x < 200; ++x) xs.push_back(x);
  std::vector<int4> ys0(xs.size()), ys1(xs.size()), modern_ys(xs.size());
  dmpf.BatchEval(false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0));
  dmpf.BatchEval(true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1));

  // Component route: gather through the test helper to compare the two paths.
  std::vector<int4> s0s(DmpfTestType::kBucketCount);
  std::vector<DmpfTestType::Cw> cws(size_t{DmpfTestType::kBucketCount} * (8 + 1));
  for (int i = 0; i < DmpfTestType::kBucketCount; ++i) {
    s0s[i] = k0.bks[i].s0;
    for (int l = 0; l <= 8; ++l) cws[size_t{i} * 9 + l] = k0.bks[i].cws[l];
  }
  dmpf.BatchEval(false, k0.sigma, k0.b_size_rt,
      cuda::std::span<const int4, DmpfTestType::kBucketCount>(s0s.data(), DmpfTestType::kBucketCount),
      cuda::std::span<const DmpfTestType::Cw>(cws.data(), cws.size()),
      cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(modern_ys));
  ExpectBlocks(ys0.data(), modern_ys.data(), xs.size());

  for (size_t i = 0; i < xs.size(); ++i) {
    int4 expected = {0, 0, 0, 0};
    for (int j = 0; j < DmpfFixture::kT; ++j) {
      if (xs[i] == fixture.alphas[j]) expected = fixture.betas[j];
    }
    expected.w &= ~1;
    ExpectBlock(BytesAdd(ys0[i], ys1[i]), expected, i);
  }

  std::vector<int4> all0(size_t{1} << 12), all1(size_t{1} << 12);
  dmpf.EvalAll(false, k0, cuda::std::span<int4>(all0));
  dmpf.EvalAll(true, k1, cuda::std::span<int4>(all1));
  for (size_t x = 0; x < all0.size(); ++x) {
    int4 expected = {0, 0, 0, 0};
    for (int j = 0; j < DmpfFixture::kT; ++j) {
      if (x == fixture.alphas[j]) expected = fixture.betas[j];
    }
    expected.w &= ~1;
    ExpectBlock(BytesAdd(all0[x], all1[x]), expected, x);
  }
  // The batch route and the full-domain route agree on the shared inputs.
  for (uint16_t x = 0; x < 200; ++x) ExpectBlock(ys0[DmpfFixture::kT + x], all0[x], x);
}

// ---------------------------------------------------------------------------
// Vdmpf

using VdmpfTestType = fss::Vdmpf<12, 40, 8, Bytes, Prg2, Blake3, Blake3, Prp, uint16_t>;

struct VdmpfFixture {
  static constexpr int kT = 30;
  VdmpfTestType vdmpf{Prg2{kNonce0, kNonce1}, Blake(kNonce0, kNonce1), Blake(kNonce1, kNonce0), Prp{}};

  uint16_t alphas[kT] = {};
  int4 betas[kT] = {};

  void FillPoints() {
    for (int i = 0; i < kT; ++i) {
      alphas[i] = static_cast<uint16_t>(i * 131 + 17);
      betas[i] = {(i + 1) * 11, 0, 0, 0};
    }
  }

  cuda::std::array<cuda::std::array<int4, 2>, VdmpfTestType::kBucketCount> SeedPairs() const {
    cuda::std::array<cuda::std::array<int4, 2>, VdmpfTestType::kBucketCount> s0s{};
    for (size_t i = 0; i < s0s.size(); ++i) {
      s0s[i][0] = {static_cast<int>(i * 5 + 1), static_cast<int>(i), 3, static_cast<int>(i / 9) & ~1};
      s0s[i][1] = {static_cast<int>(i * 7 + 2), static_cast<int>(i), 5, static_cast<int>(i / 8) & ~1};
    }
    return s0s;
  }
};

TEST(VdmpfKeyCompat, GenSeedsAndStatus) {
  VdmpfFixture fixture;
  fixture.FillPoints();
  VdmpfTestType &vdmpf = fixture.vdmpf;
  int4 sigma = {kNonce0, kNonce1, 11, 13};
  auto s0s = fixture.SeedPairs();
  auto s0s_copy = s0s;

  typename VdmpfTestType::Key k0, k1;
  ASSERT_EQ(vdmpf.Gen(k0, k1, sigma, cuda::std::span<const cuda::std::array<int4, 2>, VdmpfTestType::kBucketCount>(s0s),
      std::span<const uint16_t>(fixture.alphas, VdmpfFixture::kT),
      std::span<const int4>(fixture.betas, VdmpfFixture::kT), VdmpfFixture::kT), 0);

  // The const-seed contract: neither sigma nor the caller's seed pairs change.
  EXPECT_EQ(sigma.x, kNonce0);
  EXPECT_EQ(sigma.y, kNonce1);
  for (size_t i = 0; i < s0s.size(); ++i) {
    ExpectBlock(s0s[i][0], s0s_copy[i][0], i);
    ExpectBlock(s0s[i][1], s0s_copy[i][1], i + 1000);
    // The party-major reorder: bucket i of party 0 keeps s0s[i][0] and party 1
    // keeps s0s[i][1].
    ExpectBlock(k0.bks[i].s0, s0s[i][0], i + 2000);
    ExpectBlock(k1.bks[i].s0, s0s[i][1], i + 3000);
  }
  // Both keys share sigma, the runtime metadata, and every shared component.
  ASSERT_EQ(k0.sigma.x, sigma.x);
  EXPECT_EQ(k0.m_rt, k1.m_rt);
  EXPECT_EQ(k0.b_size_rt, k1.b_size_rt);
  for (int i = 0; i < VdmpfTestType::kBucketCount; ++i) {
    for (int l = 0; l <= 8; ++l) ExpectBlock(k1.bks[i].cws[l].s, k0.bks[i].cws[l].s, i);
    for (int j = 0; j < 4; ++j) ExpectBlock(k1.bks[i].cs[j], k0.bks[i].cs[j], i);
  }

  // A failing single attempt returns the legacy failure code and leaves both
  // keys untouched. The failing configuration comes from the component entry.
  int4 failing_sigma = sigma;
  bool found = false;
  typename VdmpfTestType::Key probe0, probe1;
  for (int attempt = 0; attempt < 64 && !found; ++attempt) {
    failing_sigma.x = kNonce0 + attempt * 37;
    std::vector<VdmpfTestType::Cw> cws(size_t{VdmpfTestType::kBucketCount} * 9);
    std::vector<int4> cs(size_t{VdmpfTestType::kBucketCount} * 4);
    int m_rt;
    __uint128_t b_size_rt;
    std::vector<int4> flat(2 * VdmpfTestType::kBucketCount);
    for (int i = 0; i < VdmpfTestType::kBucketCount; ++i) {
      flat[i] = s0s[i][0];
      flat[VdmpfTestType::kBucketCount + i] = s0s[i][1];
    }
    const int rc = vdmpf.Gen(cuda::std::span<VdmpfTestType::Cw>(cws), cuda::std::span<int4>(cs), m_rt, b_size_rt,
        failing_sigma, cuda::std::span<const int4>(flat),
        cuda::std::span<const uint16_t>(fixture.alphas, VdmpfFixture::kT),
        cuda::std::span<const int4>(fixture.betas, VdmpfFixture::kT), VdmpfFixture::kT, 0);
    if (rc != 0) {
      found = true;
      // Prefill sentinel keys and confirm the deprecated entry leaves them
      // unchanged on failure.
      probe0.sigma = {1, 1, 1, 1};
      probe1.m_rt = -5;
      EXPECT_EQ(vdmpf.Gen(probe0, probe1, failing_sigma,
          cuda::std::span<const cuda::std::array<int4, 2>, VdmpfTestType::kBucketCount>(s0s),
          std::span<const uint16_t>(fixture.alphas, VdmpfFixture::kT),
          std::span<const int4>(fixture.betas, VdmpfFixture::kT), VdmpfFixture::kT, 0), 1);
      EXPECT_EQ(probe0.sigma.x, 1);
      EXPECT_EQ(probe1.m_rt, -5);
    }
  }
  // Cuckoo failure with a zero eviction budget is the expected outcome for a
  // random sigma among this many draws; the loop above asserts the mapping
  // when it happens.
  if (!found) GTEST_SKIP() << "no cuckoo failure found in 64 draws";
}

TEST(VdmpfKeyCompat, BatchEvalAndMerkleMatchComponents) {
  VdmpfFixture fixture;
  fixture.FillPoints();
  VdmpfTestType &vdmpf = fixture.vdmpf;
  int4 sigma = {kNonce0, kNonce1, 11, 13};
  auto s0s = fixture.SeedPairs();

  typename VdmpfTestType::Key k0, k1;
  ASSERT_EQ(vdmpf.Gen(k0, k1, sigma, cuda::std::span<const cuda::std::array<int4, 2>, VdmpfTestType::kBucketCount>(s0s),
      std::span<const uint16_t>(fixture.alphas, VdmpfFixture::kT),
      std::span<const int4>(fixture.betas, VdmpfFixture::kT), VdmpfFixture::kT), 0);

  std::vector<uint16_t> xs(fixture.alphas, fixture.alphas + VdmpfFixture::kT);
  for (uint16_t x = 300; x < 500; ++x) xs.push_back(x);
  std::vector<int4> ys0(xs.size()), ys1(xs.size()), modern_ys(xs.size());
  cuda::std::array<int4, 4> pi0, pi1, pi_std, pi_merkle0, pi_merkle1;

  vdmpf.BatchEval(false, k0, std::span<const uint16_t>(xs), std::span<int4>(ys0), pi0);
  vdmpf.BatchEval(true, k1, std::span<const uint16_t>(xs), std::span<int4>(ys1), pi1);
  EXPECT_TRUE(VdmpfTestType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));

  // Component route on the same key material.
  std::vector<int4> seeds(VdmpfTestType::kBucketCount);
  std::vector<VdmpfTestType::Cw> cws(size_t{VdmpfTestType::kBucketCount} * 9);
  std::vector<int4> cs(size_t{VdmpfTestType::kBucketCount} * 4);
  for (int i = 0; i < VdmpfTestType::kBucketCount; ++i) {
    seeds[i] = k0.bks[i].s0;
    for (int l = 0; l <= 8; ++l) cws[size_t{i} * 9 + l] = k0.bks[i].cws[l];
    for (int j = 0; j < 4; ++j) cs[4 * i + j] = k0.bks[i].cs[j];
  }
  vdmpf.BatchEval(false, k0.sigma, k0.b_size_rt,
      cuda::std::span<const int4, VdmpfTestType::kBucketCount>(seeds.data(), VdmpfTestType::kBucketCount),
      cuda::std::span<const VdmpfTestType::Cw>(cws.data(), cws.size()),
      cuda::std::span<const int4>(cs.data(), cs.size()),
      cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(modern_ys),
      cuda::std::span<int4, 4>(pi_std));
  ExpectBlocks(ys0.data(), modern_ys.data(), xs.size());
  for (int j = 0; j < 4; ++j) ExpectBlock(pi0[j], pi_std[j], j);

  for (size_t i = 0; i < xs.size(); ++i) {
    int4 expected = {0, 0, 0, 0};
    for (int j = 0; j < VdmpfFixture::kT; ++j) {
      if (xs[i] == fixture.alphas[j]) expected = fixture.betas[j];
    }
    expected.w &= ~1;
    ExpectBlock(BytesAdd(ys0[i], ys1[i]), expected, i);
  }

  // The fixed-extent overload mirrors the std::span overload.
  constexpr size_t kQueryCount = 230;
  ASSERT_EQ(xs.size(), kQueryCount);
  cuda::std::span<const uint16_t, kQueryCount> xs_extent(xs.data(), kQueryCount);
  cuda::std::span<int4, kQueryCount> ys_extent(ys0.data(), kQueryCount);
  vdmpf.BatchEval(false, k0, xs_extent, ys_extent, pi0);
  ExpectBlocks(ys0.data(), modern_ys.data(), xs.size());

  vdmpf.BatchEvalMerkle(false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0), pi_merkle0);
  vdmpf.BatchEvalMerkle(true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1), pi_merkle1);
  EXPECT_TRUE(VdmpfTestType::Verify(cuda::std::span<const int4, 4>(pi_merkle0), cuda::std::span<const int4, 4>(pi_merkle1)));
  EXPECT_FALSE(VdmpfTestType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi_merkle1)));
  for (size_t i = 0; i < xs.size(); ++i) {
    int4 expected = {0, 0, 0, 0};
    for (int j = 0; j < VdmpfFixture::kT; ++j) {
      if (xs[i] == fixture.alphas[j]) expected = fixture.betas[j];
    }
    expected.w &= ~1;
    ExpectBlock(BytesAdd(ys0[i], ys1[i]), expected, i);
  }
}


// ---------------------------------------------------------------------------
// Device compilation smoke

// The kernels below instantiate the deprecated HD aggregate-Key entries in
// real device compilation: the Key-form Gen and Eval of Dpf, Dcf,
// HalfTreeDpf, and PackedHalfTreeDpf, the GrottoDcf Key-form Gen and its
// static ParityTree Eval, and the Vdpf Key-form EvalResult Eval. Every result
// lands in the ys output so the device code stays observable. The host-side
// fixture below holds the actual <<< >>> launches, so NVCC emits both the
// device bodies and the host stubs; it runs only where a CUDA device exists.
__global__ void KeyCompatDpfDcfKernel(const int4 *s0s, int4 *ys) {
  using DpfType = fss::Dpf<8, Bytes, Prg2, uint8_t>;
  DpfType dpf{Prg2{kNonce0, kNonce1}};
  DpfType::Key dk0, dk1;
  dpf.Gen(dk0, dk1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0xA7}, BetaBuf());
  ys[0] = dpf.Eval(false, dk0, uint8_t{0xA7});
  ys[1] = dpf.Eval(true, dk1, uint8_t{0});
  ys[2] = dk0.cws[8].s;
  ys[3] = dk1.s0;

  using DcfType = fss::Dcf<8, Bytes, Prg4, uint8_t>;
  DcfType dcf{Prg4{kNonce0, kNonce1}};
  DcfType::Key ck0, ck1;
  dcf.Gen(ck0, ck1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0x5A}, BetaBuf(700));
  ys[4] = dcf.Eval(false, ck0, uint8_t{0});
  ys[5] = dcf.Eval(true, ck1, uint8_t{0x5A});
  ys[6] = ck0.cws[8].v;
}

__global__ void KeyCompatHalfPackedKernel(const int4 *s0s, int4 *ys) {
  using HtType = fss::HalfTreeDpf<8, Bytes, CcrBackend, uint8_t>;
  HtType ht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
  HtType::Key hk0, hk1;
  ht.Gen(hk0, hk1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0xC3}, BetaBuf());
  ys[0] = ht.Eval(false, hk0, uint8_t{0xC3});
  ys[1] = ht.Eval(true, hk1, uint8_t{0});
  ys[2] = hk0.ocw;
  ys[3] = hk1.cws[7].s;

  using PhtType = fss::PackedHalfTreeDpf<8, 8, CcrBackend, uint8_t>;
  PhtType pht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
  PhtType::Key pk0, pk1;
  pht.Gen(pk0, pk1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0x6B}, 0xE7);
  ys[4] = pht.Eval(false, pk0, uint8_t{0x6B});
  ys[5] = pht.Eval(true, pk1, uint8_t{0});
  ys[6] = pk0.fcw;
}

__global__ void KeyCompatGrottoVdpfKernel(const int4 *s0s, bool *tree, int4 *ys,
    fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3, uint8_t>::Key key) {
  using GroType = fss::GrottoDcf<8, Prg2, uint8_t>;
  GroType gro{Prg2{kNonce0, kNonce1}};
  GroType::Key gk0, gk1;
  gro.Gen(gk0, gk1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0x31});
  GroType::ParityTree pt{tree, false};
  ys[0] = make_int4(GroType::Eval(pt, uint8_t{3}) ? 1 : 0, 0, 0, 0);
  ys[1] = gk0.cws[0].s;

  fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3, uint8_t> vdpf{Prg2{kNonce0, kNonce1},
      Blake(kNonce0, kNonce1), Blake(kNonce1, kNonce0)};
  const auto r0 = vdpf.Eval(false, key, uint8_t{0x94});
  const auto r1 = vdpf.Eval(true, key, uint8_t{0x94});
  ys[2] = r0.y;
  ys[3] = r0.pi_tilde[0];
  ys[4] = r1.y;
}

class KeyCompatGpuTest : public ::testing::Test {
protected:
  void SetUp() override {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
  }
};

TEST_F(KeyCompatGpuTest, DeviceKeyEntriesMatchHost) {
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  int4 *device_s0s, *device_ys;
  bool *device_tree;
  ASSERT_EQ(cudaMalloc(&device_s0s, 2 * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, 8 * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tree, (2 * 256 - 1) * sizeof(bool)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_s0s, s0s, 2 * sizeof(int4), cudaMemcpyHostToDevice), cudaSuccess);

  KeyCompatDpfDcfKernel<<<1, 1>>>(device_s0s, device_ys);
  int4 got[7];
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got, device_ys, 7 * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  {
    using DpfType = fss::Dpf<8, Bytes, Prg2, uint8_t>;
    DpfType dpf{Prg2{kNonce0, kNonce1}};
    DpfType::Key k0, k1;
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0xA7}, BetaBuf());
    ExpectBlock(got[0], dpf.Eval(false, k0, uint8_t{0xA7}), 0);
    ExpectBlock(got[1], dpf.Eval(true, k1, uint8_t{0}), 1);
    ExpectBlock(got[2], k0.cws[8].s, 2);
    ExpectBlock(got[3], k1.s0, 3);
    using DcfType = fss::Dcf<8, Bytes, Prg4, uint8_t>;
    DcfType dcf{Prg4{kNonce0, kNonce1}};
    DcfType::Key c0, c1;
    dcf.Gen(c0, c1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0x5A}, BetaBuf(700));
    ExpectBlock(got[4], dcf.Eval(false, c0, uint8_t{0}), 4);
    ExpectBlock(got[5], dcf.Eval(true, c1, uint8_t{0x5A}), 5);
    ExpectBlock(got[6], c0.cws[8].v, 6);
  }

  KeyCompatHalfPackedKernel<<<1, 1>>>(device_s0s, device_ys);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got, device_ys, 7 * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  {
    using HtType = fss::HalfTreeDpf<8, Bytes, CcrBackend, uint8_t>;
    HtType ht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
    HtType::Key h0, h1;
    ht.Gen(h0, h1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0xC3}, BetaBuf());
    ExpectBlock(got[0], ht.Eval(false, h0, uint8_t{0xC3}), 0);
    ExpectBlock(got[1], ht.Eval(true, h1, uint8_t{0}), 1);
    ExpectBlock(got[2], h0.ocw, 2);
    ExpectBlock(got[3], h1.cws[7].s, 3);
    using PhtType = fss::PackedHalfTreeDpf<8, 8, CcrBackend, uint8_t>;
    PhtType pht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
    PhtType::Key p0, p1;
    pht.Gen(p0, p1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0x6B}, 0xE7);
    ExpectBlock(got[4], pht.Eval(false, p0, uint8_t{0x6B}), 4);
    ExpectBlock(got[5], pht.Eval(true, p1, uint8_t{0}), 5);
    ExpectBlock(got[6], p0.fcw, 6);
  }

  using GroType = fss::GrottoDcf<8, Prg2, uint8_t>;
  GroType gro{Prg2{kNonce0, kNonce1}};
  GroType::Key g0, g1;
  gro.Gen(g0, g1, cuda::std::span<const int4, 2>(s0s, 2), uint8_t{0x31});
  auto tree = gro.Preprocess(false, g0);
  ASSERT_EQ(cudaMemcpy(device_tree, tree.p.get(), (2 * 256 - 1) * sizeof(bool), cudaMemcpyHostToDevice), cudaSuccess);

  using VdpfType = fss::Vdpf<8, Bytes, Prg2, Blake3, Blake3, uint8_t>;
  VdpfType vdpf{Prg2{kNonce0, kNonce1}, Blake(kNonce0, kNonce1), Blake(kNonce1, kNonce0)};
  int4 seed[9];
  uint8_t tr;
  cuda::std::array<int4, 4> cs;
  int4 effective[2] = {s0s[0], s0s[1]};
  int rc = 1;
  for (int attempt = 0; attempt < 64 && rc != 0; ++attempt) {
    effective[0].x = s0s[0].x + attempt * 17;
    effective[1].y = s0s[1].y + attempt * 29;
    rc = vdpf.Gen(cuda::std::span<int4>(seed, 9), tr, cuda::std::span<int4, 4>(cs),
        cuda::std::span<const int4, 2>(effective, 2), uint8_t{0x94}, BetaBuf());
  }
  ASSERT_EQ(rc, 0);
  VdpfType::Key vk;
  for (int i = 0; i < 8; ++i) vk.cws[i] = {seed[i], ((tr >> i) & 1) != 0};
  vk.cs = cs;
  vk.ocw = seed[8];
  vk.s0 = effective[0];

  KeyCompatGrottoVdpfKernel<<<1, 1>>>(device_s0s, device_tree, device_ys, vk);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got, device_ys, 5 * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(device_s0s);
  cudaFree(device_ys);
  cudaFree(device_tree);
  EXPECT_EQ(got[0].x, GroType::Eval(tree.view(), uint8_t{3}) ? 1 : 0);
  ExpectBlock(got[1], g0.cws[0].s, 1);
  const auto host_r0 = vdpf.Eval(false, vk, uint8_t{0x94});
  const auto host_r1 = vdpf.Eval(true, vk, uint8_t{0x94});
  ExpectBlock(got[2], host_r0.y, 2);
  ExpectBlock(got[3], host_r0.pi_tilde[0], 3);
  ExpectBlock(got[4], host_r1.y, 4);
}

}  // namespace

#endif  // FSS_ENABLE_V1_COMPAT
