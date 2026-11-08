// SPDX-License-Identifier: Apache-2.0
// Behavior coverage for the deprecated v1.3.0 Cw-form compatibility entries.
// Every fixture generates one key through the current split-component API and
// one through the deprecated Cw entry with the same seeds and backend, then
// compares the packed layout, point evaluations, and full-domain outputs of the
// two routes. Legacy-version byte identity is out of scope; the entries must
// match the current library's own outputs. The file builds an empty suite when
// FSS_ENABLE_V1_COMPAT is 0, so the same source compiles in both switch
// configurations.
#include <fss/compat.cuh>

#if FSS_ENABLE_V1_COMPAT

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda/std/array>
#include <cuda/std/span>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
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

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// The compat entries are member functions and one nested compat-only type, so
// they leave every scheme object's size at its data members alone. The Vdpf
// size is its three backends plus 16B-alignment trailing padding.
static_assert(sizeof(fss::Dpf<16, Bytes, Prg2, uint16_t>) == sizeof(Prg2));
static_assert(sizeof(fss::Dcf<16, Bytes, Prg4, uint16_t>) == sizeof(Prg4));
static_assert(sizeof(fss::HalfTreeDpf<16, Bytes, CcrBackend, uint16_t>) ==
    sizeof(CcrBackend) + sizeof(int4));
static_assert(sizeof(fss::GrottoDcf<16, Prg2, uint16_t>) == sizeof(Prg2));
static_assert(sizeof(fss::Vdpf<16, Bytes, Prg2, Blake3, Blake3, uint16_t>) == 80);

// Clamped 16B beta, as the generators require a zero LSB. Device kernels of
// this suite call it, so it must be callable in both execution spaces.
__host__ __device__ int4 BetaBuf() { return {604, 0, 0, 0}; }

void ExpectBlock(int4 actual, int4 expected, size_t note) {
  EXPECT_EQ(actual.x, expected.x) << "note " << note;
  EXPECT_EQ(actual.y, expected.y) << "note " << note;
  EXPECT_EQ(actual.z, expected.z) << "note " << note;
  EXPECT_EQ(actual.w, expected.w) << "note " << note;
}

void ExpectBlocks(const int4 *actual, const int4 *expected, size_t count) {
  for (size_t i = 0; i < count; ++i) ExpectBlock(actual[i], expected[i], i);
}

// ---------------------------------------------------------------------------
// Dpf

template <int in_bits, typename Group, typename In>
void CheckDpfCwCompat(In alpha) {
  using DpfType = fss::Dpf<in_bits, Group, Prg2, In>;
  DpfType dpf{Prg2{kNonce0, kNonce1}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};

  int4 seed[in_bits + 1];
  In tr;
  dpf.Gen(cuda::std::span<int4>(seed, in_bits + 1), tr,
      cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf());

  typename DpfType::Cw cws[in_bits + 1];
  dpf.Gen(cws, s0s, alpha, BetaBuf());

  // Layout: layer seeds identical, tr bits packed per layer, terminal tr false.
  for (int i = 0; i < in_bits; ++i) {
    ExpectBlock(cws[i].s, seed[i], i);
    EXPECT_EQ(cws[i].tr, ((tr >> i) & 1) != 0) << "layer " << i;
  }
  ExpectBlock(cws[in_bits].s, seed[in_bits], in_bits);
  EXPECT_FALSE(cws[in_bits].tr);

  constexpr In domain_last = in_bits == sizeof(In) * 8
      ? static_cast<In>(~In{0})
      : static_cast<In>((In{1} << in_bits) - 1);
  // alpha + 1 must stay inside the domain, which a full-domain alpha forbids.
  const In after = alpha < domain_last ? In(alpha + 1) : domain_last;
  const In inputs[] = {In{0}, In(alpha - 1), alpha, after, domain_last};
  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    for (In x : inputs) {
      const int4 legacy = dpf.Eval(b, s0, cws, x);
      const int4 modern = dpf.Eval(b, s0, cuda::std::span<const int4>(seed, in_bits + 1), tr, x);
      ExpectBlock(legacy, modern, x);
      // Reconstruction through the output group itself: the shares sum to beta
      // at alpha and to the group zero elsewhere, for any Group, not just XOR.
      const int4 other = dpf.Eval(!b, s0s[b ? 0 : 1], cws, x);
      const Group sum = Group::From(legacy) + Group::From(other);
      const int4 expect = x == alpha ? Group::From(BetaBuf()).Into() : int4{0, 0, 0, 0};
      ExpectBlock(sum.Into(), expect, x);
    }
  }
}

TEST(DpfCwCompat, LayoutEvalAndReconstruction) {
  CheckDpfCwCompat<16, Bytes, uint16_t>(107);
}

TEST(DpfCwCompat, SmallDomainUint8Uint32) {
  using Dpf8 = fss::Dpf<8, Uint64Group, Prg2, uint8_t>;
  Dpf8 dpf8{Prg2{kNonce0, kNonce1}};
  const int4 s0s[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const uint8_t alpha8 = 0x5a;

  int4 seed8[9];
  uint8_t tr8;
  dpf8.Gen(cuda::std::span<int4>(seed8, 9), tr8, cuda::std::span<const int4, 2>(s0s, 2),
      alpha8, BetaBuf());
  Dpf8::Cw cws8[9];
  dpf8.Gen(cws8, s0s, alpha8, BetaBuf());

  // Same-party buffers must agree between the routes; each party is compared
  // against its own modern full-domain output.
  int4 ys_modern0[256], ys_modern1[256], ys_legacy1[256];
  dpf8.EvalAll(false, s0s[0], cuda::std::span<const int4>(seed8, 9), tr8, ys_modern0);
  dpf8.EvalAll(true, s0s[1], cuda::std::span<const int4>(seed8, 9), tr8, ys_modern1);
  dpf8.EvalAll(true, s0s[1], cws8, ys_legacy1);
  ExpectBlocks(ys_legacy1, ys_modern1, 256);
  // The parties' full-domain shares sum to the point function through the
  // output group, which here is additive rather than XOR.
  for (int x = 0; x < 256; ++x) {
    const Uint64Group sum = Uint64Group::From(ys_modern0[x]) + Uint64Group::From(ys_legacy1[x]);
    const int4 expect = x == alpha8 ? Uint64Group::From(BetaBuf()).Into() : int4{0, 0, 0, 0};
    ExpectBlock(sum.Into(), expect, x);
  }

  using Dpf32 = fss::Dpf<32, Bytes, Prg2, uint32_t>;
  Dpf32 dpf32{Prg2{kNonce0, kNonce1}};
  const uint32_t alpha32 = 0x80000001u;  // High-bit scalar in a 32-bit domain.
  int4 seed32[33];
  uint32_t tr32;
  dpf32.Gen(cuda::std::span<int4>(seed32, 33), tr32, cuda::std::span<const int4, 2>(s0s, 2),
      alpha32, BetaBuf());
  Dpf32::Cw cws32[33];
  dpf32.Gen(cws32, s0s, alpha32, BetaBuf());
  const uint32_t inputs32[] = {0, alpha32 - 1, alpha32, alpha32 + 1, 0xffffffffu};
  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    for (uint32_t x : inputs32) {
      ExpectBlock(dpf32.Eval(b, s0, cws32, x),
          dpf32.Eval(b, s0, cuda::std::span<const int4>(seed32, 33), tr32, x), x);
    }
  }
}

TEST(DpfCwCompat, Uint128HighBit) {
  // Wide input domain with the secret point and queries at the top bit.
  const __uint128_t alpha = (__uint128_t{1} << 127) | 107;
  CheckDpfCwCompat<128, Uint128Group, __uint128_t>(alpha);
}

TEST(DpfCwCompat, OneBitDomain) {
  // The smallest domain puts the alpha at the domain edge, so the wrapper's
  // input handling cannot assume a following point exists.
  CheckDpfCwCompat<1, Bytes, uint8_t>(1);
}

// ---------------------------------------------------------------------------
// Dcf

TEST(DcfCwCompat, LayoutEvalAndEvalAll) {
  using DcfType = fss::Dcf<16, Bytes, Prg4, uint16_t>;
  DcfType dcf{Prg4{kNonce0, kNonce1}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;

  int4 seed[17], value[17];
  uint16_t tr;
  dcf.Gen(cuda::std::span<int4>(seed, 17), cuda::std::span<int4>(value, 17), tr,
      cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf());

  DcfType::Cw cws[17];
  dcf.Gen(cws, s0s, alpha, BetaBuf());

  // Layout: seeds identical, tr bits restored into the v LSBs, terminal seed zero.
  for (int i = 0; i < 16; ++i) {
    ExpectBlock(cws[i].s, seed[i], i);
    EXPECT_EQ(cws[i].v.w & 1, (tr >> i) & 1) << "layer " << i;
    ExpectBlock(fss::util::SetLsb(cws[i].v, false), value[i], i);
  }
  ExpectBlock(cws[16].s, int4{0, 0, 0, 0}, 16);
  ExpectBlock(cws[16].v, value[16], 16);

  const uint16_t inputs[] = {0, 106, 107, 108, 65535};
  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    for (uint16_t x : inputs) {
      const int4 legacy = dcf.Eval(b, s0, cws, x);
      const int4 modern = dcf.Eval(b, s0, cuda::std::span<const int4>(seed, 17),
          cuda::std::span<const int4>(value, 17), tr, x);
      ExpectBlock(legacy, modern, x);
    }
  }

  std::vector<int4> ys_modern(65536), ys_legacy(65536);
  dcf.EvalAll(true, s0s[1], cuda::std::span<const int4>(seed, 17),
      cuda::std::span<const int4>(value, 17), tr, ys_modern.data());
  dcf.EvalAll(true, s0s[1], cws, ys_legacy.data());
  ExpectBlocks(ys_legacy.data(), ys_modern.data(), 65536);
}

// ---------------------------------------------------------------------------
// HalfTreeDpf

TEST(HalfTreeCwCompat, LayoutEvalAndEvalAll) {
  using HtType = fss::HalfTreeDpf<16, Bytes, CcrBackend, uint16_t>;
  HtType ht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;

  int4 seed[17];
  uint16_t tr;
  ht.Gen(cuda::std::span<int4>(seed, 17), tr, cuda::std::span<const int4, 2>(s0s, 2),
      alpha, BetaBuf());

  HtType::Cw cws[16];
  int4 ocw{};
  ht.Gen(cws, ocw, s0s, alpha, BetaBuf());

  // Layout: plain seeds for every level, extra false below the last level,
  // LCW_1 of the last level in extra, terminal returned through ocw.
  for (int i = 0; i < 16; ++i) ExpectBlock(cws[i].s, seed[i], i);
  for (int i = 0; i < 15; ++i) EXPECT_FALSE(cws[i].extra) << "layer " << i;
  EXPECT_EQ(cws[15].extra, ((tr >> 15) & 1) != 0);
  ExpectBlock(ocw, seed[16], 16);

  const uint16_t inputs[] = {0, 106, 107, 108, 65535};
  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    for (uint16_t x : inputs) {
      const int4 legacy = ht.Eval(b, s0, cws, ocw, x);
      const int4 modern = ht.Eval(b, s0, cuda::std::span<const int4>(seed, 17), tr, x);
      ExpectBlock(legacy, modern, x);
    }
  }

  std::vector<int4> ys_modern(65536), ys_compat(65536);
  ht.EvalAll(true, s0s[1], cuda::std::span<const int4>(seed, 17), tr, ys_modern.data());
  ht.EvalAll(true, s0s[1], cws, ocw, ys_compat.data());
  ExpectBlocks(ys_compat.data(), ys_modern.data(), 65536);
}

TEST(HalfTreeCwCompat, OneBitDomain) {
  // The single tree level is also the last level, so its packed control bit is
  // the only one the layout carries.
  using HtType = fss::HalfTreeDpf<1, Bytes, CcrBackend, uint8_t>;
  HtType ht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint8_t alpha = 1;

  int4 seed[2];
  uint8_t tr;
  ht.Gen(cuda::std::span<int4>(seed, 2), tr, cuda::std::span<const int4, 2>(s0s, 2),
      alpha, BetaBuf());
  HtType::Cw cws[1];
  int4 ocw{};
  ht.Gen(cws, ocw, s0s, alpha, BetaBuf());

  ExpectBlock(cws[0].s, seed[0], 0);
  EXPECT_EQ(cws[0].extra, (tr & 1) != 0);
  ExpectBlock(ocw, seed[1], 1);

  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    for (uint8_t x : {uint8_t{0}, uint8_t{1}}) {
      ExpectBlock(ht.Eval(b, s0, cws, ocw, x),
          ht.Eval(b, s0, cuda::std::span<const int4>(seed, 2), tr, x), x);
    }
  }
  int4 ys_modern[2], ys_compat[2];
  ht.EvalAll(true, s0s[1], cuda::std::span<const int4>(seed, 2), tr, ys_modern);
  ht.EvalAll(true, s0s[1], cws, ocw, ys_compat);
  ExpectBlocks(ys_compat, ys_modern, 2);
}

// ---------------------------------------------------------------------------
// GrottoDcf

TEST(GrottoCwCompat, LayoutPreprocessAndEvalAll) {
  using GroType = fss::GrottoDcf<16, Prg2, uint16_t>;
  GroType gro{Prg2{kNonce0, kNonce1}};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;

  int4 seed[17];
  uint16_t tr;
  gro.Gen(cuda::std::span<int4>(seed, 17), tr, cuda::std::span<const int4, 2>(s0s, 2), alpha);

  GroType::Cw cws[17];
  gro.Gen(cws, s0s, alpha);

  // The key is the underlying DPF key with beta zero, so the layout matches.
  for (int i = 0; i < 16; ++i) {
    ExpectBlock(cws[i].s, seed[i], i);
    EXPECT_EQ(cws[i].tr, ((tr >> i) & 1) != 0) << "layer " << i;
  }
  ExpectBlock(cws[16].s, seed[16], 16);
  EXPECT_FALSE(cws[16].tr);

  // Full-domain outputs match between the routes for both parties.
  auto ys_modern = std::make_unique<bool[]>(65536);
  auto ys_legacy = std::make_unique<bool[]>(65536);
  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    gro.EvalAll(b, s0, cuda::std::span<const int4>(seed, 17), tr, ys_modern.get());
    gro.EvalAll(b, s0, cws, ys_legacy.get());
    for (size_t x = 0; x < 65536; ++x) EXPECT_EQ(ys_legacy[x], ys_modern[x]) << "x " << x;
  }

  // The party-tagged ParityTree destination receives the modern tree layout,
  // and both parties' trees reconstruct the comparison predicate.
  constexpr size_t kTreeSize = 2 * 65536 - 1;
  auto tree_modern = std::make_unique<bool[]>(kTreeSize);
  auto tree_legacy = std::make_unique<bool[]>(kTreeSize);
  auto tree0_legacy = std::make_unique<bool[]>(kTreeSize);
  gro.Preprocess(true, s0s[1], cuda::std::span<const int4>(seed, 17), tr, tree_modern.get());
  GroType::ParityTree pt{tree_legacy.get(), true};
  gro.Preprocess(pt, s0s[1], cws);
  for (size_t i = 0; i < kTreeSize; ++i) EXPECT_EQ(tree_legacy[i], tree_modern[i]) << "node " << i;
  GroType::ParityTree pt0{tree0_legacy.get(), false};
  gro.Preprocess(pt0, s0s[0], cws);
  const GroType::ParityTreeView view1{tree_legacy.get()}, view0{tree0_legacy.get()};
  for (uint16_t x : {uint16_t{0}, uint16_t{106}, uint16_t{107}, uint16_t{65535}}) {
    EXPECT_EQ(GroType::Eval(view0, x) != GroType::Eval(view1, x), 107 <= x) << "x " << x;
  }
}

// ---------------------------------------------------------------------------
// Vdpf

using Vdpf16 = fss::Vdpf<16, Bytes, Prg2, Blake3, Blake3, uint16_t>;

__host__ __device__ Blake3 MakeBlake3() {
  const int4 iv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  return Blake3{cuda::std::span<const int4, 2>(iv, 2)};
}

TEST(VdpfCwCompat, LayoutEvalEvalAllAndProof) {
  Vdpf16 vdpf{Prg2{kNonce0, kNonce1}, MakeBlake3(), MakeBlake3()};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;

  int4 seed[17];
  uint16_t tr;
  cuda::std::array<int4, 4> cs_modern{};
  ASSERT_EQ(vdpf.Gen(cuda::std::span<int4>(seed, 17), tr, cuda::std::span<int4, 4>(cs_modern),
      cuda::std::span<const int4, 2>(s0s, 2), alpha, BetaBuf()), 0);

  Vdpf16::Cw cws[17];  // One extra entry as a sentinel: Gen must not touch it.
  cws[16] = {int4{0x5a, 0x5a, 0x5a, 0x5a}, true};
  cuda::std::array<int4, 4> cs_legacy{};
  int4 ocw{};
  ASSERT_EQ(vdpf.Gen(cws, cs_legacy, ocw, cuda::std::span<const int4, 2>(s0s, 2),
      alpha, BetaBuf()), 0);

  // Layout: 16 tree entries packed from the split components, terminal through
  // ocw, cs written directly, and no write past the last tree entry.
  for (int i = 0; i < 16; ++i) {
    ExpectBlock(cws[i].s, seed[i], i);
    EXPECT_EQ(cws[i].tr, ((tr >> i) & 1) != 0) << "layer " << i;
  }
  ExpectBlock(ocw, seed[16], 16);
  ExpectBlocks(cs_legacy.data(), cs_modern.data(), 4);
  ExpectBlock(cws[16].s, int4{0x5a, 0x5a, 0x5a, 0x5a}, 16);
  EXPECT_TRUE(cws[16].tr);

  // Point evaluation: output share and all four proof-hash blocks match.
  const uint16_t inputs[] = {0, 106, 107, 108, 65535};
  for (bool b : {false, true}) {
    const int4 s0 = s0s[b ? 1 : 0];
    for (uint16_t x : inputs) {
      int4 y_legacy{}, y_modern{};
      const auto pi_legacy = vdpf.Eval(b, s0, cuda::std::span<const Vdpf16::Cw>(cws, 16),
          cuda::std::span<const int4, 4>(cs_legacy), ocw, x, y_legacy);
      const auto pi_modern = vdpf.Eval(b, s0, cuda::std::span<const int4>(seed, 17), tr,
          cuda::std::span<const int4, 4>(cs_modern), x, y_modern);
      ExpectBlock(y_legacy, y_modern, x);
      ExpectBlocks(pi_legacy.data(), pi_modern.data(), 4);
    }
  }

  // Full-domain evaluation: outputs and the four-block chain proofs match the
  // modern route for both parties, and the proofs verify across parties.
  std::vector<int4> ys_modern[2], ys_legacy[2];
  cuda::std::array<int4, 4> pi_modern[2], pi_legacy[2];
  for (int party = 0; party < 2; ++party) {
    const bool b = party != 0;
    ys_modern[party].resize(65536);
    ys_legacy[party].resize(65536);
    vdpf.EvalAll(b, s0s[party], cuda::std::span<const int4>(seed, 17), tr,
        cuda::std::span<const int4, 4>(cs_modern), ys_modern[party],
        cuda::std::span<int4, 4>(pi_modern[party]));
    vdpf.EvalAll(b, s0s[party], cuda::std::span<const Vdpf16::Cw>(cws, 16),
        cuda::std::span<const int4, 4>(cs_legacy), ocw, ys_legacy[party], pi_legacy[party]);
    ExpectBlocks(ys_legacy[party].data(), ys_modern[party].data(), 65536);
    ExpectBlocks(pi_legacy[party].data(), pi_modern[party].data(), 4);
  }
  EXPECT_TRUE(Vdpf16::Verify(cuda::std::span<const int4, 4>(pi_legacy[0]),
      cuda::std::span<const int4, 4>(pi_legacy[1])));
}

// A zero XorHash forces the all-zero cs rejection of the single-attempt Gen().
struct ZeroXorHash {
  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4> msg) {
    (void)msg;
    return {int4{0, 0, 0, 0}, int4{0, 0, 0, 0}, int4{0, 0, 0, 0}, int4{0, 0, 0, 0}};
  }
};
static_assert(XorHashable<ZeroXorHash>);

TEST(VdpfCwCompat, GenFailurePropagatesWithoutScatter) {
  using VdpfZero = fss::Vdpf<16, Bytes, Prg2, ZeroXorHash, Blake3, uint16_t>;
  VdpfZero vdpf{Prg2{kNonce0, kNonce1}, ZeroXorHash{}, MakeBlake3()};
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};

  VdpfZero::Cw cws[16];
  for (int i = 0; i < 16; ++i) cws[i] = {int4{i, i, i, i}, i % 2 == 0};
  cuda::std::array<int4, 4> cs{};
  int4 ocw = {7, 7, 7, 7};

  const int rc = vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2),
      uint16_t{107}, BetaBuf());
  EXPECT_EQ(rc, 1);
  // The failed attempt leaves the caller's correction words and ocw untouched.
  for (int i = 0; i < 16; ++i) {
    ExpectBlock(cws[i].s, int4{i, i, i, i}, i);
    EXPECT_EQ(cws[i].tr, i % 2 == 0) << "layer " << i;
  }
  ExpectBlock(ocw, int4{7, 7, 7, 7}, 0);
}

// ---------------------------------------------------------------------------
// Device execution of the host-device Cw entries with device-capable backends.
// The wrappers run entirely on the device; the host only uploads seeds and
// downloads results. Expected values come from the same compat entries on the
// host, because the backends are deterministic across execution spaces.

__global__ void CwCompatDpfDcfGrottoKernel(const int4 *s0s, int4 *ys) {
  using DpfType = fss::Dpf<16, Bytes, Prg2, uint16_t>;
  DpfType dpf{Prg2{kNonce0, kNonce1}};
  DpfType::Cw cws[17];
  dpf.Gen(cws, s0s, uint16_t{107}, BetaBuf());
  ys[0] = dpf.Eval(false, s0s[0], cws, uint16_t{0});
  ys[1] = dpf.Eval(true, s0s[1], cws, uint16_t{107});
  ys[2] = cws[16].s;

  using DcfType = fss::Dcf<16, Bytes, Prg4, uint16_t>;
  DcfType dcf{Prg4{kNonce0, kNonce1}};
  DcfType::Cw vcws[17];
  dcf.Gen(vcws, s0s, uint16_t{107}, BetaBuf());
  ys[3] = dcf.Eval(false, s0s[0], vcws, uint16_t{107});
  ys[4] = vcws[16].v;

  using GroType = fss::GrottoDcf<16, Prg2, uint16_t>;
  GroType gro{Prg2{kNonce0, kNonce1}};
  GroType::Cw gcws[17];
  gro.Gen(gcws, s0s, uint16_t{107});
  ys[5] = gcws[0].s;
}

__global__ void CwCompatHalfTreeKernel(const int4 *s0s, int4 *ys) {
  using HtType = fss::HalfTreeDpf<16, Bytes, CcrBackend, uint16_t>;
  HtType ht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
  HtType::Cw cws[16];
  int4 ocw{};
  ht.Gen(cws, ocw, s0s, uint16_t{107}, BetaBuf());
  ys[0] = ht.Eval(false, s0s[0], cws, ocw, uint16_t{0});
  ys[1] = ht.Eval(true, s0s[1], cws, ocw, uint16_t{107});
  ys[2] = ocw;
}

__global__ void CwCompatVdpfKernel(const int4 *s0s, int4 *ys) {
  Vdpf16 vdpf{Prg2{kNonce0, kNonce1}, MakeBlake3(), MakeBlake3()};
  Vdpf16::Cw cws[16];
  cuda::std::array<int4, 4> cs{};
  int4 ocw{};
  const int rc = vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2),
      uint16_t{107}, BetaBuf());
  ys[4] = {rc, 0, 0, 0};
  if (rc != 0) return;
  int4 y{};
  const auto pi_tilde = vdpf.Eval(false, s0s[0], cuda::std::span<const Vdpf16::Cw>(cws, 16),
      cuda::std::span<const int4, 4>(cs), ocw, uint16_t{107}, y);
  ys[0] = y;
  ys[1] = pi_tilde[0];
  ys[2] = ocw;
  ys[3] = cws[15].s;
}

class CwCompatGpuTest : public ::testing::Test {
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

TEST_F(CwCompatGpuTest, DeviceRunsMatchHost) {
  const int4 s0s[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  constexpr size_t kSlots = 6;
  int4 *device_s0s, *device_ys;
  ASSERT_EQ(cudaMalloc(&device_s0s, 2 * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, kSlots * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_s0s, s0s, 2 * sizeof(int4), cudaMemcpyHostToDevice), cudaSuccess);

  CwCompatDpfDcfGrottoKernel<<<1, 1>>>(device_s0s, device_ys);
  int4 dpf_ys[kSlots];
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dpf_ys, device_ys, kSlots * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);

  CwCompatHalfTreeKernel<<<1, 1>>>(device_s0s, device_ys);
  int4 ht_ys[3];
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(ht_ys, device_ys, 3 * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);

  CwCompatVdpfKernel<<<1, 1>>>(device_s0s, device_ys);
  int4 vdpf_ys[5];
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(vdpf_ys, device_ys, 5 * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(device_s0s);
  cudaFree(device_ys);

  {
    using DpfType = fss::Dpf<16, Bytes, Prg2, uint16_t>;
    DpfType dpf{Prg2{kNonce0, kNonce1}};
    DpfType::Cw cws[17];
    dpf.Gen(cws, s0s, uint16_t{107}, BetaBuf());
    ExpectBlock(dpf_ys[0], dpf.Eval(false, s0s[0], cws, uint16_t{0}), 0);
    ExpectBlock(dpf_ys[1], dpf.Eval(true, s0s[1], cws, uint16_t{107}), 1);
    ExpectBlock(dpf_ys[2], cws[16].s, 2);
  }
  {
    using DcfType = fss::Dcf<16, Bytes, Prg4, uint16_t>;
    DcfType dcf{Prg4{kNonce0, kNonce1}};
    DcfType::Cw cws[17];
    dcf.Gen(cws, s0s, uint16_t{107}, BetaBuf());
    ExpectBlock(dpf_ys[3], dcf.Eval(false, s0s[0], cws, uint16_t{107}), 3);
    ExpectBlock(dpf_ys[4], cws[16].v, 4);
  }
  {
    using GroType = fss::GrottoDcf<16, Prg2, uint16_t>;
    GroType gro{Prg2{kNonce0, kNonce1}};
    GroType::Cw gcws[17];
    gro.Gen(gcws, s0s, uint16_t{107});
    ExpectBlock(dpf_ys[5], gcws[0].s, 5);
  }
  {
    using HtType = fss::HalfTreeDpf<16, Bytes, CcrBackend, uint16_t>;
    HtType ht{fss_test::Ccr(TestCcrHash{kNonce0, kNonce1}), int4{11, 22, 33, 44}};
    HtType::Cw cws[16];
    int4 ocw{};
    ht.Gen(cws, ocw, s0s, uint16_t{107}, BetaBuf());
    ExpectBlock(ht_ys[0], ht.Eval(false, s0s[0], cws, ocw, uint16_t{0}), 0);
    ExpectBlock(ht_ys[1], ht.Eval(true, s0s[1], cws, ocw, uint16_t{107}), 1);
    ExpectBlock(ht_ys[2], ocw, 2);
  }
  {
    Vdpf16 vdpf{Prg2{kNonce0, kNonce1}, MakeBlake3(), MakeBlake3()};
    Vdpf16::Cw cws[16];
    cuda::std::array<int4, 4> cs{};
    int4 ocw{};
    ASSERT_EQ(vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2),
        uint16_t{107}, BetaBuf()), 0);
    EXPECT_EQ(vdpf_ys[4].x, 0);
    int4 y{};
    const auto pi_tilde = vdpf.Eval(false, s0s[0], cuda::std::span<const Vdpf16::Cw>(cws, 16),
        cuda::std::span<const int4, 4>(cs), ocw, uint16_t{107}, y);
    ExpectBlock(vdpf_ys[0], y, 0);
    ExpectBlock(vdpf_ys[1], pi_tilde[0], 1);
    ExpectBlock(vdpf_ys[2], ocw, 2);
    ExpectBlock(vdpf_ys[3], cws[15].s, 3);
  }
}

}  // namespace

#endif  // FSS_ENABLE_V1_COMPAT
