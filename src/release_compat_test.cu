// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <climits>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>
#include <atomic>
#include <thread>
#include <omp.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/dmpf.cuh>
#include <fss/point_eval_cpu.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prp/chacha_feistel.cuh>
#include <fss/hash/blake3.cuh>

namespace {

using Bytes = fss::group::Bytes;
constexpr int4 kSeeds[2] = {{12345, 87, 6543, 42}, {8765, 4321, 111, 222}};
constexpr int4 kBeta = {604, 0, 0, 0};
constexpr int4 kHashKey = {1, 2, 3, 4};
constexpr int4 kIv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};

void ExpectInt4(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

template <typename Cw>
void ExpectCw(const Cw &actual, const Cw &expected) {
  ExpectInt4(actual.s, expected.s);
  if constexpr (requires { actual.v; }) ExpectInt4(actual.v, expected.v);
  if constexpr (requires { actual.tr; }) EXPECT_EQ(actual.tr, expected.tr);
  if constexpr (requires { actual.extra; }) EXPECT_EQ(actual.extra, expected.extra);
}

// Test-only backends deliberately force generation failure without invoking
// unrelated arithmetic in a cryptographic backend during UBSan retry checks.
template <int mul>
struct ZeroPrg {
  __host__ __device__ cuda::std::array<int4, mul> Gen(int4) { return {}; }
};
struct ZeroXorHash {
  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4>) { return {}; }
};
struct ZeroHash {
  __host__ __device__ cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4>) { return {}; }
};
struct IdentityPrp {
  __host__ __device__ __uint128_t Permu(int4, __uint128_t x, __uint128_t) { return x; }
};

struct AesExecutionState {
  std::thread::id caller = std::this_thread::get_id();
  std::atomic<int> active{0};
  std::atomic<bool> invalid_context{false};
};
struct CheckedAes {
  fss::prg::Aes128Mmo<2> prg;
  AesExecutionState *state;
  __host__ __device__ cuda::std::array<int4, 2> Gen(int4 seed) {
#ifdef __CUDA_ARCH__
    return prg.Gen(seed);
#else
    if (state->active.fetch_add(1) != 0 || omp_in_parallel() || std::this_thread::get_id() != state->caller) {
      state->invalid_context = true;
    }
    auto result = prg.Gen(seed);
    state->active.fetch_sub(1);
    return result;
#endif
  }
};

template <int depth>
void CheckConfiguredPackers() {
  using Dpf = fss::Dpf<8, Bytes, fss::prg::ChaCha<2>, uint16_t, depth>;
  using Half = fss::HalfTreeDpf<8, Bytes, fss::prg::ChaCha<1>, uint16_t, depth>;
  Dpf dpf{fss::prg::ChaCha<2>{1, 2}};
  Half half{fss::prg::ChaCha<1>{1, 2}, kHashKey};
  typename Dpf::Key dk0, dk1;
  typename Half::Key hk0, hk1;
  dpf.Gen(dk0, dk1, kSeeds, 107, kBeta);
  half.Gen(hk0, hk1, kSeeds, 107, kBeta);
  // Existing four explicit template arguments deduce configured key depth.
  auto dp = fss::cpu::DpfPackPoint<8, Bytes, fss::prg::ChaCha<2>, uint16_t>(dk0);
  auto hp = fss::cpu::HalfTreeDpfPackPoint<8, Bytes, fss::prg::ChaCha<1>, uint16_t>(hk0);
  auto dp_explicit = fss::cpu::DpfPackPoint<8, Bytes, fss::prg::ChaCha<2>, uint16_t, depth>(dk0);
  ExpectInt4(dp_explicit.s0, dp.s0);
  for (int x = 0; x < 256; ++x) {
    auto dy = fss::cpu::DpfEvalPoint<8, Bytes, fss::prg::ChaCha<2>, uint16_t>(false, dp, x, dpf.prg);
    auto hy = fss::cpu::HalfTreeDpfEvalPoint<8, Bytes, fss::prg::ChaCha<1>, uint16_t>(
        false, hp, x, half.prg, kHashKey);
    ExpectInt4(dy, dpf.Eval(false, dk0, x));
    ExpectInt4(hy, half.Eval(false, hk0, x));
  }
}

}  // namespace

TEST(ReleaseCompatibility, RawDpfAndDcfKeepCorrectionWordsAndOutputOrder) {
  using Dpf = fss::Dpf<8, Bytes, fss::prg::ChaCha<2>, uint16_t, 0>;
  using Dcf = fss::Dcf<8, Bytes, fss::prg::ChaCha<4>, uint16_t, fss::DcfPred::kLt, 0>;
  Dpf dpf{fss::prg::ChaCha<2>{1, 2}};
  Dcf dcf{fss::prg::ChaCha<4>{1, 2}};
  Dpf::Cw dcws[9];
  Dcf::Cw ccws[9];
  Dpf::Key dk0, dk1;
  Dcf::Key ck0, ck1;
  dpf.Gen(dcws, kSeeds, 107, kBeta);
  dcf.Gen(ccws, kSeeds, 107, kBeta);
  dpf.Gen(dk0, dk1, kSeeds, 107, kBeta);
  dcf.Gen(ck0, ck1, kSeeds, 107, kBeta);
  for (int i = 0; i < 9; ++i) {
    ExpectCw(dcws[i], dk0.cws[i]);
    ExpectCw(ccws[i], ck0.cws[i]);
  }
  int4 dy[256], cy[256];
  for (bool party : {false, true}) {
    dpf.EvalAll(party, kSeeds[party], dcws, dy);
    dcf.EvalAll(party, kSeeds[party], ccws, cy);
    for (int x = 0; x < 256; ++x) {
      ExpectInt4(dy[x], dpf.Eval(party, party ? dk1 : dk0, x));
      ExpectInt4(cy[x], dcf.Eval(party, party ? ck1 : ck0, x));
    }
  }
}

TEST(ReleaseCompatibility, RawHalfTreeAndCallerOwnedGrotto) {
  using Half = fss::HalfTreeDpf<8, Bytes, fss::prg::ChaCha<1>, uint16_t, 0>;
  Half half{fss::prg::ChaCha<1>{1, 2}, kHashKey};
  Half::Key k0, k1;
  Half::Cw cws[8];
  int4 ocw;
  half.Gen(cws, ocw, kSeeds, 107, kBeta);
  half.Gen(k0, k1, kSeeds, 107, kBeta);
  for (int i = 0; i < 8; ++i) ExpectCw(cws[i], k0.cws[i]);
  ExpectInt4(ocw, k0.ocw);
  int4 ys[256];
  half.EvalAll(false, kSeeds[0], cws, ocw, ys);
  for (int x = 0; x < 256; ++x) ExpectInt4(ys[x], half.Eval(false, k0, x));

  using Grotto = fss::GrottoDcf<8, fss::prg::ChaCha<2>, uint16_t, 0>;
  Grotto grotto{fss::prg::ChaCha<2>{1, 2}};
  Grotto::Cw gcws[9];
  Grotto::Key g0, g1;
  grotto.Gen(gcws, kSeeds, 107);
  grotto.Gen(g0, g1, kSeeds, 107);
  bool storage[511], raw_ys[256];
  Grotto::ParityTree view{storage, false};
  bool *allocation = view.p;
  grotto.Preprocess(view, kSeeds[0], gcws);
  EXPECT_EQ(view.p, allocation);
  auto owned = grotto.Preprocess(false, g0);
  grotto.EvalAll(false, kSeeds[0], gcws, raw_ys);
  for (int x = 0; x < 256; ++x) {
    EXPECT_EQ(grotto.Eval(view, x), grotto.Eval(owned, x));
    EXPECT_EQ(raw_ys[x], grotto.Eval(view, x));
  }
}

TEST(ReleaseCompatibility, RawVdpfPreservesChainProofAndConstSeeds) {
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<8, Bytes, fss::prg::ChaCha<2>, Hash, Hash, uint16_t, 0>;
  Scheme scheme{fss::prg::ChaCha<2>{1, 2}, Hash{kIv}, Hash{kIv}};
  Scheme::Key k0, k1;
  Scheme::Cw cws[8];
  cuda::std::array<int4, 4> cs;
  int4 ocw;
  int4 effective[2] = {kSeeds[0], kSeeds[1]};
  ASSERT_EQ(scheme.Gen(k0, k1, effective, 107, kBeta), 0);
  const int4 seeds[2] = {effective[0], effective[1]};
  ASSERT_EQ(scheme.Gen(cws, cs, ocw, seeds, 107, kBeta), 0);
  for (int i = 0; i < 8; ++i) ExpectCw(cws[i], k0.cws[i]);
  int4 raw_ys[256], new_ys[256];
  cuda::std::array<int4, 4> raw_pi, new_pi;
  scheme.EvalAll(false, seeds[0], cws, cs, ocw, raw_ys, raw_pi);
  scheme.EvalAll(false, k0, new_ys, new_pi);
  for (int i = 0; i < 4; ++i) ExpectInt4(raw_pi[i], new_pi[i]);
  for (int x = 0; x < 256; ++x) {
    int4 y;
    auto point_pi = scheme.Eval(false, seeds[0], cws, cs, ocw, x, y);
    auto result = scheme.Eval(false, k0, x);
    ExpectInt4(y, result.y);
    ExpectInt4(raw_ys[x], new_ys[x]);
    for (int i = 0; i < 4; ++i) ExpectInt4(point_pi[i], result.pi_tilde[i]);
  }
}

TEST(ReleaseCompatibility, ChaChaLegacyNonceReadsAreDeferred) {
  int nonce[2] = {1, 2};
  fss::prg::ChaCha<2> legacy{nonce};
  static_assert(std::is_trivially_copyable_v<decltype(legacy)>);
  for (int i = 0; i < 3; ++i) {
    nonce[0] = i * 3;
    nonce[1] = 100 - i;
    auto actual = legacy.Gen(kSeeds[0]);
    auto expected = fss::prg::ChaCha<2>{nonce[0], nonce[1]}.Gen(kSeeds[0]);
    for (int j = 0; j < 2; ++j) ExpectInt4(actual[j], expected[j]);
  }
}

TEST(ReleaseCompatibility, ConfiguredCpuPackers) {
  CheckConfiguredPackers<-1>();
  CheckConfiguredPackers<0>();
  CheckConfiguredPackers<2>();
}

TEST(ReleaseCompatibility, RetryArithmeticWrapsSeedWords) {
  EXPECT_EQ(fss::util::IncrementWord(INT_MAX), INT_MIN);
  EXPECT_EQ(fss::util::IncrementWord(INT_MIN), INT_MIN + 1);
  EXPECT_EQ(fss::util::IncrementWord(-1), 0);
  using Vdpf = fss::Vdpf<1, Bytes, ZeroPrg<2>, ZeroXorHash, ZeroHash, uint16_t, 0>;
  Vdpf vdpf{{}, {}, {}};
  Vdpf::Key k0, k1;
  for (int word : {INT_MAX, INT_MIN, -1}) {
    int4 seeds[2] = {{word, 0, 0, 0}, {0, word, 0, 0}};
    Vdpf::Cw cws[1];
    cuda::std::array<int4, 4> cs;
    int4 ocw;
    ASSERT_EQ(vdpf.Gen(cws, cs, ocw, seeds, 0, kBeta), 1);
    EXPECT_EQ(seeds[0].x, word);
    EXPECT_EQ(seeds[1].y, word);
    ASSERT_EQ(vdpf.Gen(k0, k1, seeds, 0, kBeta, false), 1);
    EXPECT_EQ(seeds[0].x, word);
    EXPECT_EQ(seeds[1].y, word);
    ASSERT_EQ(vdpf.Gen(k0, k1, seeds, 0, kBeta), 1);
    int expected = word;
    for (int i = 0; i < 128; ++i) expected = fss::util::IncrementWord(expected);
    EXPECT_EQ(seeds[0].x, expected);
    EXPECT_EQ(seeds[1].y, expected);
  }
}

TEST(ReleaseCompatibility, RetryArithmeticAndReleasedVdmpfFailFast) {
  using Vdmpf = fss::Vdmpf<8, 30, 6, Bytes, ZeroPrg<2>, ZeroXorHash, ZeroHash, IdentityPrp, uint16_t>;
  using Dmpf = fss::Dmpf<8, 30, 6, Bytes, ZeroPrg<2>, IdentityPrp, uint16_t, 3, 80, 0>;
  Vdmpf vdmpf{{}, {}, {}, {}};
  Dmpf dmpf{{}, {}};
  Vdmpf::Key vk0, vk1;
  Dmpf::Key dk0, dk1;
  Vdmpf::BucketKey aggregate{{}, {}, {}, {1, 2, 3, 4}};
  ExpectInt4(aggregate.s0, int4{1, 2, 3, 4});
  uint16_t duplicate[30] = {};
  int4 betas[30] = {};
  int4 sigma = {INT_MAX, 1, 2, 3};
  cuda::std::array<cuda::std::array<int4, 2>, Vdmpf::m> seeds;
  for (auto &pair : seeds) pair = {int4{INT_MAX, 0, 0, 0}, int4{0, INT_MAX, 0, 0}};
  // Omitting optional arguments on mutable storage still selects v1.2.0's
  // fail-fast form and never writes the seeds or sigma.
  ASSERT_EQ(vdmpf.Gen(vk0, vk1, sigma, seeds, std::span<const uint16_t>(duplicate),
      std::span<const int4>(betas), 30), 1);
  EXPECT_EQ(sigma.x, INT_MAX);
  EXPECT_EQ(seeds[0][0].x, INT_MAX);
  ASSERT_EQ(vdmpf.Gen(vk0, vk1, sigma, seeds, duplicate, betas, 30, 0, false), 1);
  EXPECT_EQ(sigma.x, INT_MAX);
  ASSERT_EQ(vdmpf.Gen(vk0, vk1, sigma, seeds, duplicate, betas, 30, 0, true), 1);
  EXPECT_EQ(sigma.x, INT_MIN + 127);
  EXPECT_EQ(seeds[0][0].x, INT_MIN + 127);
  EXPECT_EQ(seeds[0][1].y, INT_MIN + 127);
  sigma.x = INT_MAX;
  for (auto &pair : seeds) pair = {int4{INT_MAX, 0, 0, 0}, int4{0, INT_MAX, 0, 0}};
  ASSERT_EQ(dmpf.Gen(dk0, dk1, sigma, seeds, duplicate, betas, 30, 0, true), 1);
  EXPECT_EQ(sigma.x, INT_MIN + 127);
  EXPECT_EQ(seeds[0][0].x, INT_MIN + 127);
}

TEST(ReleaseCompatibility, BucketBoundsCompileBeforeAllocation) {
  using Small = fss::Dmpf<5, 30, 1, Bytes, ZeroPrg<2>, IdentityPrp>;
  using Large = fss::Dmpf<34, 30, 30, Bytes, ZeroPrg<2>, IdentityPrp, uint64_t>;
  using Verifiable = fss::Vdmpf<34, 30, 30, Bytes, ZeroPrg<2>, ZeroXorHash, ZeroHash, IdentityPrp, uint64_t>;
  static_assert(Small::b_size <= 2);
  static_assert(Large::b_size > 0 && Large::b_size <= (uint64_t{1} << 30));
  static_assert(Verifiable::b_size == Large::b_size);
  EXPECT_EQ(Large::b_size, Verifiable::b_size);
}

TEST(ReleaseCompatibility, SequentialDmpfUsesOneSetOfAesContexts) {
  omp_set_num_threads(4);
  unsigned char key0[16] = {}, key1[16] = {1};
  const unsigned char *keys[2] = {key0, key1};
  auto contexts = fss::prg::Aes128Mmo<2>::CreateCtxs(keys);
  AesExecutionState state;
  using Scheme = fss::Dmpf<8, 30, 6, Bytes, CheckedAes, fss::prp::ChaChaFeistel, uint16_t, 3, 80, 0>;
  Scheme scheme{CheckedAes{fss::prg::Aes128Mmo<2>{contexts}, &state}, {}};
  Scheme::Key k0, k1;
  int4 sigma = kHashKey;
  cuda::std::array<cuda::std::array<int4, 2>, Scheme::m> seeds;
  for (auto &pair : seeds) pair = {kSeeds[0], kSeeds[1]};
  uint16_t alphas[30];
  int4 betas[30];
  for (int i = 0; i < 30; ++i) {
    alphas[i] = i * 3;
    betas[i] = kBeta;
  }
  int ret = scheme.Gen(k0, k1, sigma, seeds, alphas, betas, 30);
  EXPECT_EQ(ret, 0);
  if (ret == 0) {
    std::vector<int4> ys0(256), ys1(256);
    scheme.EvalAll(false, k0, ys0);
    scheme.EvalAll(true, k1, ys1);
    for (int x = 0; x < 256; ++x) {
      int4 expected = x < 90 && x % 3 == 0 ? Bytes::From(kBeta).Into() : int4{0, 0, 0, 0};
      ExpectInt4(fss::util::Xor(ys0[x], ys1[x]), expected);
    }
  }
  EXPECT_FALSE(state.invalid_context.load());
  fss::prg::Aes128Mmo<2>::FreeCtxs(contexts);
}
