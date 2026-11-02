#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <fss/prg/aes128_mmo_raw.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include "core_api_test_helpers.cuh"
#include <atomic>
#include <limits>

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

static bool EqualInt4(int4 a, int4 b) {
  return memcmp(&a, &b, sizeof(int4)) == 0;
}

// Owned-table AES-CCR test fixture over all supported output widths
template <int OutBits>
class PackedHalfTreeDpfOwnedAesCcrTest : public ::testing::Test {
protected:
  using Prg = fss_test::TestCcrHash;
  using DpfType = fss::PackedHalfTreeDpf<kAlphaBits, OutBits, fss_test::CcrBackend<Prg>, uint16_t>;

  static constexpr uint64_t kMaxBeta = OutBits == 64 ? ~0ULL : (1ULL << OutBits) - 1;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;
  int4 hash_key;

  PackedHalfTreeDpfOwnedAesCcrTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  void TestEvalAtAlpha() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    int4 y0 = fss_test::EvalKey(dpf, false, k0, kAlpha);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, kAlpha);

    int4 diff = fss::util::Xor(y0, y1);
    EXPECT_TRUE(EqualInt4(diff, fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta)));
    EXPECT_EQ(DpfType::Extract(diff, kAlpha), kMaxBeta);
  }

  void TestEvalAtNonAlpha() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    int4 zero = {0, 0, 0, 0};

    // Same block as alpha, different lane: the block reconstructs to the
    // packed beta, and x's lane within it is zero.
    uint16_t x = kAlpha ^ 1;
    int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, x);
    EXPECT_TRUE(EqualInt4(fss::util::Xor(y0, y1), fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta))) << "Failed at x=" << x;
    EXPECT_EQ(DpfType::Extract(fss::util::Xor(y0, y1), x), 0) << "Failed at x=" << x;

    for (int i = 0; i < 100; i++) {
      x = dis(gen);

      y0 = fss_test::EvalKey(dpf, false, k0, x);
      y1 = fss_test::EvalKey(dpf, true, k1, x);
      int4 diff = fss::util::Xor(y0, y1);
      if (x / DpfType::kLanes == kAlpha / DpfType::kLanes) {
        EXPECT_TRUE(EqualInt4(diff, fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta))) << "Failed at x=" << x;
      } else {
        EXPECT_TRUE(EqualInt4(diff, zero)) << "Failed at x=" << x;
      }
      EXPECT_EQ(DpfType::Extract(diff, x), x == kAlpha ? kMaxBeta : 0) << "Failed at x=" << x;
    }
  }

  void TestEvalAll() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks), ys1(blocks);

    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    size_t alpha_block = kAlpha / DpfType::kLanes;
    int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta);
    int4 zero = {0, 0, 0, 0};

    for (size_t j = 0; j < blocks; ++j) {
      int4 diff = fss::util::Xor(ys0[j], ys1[j]);
      if (j == alpha_block) {
        EXPECT_TRUE(EqualInt4(diff, expected)) << "Failed at block " << j;
      } else {
        EXPECT_TRUE(EqualInt4(diff, zero)) << "Failed at block " << j;
      }
    }
  }

  void TestEvalMatchesEvalAll() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks);
    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 50; i++) {
      uint16_t x = dis(gen);
      int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
      EXPECT_TRUE(EqualInt4(y0, ys0[x / DpfType::kLanes])) << "Failed at x=" << x;
    }
  }

  void TestAlphaSweep() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(blocks), ys1(blocks);

    // Alphas on lane and block boundaries, plus a random interior point
    std::vector<uint16_t> alphas = {0, 1, static_cast<uint16_t>(DpfType::kLanes / 2),
        static_cast<uint16_t>(DpfType::kLanes - 1), static_cast<uint16_t>(DpfType::kLanes),
        static_cast<uint16_t>(2 * DpfType::kLanes - 1), kAlpha, static_cast<uint16_t>(n - 1)};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> alpha_dis(0, static_cast<uint16_t>(n - 1));
    alphas.push_back(alpha_dis(gen));

    std::uniform_int_distribution<uint64_t> beta_dis(1, kMaxBeta);

    int4 zero = {0, 0, 0, 0};

    for (uint16_t alpha : alphas) {
      uint64_t beta = beta_dis(gen);
      ASSERT_TRUE(beta <= kMaxBeta);
      fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
      fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
      fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

      size_t alpha_block = alpha / DpfType::kLanes;
      int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(alpha, beta);

      for (size_t j = 0; j < blocks; ++j) {
        int4 diff = fss::util::Xor(ys0[j], ys1[j]);
        EXPECT_TRUE(EqualInt4(diff, j == alpha_block ? expected : zero))
            << "alpha=" << alpha << " block=" << j;
      }

      int4 alpha_diff = fss::util::Xor(ys0[alpha_block], ys1[alpha_block]);
      for (int lane = 0; lane < DpfType::kLanes; ++lane) {
        uint16_t x = static_cast<uint16_t>(alpha_block * DpfType::kLanes + lane);
        EXPECT_EQ(DpfType::Extract(alpha_diff, x), x == alpha ? beta : 0)
            << "alpha=" << alpha << " lane=" << lane;
      }
    }
  }

  void TestPackBetaExtract() {
    int lanes[] = {0, 1, 5, 30, 31, 32, 33, 63, 64, 95, 96, 127, DpfType::kLanes - 1};
    for (int lane : lanes) {
      if (lane >= DpfType::kLanes) continue;

      DpfType dpf{fss_test::Ccr(prg), hash_key};
      fss_test::GenPair(dpf, k0, k1, s0s, static_cast<uint16_t>(lane), kMaxBeta);
      int4 packed = fss::util::Xor(fss_test::EvalKey(dpf, false, k0, static_cast<uint16_t>(lane)),
          fss_test::EvalKey(dpf, true, k1, static_cast<uint16_t>(lane)));
      for (int j = 0; j < DpfType::kLanes; ++j) {
        uint64_t v = DpfType::Extract(packed, static_cast<uint16_t>(j));
        EXPECT_EQ(v, j == lane ? kMaxBeta : 0) << "lane=" << lane << " j=" << j;
      }
    }
  }
};

#define PACKED_HALF_TREE_DPF_CHACHA_TEST(OutBits)                                        \
  using PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest = PackedHalfTreeDpfOwnedAesCcrTest<OutBits>; \
  TEST_F(PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest, EvalAtAlpha) {                       \
    TestEvalAtAlpha();                                                                   \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest, EvalAtNonAlpha) {                    \
    TestEvalAtNonAlpha();                                                                \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest, EvalAll) {                           \
    TestEvalAll();                                                                       \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest, EvalMatchesEvalAll) {                \
    TestEvalMatchesEvalAll();                                                            \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest, AlphaSweep) {                        \
    TestAlphaSweep();                                                                    \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitOwnedAesCcrTest, PackBetaExtract) {                   \
    TestPackBetaExtract();                                                               \
  }

PACKED_HALF_TREE_DPF_CHACHA_TEST(1)
PACKED_HALF_TREE_DPF_CHACHA_TEST(2)
PACKED_HALF_TREE_DPF_CHACHA_TEST(4)
PACKED_HALF_TREE_DPF_CHACHA_TEST(8)
PACKED_HALF_TREE_DPF_CHACHA_TEST(16)
PACKED_HALF_TREE_DPF_CHACHA_TEST(32)
PACKED_HALF_TREE_DPF_CHACHA_TEST(64)

// Aes128Soft-based test fixture
class PackedHalfTreeDpfAesSoftTest : public ::testing::Test {
protected:
  using Prg = fss::prg::Aes128Soft<1>;
  using DpfType = fss::PackedHalfTreeDpf<kAlphaBits, 1, fss_test::CcrBackend<Prg>, uint16_t>;

  static constexpr uint64_t kMaxBeta = 1;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  uint8_t aes_key[1][16] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};
  // Aes128Soft borrows these tables, including after MakePrg() returns.
  uint32_t te0[256];
  uint8_t sbox[256];
  int4 hash_key;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  Prg MakePrg() {
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    return Prg(aes_key, te0, sbox);
  }

  void TestEvalAll() {
    Prg prg = MakePrg();
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks), ys1(blocks);

    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    size_t alpha_block = kAlpha / DpfType::kLanes;
    int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta);
    int4 zero = {0, 0, 0, 0};

    for (size_t j = 0; j < blocks; ++j) {
      int4 diff = fss::util::Xor(ys0[j], ys1[j]);
      if (j == alpha_block) {
        EXPECT_TRUE(EqualInt4(diff, expected)) << "Failed at block " << j;
      } else {
        EXPECT_TRUE(EqualInt4(diff, zero)) << "Failed at block " << j;
      }
    }
  }
};

TEST_F(PackedHalfTreeDpfAesSoftTest, EvalAll) {
  TestEvalAll();
}

TEST_F(PackedHalfTreeDpfAesSoftTest, EvalMatchesEvalAll) {
  // Fixed inputs make table lifetime and both parties' output shares reproducible.
  s0s[0] = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};
  s0s[1] = {0x56789abc, 0x6789abcd, 0x789abcde, 0x12345679};
  hash_key = {0x23456789, 0x3456789a, 0x456789ab, 0x56789abc};
  Prg prg = MakePrg();
  DpfType dpf{fss_test::Ccr(prg), hash_key};

  constexpr size_t blocks = 1ULL << DpfType::kDepth;
  std::vector<int4> ys0(blocks), ys1(blocks);
  const uint16_t alphas[] = {0, DpfType::kLanes - 1, DpfType::kLanes, 0xffff};
  const int4 zero = {0, 0, 0, 0};
  for (uint16_t alpha : alphas) {
    for (uint64_t beta : {0ULL, 1ULL}) {
      SCOPED_TRACE(::testing::Message() << "alpha=" << alpha << " beta=" << beta);
      fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
      fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
      fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));
      for (size_t j = 0; j < blocks; ++j) {
        uint16_t x = static_cast<uint16_t>(j * DpfType::kLanes);
        ASSERT_TRUE(EqualInt4(fss_test::EvalKey(dpf, false, k0, x), ys0[j])) << "block=" << j;
        ASSERT_TRUE(EqualInt4(fss_test::EvalKey(dpf, true, k1, x), ys1[j])) << "block=" << j;
        int4 diff = fss::util::Xor(ys0[j], ys1[j]);
        ASSERT_TRUE(EqualInt4(diff, j == alpha / DpfType::kLanes ? fss_test::PackBetaReference<128 / DpfType::kLanes>(alpha, beta) : zero))
            << "block=" << j;
      }
    }
  }
}

// Edge cases: kDepth == 1, the shallowest legal trees
template <int InBits, int OutBits>
class PackedHalfTreeDpfOneLevelTest : public ::testing::Test {
protected:
  using Prg = fss_test::TestCcrHash;
  using In = uint8_t;
  using DpfType = fss::PackedHalfTreeDpf<InBits, OutBits, fss_test::CcrBackend<Prg>, In>;

  static constexpr uint64_t kMaxBeta = OutBits == 64 ? ~0ULL : (1ULL << OutBits) - 1;
  static_assert(DpfType::kDepth == 1);

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;
  int4 hash_key;

  PackedHalfTreeDpfOneLevelTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  void TestEvalAll() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 zero = {0, 0, 0, 0};
    for (int alpha = 0; alpha < (1 << InBits); ++alpha) {
      fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), static_cast<In>(alpha), kMaxBeta);

      int4 ys0[2], ys1[2];
      fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0, 2));
      fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1, 2));

      size_t alpha_block = alpha / DpfType::kLanes;
      int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(static_cast<In>(alpha), kMaxBeta);
      for (size_t j = 0; j < 2; ++j) {
        int4 diff = fss::util::Xor(ys0[j], ys1[j]);
        EXPECT_TRUE(EqualInt4(diff, j == alpha_block ? expected : zero))
            << "alpha=" << alpha << " block=" << j;
      }
      for (int x = 0; x < (1 << InBits); ++x) {
        int4 diff = fss::util::Xor(ys0[x / DpfType::kLanes], ys1[x / DpfType::kLanes]);
        EXPECT_EQ(DpfType::Extract(diff, static_cast<In>(x)), x == alpha ? kMaxBeta : 0)
            << "alpha=" << alpha << " x=" << x;
      }
    }
  }
};

using PackedHalfTreeDpfOneLevel1BitTest = PackedHalfTreeDpfOneLevelTest<8, 1>;
using PackedHalfTreeDpfOneLevel64BitTest = PackedHalfTreeDpfOneLevelTest<2, 64>;

TEST_F(PackedHalfTreeDpfOneLevel1BitTest, EvalAll) {
  TestEvalAll();
}
TEST_F(PackedHalfTreeDpfOneLevel64BitTest, EvalAll) {
  TestEvalAll();
}

// Every PRG takes the level-order EvalAll() path when kDepth > kTopDepth (3);
// Aes128MmoRaw adds the Gen8() pipelining behind detail::GenBatch(). These
// tests pin the frontier path against the scalar Eval walk and the
// XOR-reconstruction semantics of the scheme.
class PackedHalfTreeDpfAesRawTest : public ::testing::Test {
protected:
  using Prg = fss::prg::Aes128MmoRaw<1>;
  using DpfType = fss::PackedHalfTreeDpf<kAlphaBits, 1, fss_test::CcrBackend<Prg>, uint16_t>;

  static constexpr uint64_t kMaxBeta = 1;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  uint8_t aes_key[1][16] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};
  int4 hash_key;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  void TestEvalAll() {
    Prg prg(aes_key);
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks), ys1(blocks);

    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    size_t alpha_block = kAlpha / DpfType::kLanes;
    int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta);
    int4 zero = {0, 0, 0, 0};

    for (size_t j = 0; j < blocks; ++j) {
      int4 diff = fss::util::Xor(ys0[j], ys1[j]);
      if (j == alpha_block) {
        EXPECT_TRUE(EqualInt4(diff, expected)) << "Failed at block " << j;
      } else {
        EXPECT_TRUE(EqualInt4(diff, zero)) << "Failed at block " << j;
      }
    }
  }

  void TestEvalMatchesEvalAll() {
    Prg prg(aes_key);
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks);
    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 50; i++) {
      uint16_t x = dis(gen);
      int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
      EXPECT_TRUE(EqualInt4(y0, ys0[x / DpfType::kLanes])) << "Failed at x=" << x;
    }
  }

  void TestAlphaSweep() {
    Prg prg(aes_key);
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(blocks), ys1(blocks);

    std::vector<uint16_t> alphas = {0, 1, static_cast<uint16_t>(DpfType::kLanes - 1),
        static_cast<uint16_t>(DpfType::kLanes), static_cast<uint16_t>(2 * DpfType::kLanes - 1),
        static_cast<uint16_t>(blocks * DpfType::kLanes / 2), kAlpha, static_cast<uint16_t>(n - 1)};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> alpha_dis(0, static_cast<uint16_t>(n - 1));
    for (int i = 0; i < 8; i++) alphas.push_back(alpha_dis(gen));

    int4 zero = {0, 0, 0, 0};

    for (uint16_t alpha : alphas) {
      fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, kMaxBeta);
      fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
      fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

      size_t alpha_block = alpha / DpfType::kLanes;
      int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(alpha, kMaxBeta);

      for (size_t j = 0; j < blocks; ++j) {
        int4 diff = fss::util::Xor(ys0[j], ys1[j]);
        EXPECT_TRUE(EqualInt4(diff, j == alpha_block ? expected : zero))
            << "alpha=" << alpha << " block=" << j;
      }
    }
  }
};

TEST_F(PackedHalfTreeDpfAesRawTest, EvalAll) {
  TestEvalAll();
}
TEST_F(PackedHalfTreeDpfAesRawTest, EvalMatchesEvalAll) {
  TestEvalMatchesEvalAll();
}
TEST_F(PackedHalfTreeDpfAesRawTest, AlphaSweep) {
  TestAlphaSweep();
}

// Boundary domains around the 8-wide frontier: kDepth 4 and 5 take the batched path with
// the shallowest frontiers, kDepth 1..3 keep the per-block point walk.
// Every alpha is swept so each block position is the correction target once.
template <int InBits>
class PackedHalfTreeDpfAesRawBoundaryTest : public ::testing::Test {
protected:
  using Prg = fss::prg::Aes128MmoRaw<1>;
  using In = uint16_t;
  using DpfType = fss::PackedHalfTreeDpf<InBits, 1, fss_test::CcrBackend<Prg>, In>;
  static_assert(InBits >= 8 && InBits <= 12);
  // out_bits = 1 gives kLanes = 128 and kDepth = InBits - 7, so an 8-wide frontier
  // forms for InBits >= 11 and shallower domains keep the point walk.

  static constexpr uint64_t kMaxBeta = 1;
  static constexpr size_t n = 1ULL << InBits;

  int4 s0s[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  uint8_t aes_key[1][16] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};
  int4 hash_key = {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, 0x2468ace0};

  void TestFullAlphaSweep() {
    Prg prg(aes_key);
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    int4 ys0[blocks], ys1[blocks];

    int4 zero = {0, 0, 0, 0};

    for (size_t alpha = 0; alpha < n; ++alpha) {
      typename DpfType::Key k0, k1;
      fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), static_cast<In>(alpha), kMaxBeta);

      fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0, blocks));
      fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1, blocks));

      size_t alpha_block = alpha / DpfType::kLanes;
      int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(static_cast<In>(alpha), kMaxBeta);

      for (size_t j = 0; j < blocks; ++j) {
        int4 diff = fss::util::Xor(ys0[j], ys1[j]);
        EXPECT_TRUE(EqualInt4(diff, j == alpha_block ? expected : zero))
            << "in_bits=" << InBits << " alpha=" << alpha << " block=" << j;

        // The batched path must stay bit-exact with the scalar per-point walk.
        int4 y0 = fss_test::EvalKey(dpf, false, k0, static_cast<In>(j * DpfType::kLanes));
        EXPECT_TRUE(EqualInt4(y0, ys0[j]))
            << "in_bits=" << InBits << " alpha=" << alpha << " block=" << j;
      }
    }
  }
};

using PackedHalfTreeDpfAesRawDepth1Test = PackedHalfTreeDpfAesRawBoundaryTest<8>;
using PackedHalfTreeDpfAesRawDepth2Test = PackedHalfTreeDpfAesRawBoundaryTest<9>;
using PackedHalfTreeDpfAesRawDepth3Test = PackedHalfTreeDpfAesRawBoundaryTest<10>;
using PackedHalfTreeDpfAesRawDepth4Test = PackedHalfTreeDpfAesRawBoundaryTest<11>;
using PackedHalfTreeDpfAesRawDepth5Test = PackedHalfTreeDpfAesRawBoundaryTest<12>;

TEST_F(PackedHalfTreeDpfAesRawDepth1Test, FullAlphaSweep) {
  TestFullAlphaSweep();
}
TEST_F(PackedHalfTreeDpfAesRawDepth2Test, FullAlphaSweep) {
  TestFullAlphaSweep();
}
TEST_F(PackedHalfTreeDpfAesRawDepth3Test, FullAlphaSweep) {
  TestFullAlphaSweep();
}
TEST_F(PackedHalfTreeDpfAesRawDepth4Test, FullAlphaSweep) {
  TestFullAlphaSweep();
}
TEST_F(PackedHalfTreeDpfAesRawDepth5Test, FullAlphaSweep) {
  TestFullAlphaSweep();
}

// Aes128Mmo has no Gen8(), so EvalAll() runs the frontier through the
// detail::GenBatch() loop; this pins the OpenSSL PRG cell of the EvalAll matrix.
class PackedHalfTreeDpfBytesAesTest : public ::testing::Test {
protected:
  using Prg = fss::prg::Aes128Mmo<1>;
  using DpfType = fss::PackedHalfTreeDpf<kAlphaBits, 1, fss_test::CcrBackend<Prg>, uint16_t>;

  static constexpr uint64_t kMaxBeta = 1;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  int4 hash_key;
  cuda::std::array<EVP_CIPHER_CTX *, 1> ctxs;

  void SetUp() override {
    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const unsigned char *keys[1] = {key0};
    ctxs = Prg::CreateCtxs(keys);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  void TearDown() override {
    Prg::FreeCtxs(ctxs);
  }

  void TestEvalAll() {
    Prg prg(ctxs);
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks), ys1(blocks);
    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    size_t alpha_block = kAlpha / DpfType::kLanes;
    int4 expected = fss_test::PackBetaReference<128 / DpfType::kLanes>(kAlpha, kMaxBeta);
    int4 zero = {0, 0, 0, 0};

    for (size_t j = 0; j < blocks; ++j) {
      int4 diff = fss::util::Xor(ys0[j], ys1[j]);
      if (j == alpha_block) {
        EXPECT_TRUE(EqualInt4(diff, expected)) << "Failed at block " << j;
      } else {
        EXPECT_TRUE(EqualInt4(diff, zero)) << "Failed at block " << j;
      }
    }
  }
};

TEST_F(PackedHalfTreeDpfBytesAesTest, EvalAll) {
  TestEvalAll();
}

// Layout constants and lane known vectors
TEST(PackedHalfTreeDpfLayout, Constants) {
  using D1 = fss::PackedHalfTreeDpf<16, 1, fss_test::CcrBackend<fss_test::TestCcrHash>, uint16_t>;
  static_assert(D1::kLanes == 128);
  static_assert(D1::kDepth == 9);
  static_assert(sizeof(D1::Cw) == 16);

  using D8 = fss::PackedHalfTreeDpf<16, 8, fss_test::CcrBackend<fss_test::TestCcrHash>, uint16_t>;
  static_assert(D8::kLanes == 16);
  static_assert(D8::kDepth == 12);

  using D64 = fss::PackedHalfTreeDpf<20, 64, fss_test::CcrBackend<fss_test::TestCcrHash>, uint32_t>;
  static_assert(D64::kLanes == 2);
  static_assert(D64::kDepth == 19);

  SUCCEED();
}

template <typename Scheme>
int4 ReconstructPackedKnownVector(uint16_t alpha, uint64_t beta) {
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const int4 seeds[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  typename Scheme::Cw cws[Scheme::kDepth + 1];
  scheme.Gen(cws, seeds, alpha, beta);
  return fss::util::Xor(scheme.Eval(false, seeds[0], cws, alpha),
      scheme.Eval(true, seeds[1], cws, alpha));
}

TEST(PackedHalfTreeDpfLayout, LaneKnownVectors) {
  using D1 = fss::PackedHalfTreeDpf<16, 1, fss_test::CcrBackend<fss_test::TestCcrHash>, uint16_t>;
  int4 p = ReconstructPackedKnownVector<D1>(5, 1);
  EXPECT_EQ(p.x, 1 << 5);
  EXPECT_EQ(p.y, 0);
  EXPECT_EQ(p.z, 0);
  EXPECT_EQ(p.w, 0);

  using D8 = fss::PackedHalfTreeDpf<16, 8, fss_test::CcrBackend<fss_test::TestCcrHash>, uint16_t>;
  // Lane 3 of 16: byte lane 3 lives at bits 24..31 of the low word
  p = ReconstructPackedKnownVector<D8>(3, 0xAB);
  EXPECT_EQ(p.x, 0xAB << 24);
  EXPECT_EQ(D8::Extract(p, 3), 0xAB);
  EXPECT_EQ(D8::Extract(p, 2), 0);

  using D64 = fss::PackedHalfTreeDpf<16, 64, fss_test::CcrBackend<fss_test::TestCcrHash>, uint16_t>;
  int4 q = ReconstructPackedKnownVector<D64>(1, 0x1122334455667788ULL);
  EXPECT_EQ(q.x, 0);
  EXPECT_EQ(q.y, 0);
  EXPECT_EQ(q.z, static_cast<int>(0x55667788u));
  EXPECT_EQ(q.w, static_cast<int>(0x11223344u));
  EXPECT_EQ(D64::Extract(q, 1), 0x1122334455667788ULL);
  EXPECT_EQ(D64::Extract(q, 0), 0);
}

namespace {
template <int width>
concept PackedOutputWidth = requires {
  typename fss::PackedHalfTreeDpf<12, width, fss_test::TestCcrHash>;
};
static_assert(PackedOutputWidth<1> && PackedOutputWidth<64>);
static_assert(!PackedOutputWidth<0> && !PackedOutputWidth<3> && !PackedOutputWidth<65>);
}  // namespace

namespace {
struct PackedRangeHash {
  fss_test::TestCcrHash backend;
  std::atomic<size_t> *calls;
  std::atomic<bool> *parallel;

  __host__ __device__ int4 Hash(int4 input) {
#ifndef __CUDA_ARCH__
    calls->fetch_add(1, std::memory_order_relaxed);
#ifdef _OPENMP
    if (omp_in_parallel()) parallel->store(true, std::memory_order_relaxed);
#endif
#endif
    return backend.Hash(input);
  }
};

template <int bits, int width, typename In>
void CheckPackedRange(In first, In last, In alpha, int depth) {
  std::atomic<size_t> calls{0};
  std::atomic<bool> parallel{false};
  using Scheme = fss::PackedHalfTreeDpf<bits, width, PackedRangeHash, In>;
  Scheme scheme{PackedRangeHash{fss_test::TestCcrHash{}, &calls, &parallel}, {31, 47, 59, 83}};
  const int4 seeds[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  constexpr uint64_t beta = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
  typename Scheme::Cw cws[Scheme::kDepth + 1];
  scheme.Gen(cws, seeds, alpha, beta);
  const size_t count = static_cast<size_t>(last - first) + 1;
  const size_t blocks = count / Scheme::kLanes + (count % Scheme::kLanes != 0);
  const int4 sentinel = {101, 103, 107, 109};
  std::vector<int4> outputs[2] = {
      std::vector<int4>(blocks + 2, sentinel), std::vector<int4>(blocks + 2, sentinel)};
  for (int party = 0; party < 2; ++party) {
    calls = 0;
    parallel = false;
    scheme.EvalRange(party, first, last, seeds[party], cws,
        cuda::std::span<int4>(outputs[party].data() + 1, blocks + 1), depth);
    if (depth == 0) EXPECT_FALSE(parallel.load());
#ifdef _OPENMP
    if (depth > 0 && blocks > 1 && omp_get_max_threads() > 1) EXPECT_TRUE(parallel.load());
#endif
    const size_t source_blocks = static_cast<size_t>(last / Scheme::kLanes - first / Scheme::kLanes) + 1;
    if (depth == 0 && source_blocks > 1) EXPECT_LT(calls.load(), source_blocks * (Scheme::kDepth + 1));
    EXPECT_TRUE(EqualInt4(outputs[party].front(), sentinel));
    EXPECT_TRUE(EqualInt4(outputs[party].back(), sentinel));
    for (size_t j = 0; j < count; ++j) {
      const In x = static_cast<In>(first + j);
      const uint64_t expected = Scheme::Extract(scheme.Eval(party, seeds[party], cws, x), x);
      EXPECT_EQ(Scheme::Extract(outputs[party][1 + j / Scheme::kLanes], static_cast<In>(j)), expected);
    }
    for (size_t j = count; j < blocks * Scheme::kLanes; ++j) {
      EXPECT_EQ(Scheme::Extract(outputs[party][1 + j / Scheme::kLanes], static_cast<In>(j)), 0);
    }
  }
  for (size_t j = 0; j < count; ++j) {
    const int4 reconstructed = fss::util::Xor(
        outputs[0][1 + j / Scheme::kLanes], outputs[1][1 + j / Scheme::kLanes]);
    EXPECT_EQ(Scheme::Extract(reconstructed, static_cast<In>(j)), first + j == alpha ? beta : 0);
  }
}

template <int width>
void CheckPackedRangeBoundaries() {
  using In = uint16_t;
  constexpr In lanes = 128 / width;
  for (int depth : {0, 3, 1000}) {
    for (In first : {In{0}, In{1}, static_cast<In>(lanes - 1), lanes,
             static_cast<In>(lanes + 1), In{255}, In{65535}}) {
      for (In length : {In{1}, In{2}, static_cast<In>(lanes - 1), lanes,
               static_cast<In>(lanes + 1), static_cast<In>(2 * lanes + 1)}) {
        if (static_cast<unsigned>(first) + length > 65536) continue;
        const In last = static_cast<In>(first + length - 1);
        CheckPackedRange<16, width>(first, last, static_cast<In>(first + length / 2), depth);
        CheckPackedRange<16, width>(first, last, static_cast<In>(last ^ 32768), depth);
      }
    }
  }
}

TEST(PackedHalfTreeRange, EverySmallClosedInterval) {
  for (uint8_t first = 0; first < 32; ++first) {
    for (uint8_t last = first; last < 32; ++last) {
      for (int depth : {0, 2, 1000}) {
        CheckPackedRange<5, 8>(first, last, uint8_t{17}, depth);
      }
    }
  }
}

TEST(PackedHalfTreeRange, AllWidthsAndNonalignedBoundaries) {
  CheckPackedRangeBoundaries<1>();
  CheckPackedRangeBoundaries<2>();
  CheckPackedRangeBoundaries<4>();
  CheckPackedRangeBoundaries<8>();
  CheckPackedRangeBoundaries<16>();
  CheckPackedRangeBoundaries<32>();
  CheckPackedRangeBoundaries<64>();
}

TEST(PackedHalfTreeRange, WideCoordinatesAndLastInput) {
  using In = __uint128_t;
  const In high = (In{1} << 100) + 3;
  const In last = ~In{0};
  for (int depth : {0, 3, 1000}) {
    CheckPackedRange<65, 1>((In{1} << 64) - 3, (In{1} << 64) + 5, In{1} << 64, depth);
    CheckPackedRange<128, 1>(high, high + 130, high + 5, depth);
    CheckPackedRange<128, 8>(last - 18, last, last, depth);
    CheckPackedRange<128, 64>(last - 12, last, last, depth);
    CheckPackedRange<128, 1>(last, last, last, depth);
  }
}

#ifndef NDEBUG
TEST(PackedHalfTreeRange, RejectsInvalidPreconditions) {
  using Scheme = fss::PackedHalfTreeDpf<8, 8, fss_test::TestCcrHash, uint16_t>;
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  Scheme::Cw cws[Scheme::kDepth + 1]{};
  int4 seed{};
  int4 outputs[1]{};
  auto ys = cuda::std::span<int4>(outputs, 1);
  EXPECT_DEATH(scheme.EvalRange(false, 0, 0, seed, cws, ys, -1), "par_depth");
  EXPECT_DEATH(scheme.EvalRange(false, 2, 1, seed, cws, ys), "first");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 256, seed, cws, ys), "last");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 16, seed, cws, ys), "ys.size");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 0, seed, nullptr, ys), "cws");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 0, seed, cws,
      cuda::std::span<int4>(static_cast<int4 *>(nullptr), size_t{1})), "ys.data");
  using Wide = fss::PackedHalfTreeDpf<128, 1, fss_test::TestCcrHash, __uint128_t>;
  Wide wide{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  Wide::Cw wide_cws[Wide::kDepth + 1]{};
  EXPECT_DEATH(wide.EvalRange(false, 0, ~__uint128_t{0}, seed, wide_cws, ys), "block_distance");
}
#endif  // NDEBUG
}  // namespace

namespace {

constexpr int4 kSplitSentinel = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};

// Checks the split-corrections API against the Cw forms for one configuration:
// every correction entry, every block of Eval, EvalAll and EvalRange, the
// untouched gap entries, and the two-party reconstruction. stride == 0 selects
// the default-argument call.
template <int in_bits, int out_bits, typename In>
void CheckPackedHalfTreeSplit(In alpha, uint64_t beta, size_t stride) {
  using Scheme = fss::PackedHalfTreeDpf<in_bits, out_bits, fss_test::TestCcrHash, In>;
  constexpr int kDepth = Scheme::kDepth;
  constexpr int kLanes = Scheme::kLanes;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const bool default_stride = stride == 0;
  const size_t level_stride = default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(kDepth) * level_stride + 1;
  constexpr size_t kExtra = 3;
  constexpr size_t kDomain = size_t{1} << in_bits;
  constexpr size_t kBlocks = size_t{1} << kDepth;

  typename Scheme::Cw ref[kDepth + 1];
  scheme.Gen(ref, seeds, alpha, beta);

  std::vector<int4> cws(used + kExtra, kSplitSentinel);
  const auto cws_out = cuda::std::span<int4>(cws.data(), cws.size());
  if (default_stride) scheme.Gen(cws_out, seed_span, alpha, beta);
  else scheme.Gen(cws_out, seed_span, alpha, beta, stride);

  for (int l = 0; l <= kDepth; ++l) {
    const size_t index = static_cast<size_t>(l) * level_stride;
    EXPECT_EQ(memcmp(&cws[index], &ref[l].s, sizeof(int4)), 0) << "cws layer " << l;
  }
  // Only the described prefix is written: gap entries and trailing capacity keep the sentinel.
  for (size_t i = 0; i < cws.size(); ++i) {
    if (i % level_stride == 0 && i / level_stride <= static_cast<size_t>(kDepth)) continue;
    EXPECT_EQ(memcmp(&cws[i], &kSplitSentinel, sizeof(int4)), 0) << "cws untouched at " << i;
  }

  const cuda::std::span<const int4> split_cws(cws.data(), used);
  std::vector<int4> split_all[2];
  for (int party = 0; party < 2; ++party) {
    const bool b = party != 0;
    const int4 s0 = seeds[party];
    for (size_t x = 0; x < kDomain; ++x) {
      const In xi = static_cast<In>(x);
      const int4 aos = scheme.Eval(b, s0, ref, xi);
      const int4 got = default_stride
          ? scheme.Eval(b, s0, split_cws, xi)
          : scheme.Eval(b, s0, split_cws, xi, stride);
      EXPECT_EQ(memcmp(&aos, &got, sizeof(int4)), 0) << "Eval party=" << party << " x=" << x;
    }

    std::vector<int4> aos_all(kBlocks, kSplitSentinel);
    split_all[party].assign(kBlocks + 2, kSplitSentinel);
    scheme.EvalAll(b, s0, ref, aos_all.data());
    if (default_stride) scheme.EvalAll(b, s0, split_cws, split_all[party].data());
    else scheme.EvalAll(b, s0, split_cws, split_all[party].data(), 0, stride);
    EXPECT_EQ(memcmp(aos_all.data(), split_all[party].data(), kBlocks * sizeof(int4)), 0)
        << "EvalAll party=" << party;
    EXPECT_EQ(memcmp(&split_all[party][kBlocks], &kSplitSentinel, sizeof(int4)), 0)
        << "EvalAll extra capacity party=" << party;

    // A full-domain interval and a partial one that starts and ends unaligned.
    for (int interval = 0; interval < 2; ++interval) {
      const size_t first = interval == 0 ? 0 : 3;
      const size_t last = interval == 0 ? kDomain - 1 : kDomain - 5;
      const size_t out_blocks = (last - first) / kLanes + 1;
      std::vector<int4> aos_range(out_blocks + 2, kSplitSentinel);
      std::vector<int4> split_range(out_blocks + 2, kSplitSentinel);
      scheme.EvalRange(b, static_cast<In>(first), static_cast<In>(last), s0, ref,
          cuda::std::span<int4>(aos_range.data(), out_blocks), 0);
      const auto ys = cuda::std::span<int4>(split_range.data(), out_blocks);
      if (default_stride) {
        scheme.EvalRange(b, static_cast<In>(first), static_cast<In>(last), s0, split_cws, ys, 0);
      } else {
        scheme.EvalRange(b, static_cast<In>(first), static_cast<In>(last), s0, split_cws, ys,
            0, stride);
      }
      EXPECT_EQ(memcmp(aos_range.data(), split_range.data(), out_blocks * sizeof(int4)), 0)
          << "EvalRange party=" << party << " first=" << first;
      for (size_t i = out_blocks; i < split_range.size(); ++i) {
        EXPECT_EQ(memcmp(&split_range[i], &kSplitSentinel, sizeof(int4)), 0)
            << "EvalRange extra capacity at " << i;
      }
    }
  }

  for (size_t x = 0; x < kDomain; ++x) {
    const int4 block = fss::util::Xor(split_all[0][x / kLanes], split_all[1][x / kLanes]);
    const uint64_t actual = Scheme::Extract(block, static_cast<In>(x));
    const uint64_t expected = static_cast<In>(x) == alpha ? beta : 0;
    EXPECT_EQ(actual, expected) << "reconstruction x=" << x;
  }
}

template <int in_bits, int out_bits>
void CheckPackedHalfTreeSplitStrides(uint64_t beta) {
  constexpr uint64_t kDomain = uint64_t{1} << in_bits;
  const uint64_t alpha = 107 % kDomain;
  for (size_t stride : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
    CheckPackedHalfTreeSplit<in_bits, out_bits, uint16_t>(static_cast<uint16_t>(alpha), beta, stride);
  }
}

}  // namespace

// kDepth 1 is the shallowest packed tree, 4 is the smallest frontier path, and
// the wider output widths cover the lane counts. The 64-bit width also exercises
// one-lane blocks.
TEST(PackedHalfTreeDpfSplitCorrectionsTest, Depth1Width1) {
  CheckPackedHalfTreeSplitStrides<8, 1>(1);
}
TEST(PackedHalfTreeDpfSplitCorrectionsTest, Depth4Width1) {
  CheckPackedHalfTreeSplitStrides<11, 1>(1);
}
TEST(PackedHalfTreeDpfSplitCorrectionsTest, Depth6Width2) {
  CheckPackedHalfTreeSplitStrides<12, 2>(3);
}
TEST(PackedHalfTreeDpfSplitCorrectionsTest, Depth8Width8) {
  CheckPackedHalfTreeSplitStrides<12, 8>(0x5a);
}
TEST(PackedHalfTreeDpfSplitCorrectionsTest, Depth10Width64) {
  CheckPackedHalfTreeSplitStrides<11, 64>(0x0123456789abcdefULL);
}
TEST(PackedHalfTreeDpfSplitCorrectionsTest, WideDomainWidth16) {
  CheckPackedHalfTreeSplit<16, 16, uint16_t>(12345, 0xbeef, 0);
  CheckPackedHalfTreeSplit<16, 16, uint16_t>(12345, 0xbeef, 2);
}

#ifndef NDEBUG
TEST(PackedHalfTreeDpfSplitCorrectionsTest, RejectsInvalidInputs) {
  using Scheme = fss::PackedHalfTreeDpf<10, 1, fss_test::TestCcrHash, uint16_t>;
  constexpr int kDepth = Scheme::kDepth;
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  int4 cws[kDepth + 1] = {};
  int4 ys[8] = {};
  const auto cws_out = cuda::std::span<int4>(cws, kDepth + 1);
  const auto cws_in = cuda::std::span<const int4>(cws, kDepth + 1);
  const int4 s0 = seeds[0];

  // A zero stride is only the batch sentinel, so a non-batch method rejects it.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cws_out, seed_span, uint16_t{5}, 1, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, uint16_t{0}, 0), "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cws_in, ys, 0, 0), "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalRange(false, uint16_t{0}, uint16_t{7}, s0, cws_in,
                        cuda::std::span<int4>(ys, 8), 0, 0),
      "Assertion.*failed");
  // A span that omits the packed-output correction at cws[kDepth*stride] is rejected.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cuda::std::span<int4>(cws, kDepth), seed_span,
                        uint16_t{5}, 1),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(cws, kDepth),
                        uint16_t{0}),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(cws, kDepth), ys),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalRange(false, uint16_t{0}, uint16_t{7}, s0,
                        cuda::std::span<const int4>(cws, kDepth),
                        cuda::std::span<int4>(ys, 8)),
      "Assertion.*failed");
  // A stride past the addressable guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / kDepth + 1;
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, uint16_t{0}, overflowing_stride),
      "Assertion.*failed");
}
#endif  // NDEBUG

// The batch geometry validation is host code that runs before any CUDA call, so
// these checks need no device and read no buffer.
TEST(PackedHalfTreeDpfSplitGpuTest, RejectsInvalidBatchGeometry) {
  // out_bits 64 gives one-lane blocks, so kDepth is 32 for in_bits 33.
  using Wide = fss::PackedHalfTreeDpf<33, 64, fss_test::TestCcrHash, uint64_t>;
  static_assert(Wide::kDepth == 32);
  Wide wide{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  int4 seed_storage[1] = {};
  int4 cws_storage[1] = {};
  int4 ys_storage[1] = {};

  // 2^32 keys of 2^32 blocks make the output product wrap in size_t, so the guard
  // must reject the call before the product is formed. The correction span is
  // sized nominally here and the key count is above the launch grid limit for
  // every launch shape, so a variant without the output guard would stop at
  // cudaErrorInvalidConfiguration instead of launching into the one-element
  // output buffer.
  constexpr size_t kKeys = size_t{1} << 32;
  constexpr size_t kMinCorrections = 32 * kKeys + kKeys;
  EXPECT_EQ((wide.template EvalAllGpuBatch<24, 256>(false,
      cuda::std::span<const int4>(seed_storage, kKeys),
      cuda::std::span<const int4>(cws_storage, kMinCorrections),
      cuda::std::span<int4>(ys_storage, 1), 0)), cudaErrorInvalidValue);

  using Small = fss::PackedHalfTreeDpf<8, 1, fss_test::TestCcrHash, uint16_t>;
  static_assert(Small::kDepth == 1);
  Small small{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  constexpr size_t kSmallKeys = 2;
  constexpr size_t kSmallRequired = kSmallKeys + kSmallKeys;
  int4 small_s0s[kSmallKeys] = {};
  int4 small_cws[kSmallRequired] = {};
  int4 small_ys[kSmallKeys * 2] = {};
  const auto s0s = cuda::std::span<const int4>(small_s0s, kSmallKeys);
  const auto cws = cuda::std::span<const int4>(small_cws, kSmallRequired);
  const auto ys = cuda::std::span<int4>(small_ys, kSmallKeys * 2);

  // A stride below the key count is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s, cws, ys, 0, 1)),
      cudaErrorInvalidValue);
  // A stride past the element guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s, cws, ys, 0,
                        kMaxElements + 1)),
      cudaErrorInvalidValue);
  // A correction span that omits the terminal column is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s,
                        cuda::std::span<const int4>(small_cws, kSmallRequired - 1), ys)),
      cudaErrorInvalidValue);
  // An empty batch succeeds without reading any buffer.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, cuda::std::span<const int4>(),
                        cuda::std::span<const int4>(), cuda::std::span<int4>())),
      cudaSuccess);
}

// The GPU split launches are compile-verified here and executed only on a
// machine with a free device, comparing against the host Cw-form outputs.
namespace {

void CheckPackedHalfTreeSplitGpuLaunches() {
  int devices = 0;
  cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
      (device_status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(device_status, cudaSuccess);

  constexpr int kBits = 8;
  constexpr size_t kDomain = size_t{1} << kBits;
  using Scheme = fss::PackedHalfTreeDpf<kBits, 1, fss_test::TestCcrHash, uint16_t>;
  constexpr int kDepth = Scheme::kDepth;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  Scheme scheme{fss_test::TestCcrHash{}, {11, 22, 33, 44}};
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;
  const uint64_t beta = 1;
  constexpr size_t kStride = 2;

  typename Scheme::Cw ref[kDepth + 1];
  scheme.Gen(ref, seeds, alpha, beta);
  std::vector<int4> cws((kDepth + 1) * kStride + 2, kSplitSentinel);
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()),
      cuda::std::span<const int4, 2>(seeds, 2), alpha, beta, kStride);

  int4 *device_cws = nullptr;
  int4 *device_ys = nullptr;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  constexpr size_t kKeys = 2;
  ASSERT_EQ(cudaMalloc(&device_cws, cws.size() * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, kBlocks * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_cws, cws.data(), cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  const auto span_cws = cuda::std::span<const int4>(device_cws, cws.size());

  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(party);
    const bool b = party != 0;
    std::vector<int4> actual(kBlocks, kSplitSentinel);
    ASSERT_EQ((scheme.template EvalAllGpu<8, 16>(b, seeds[party], span_cws,
        cuda::std::span<int4>(device_ys, kBlocks), stream, kStride)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_ys, kBlocks * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t j = 0; j < kBlocks; ++j) {
      const int4 expected = scheme.Eval(b, seeds[party], ref, static_cast<uint16_t>(j * 128));
      EXPECT_EQ(memcmp(&actual[j], &expected, sizeof(int4)), 0) << "block=" << j;
    }
  }

  // A zero stride and a span that omits the terminal are rejected.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, seeds[0], span_cws,
      cuda::std::span<int4>(device_ys, kBlocks), stream, 0)), cudaErrorInvalidValue);

  // Batch launches: the default stride is the dense level-major key count.
  typename Scheme::Cw ref_batch[kKeys][kDepth + 1];
  std::vector<int4> batch_cws((kDepth + 1) * kKeys, kSplitSentinel);
  int4 batch_s0s[kKeys];
  for (size_t key = 0; key < kKeys; ++key) {
    batch_s0s[key] = seeds[key];
    scheme.Gen(ref_batch[key], seeds, static_cast<uint16_t>(alpha + key), beta);
    // Offsetting by the key index fills the dense level-major layout.
    scheme.Gen(cuda::std::span<int4>(batch_cws.data() + key, batch_cws.size() - key),
        cuda::std::span<const int4, 2>(seeds, 2), static_cast<uint16_t>(alpha + key), beta, kKeys);
  }
  ASSERT_EQ(cudaMemcpyAsync(device_cws, batch_cws.data(), batch_cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  int4 *device_s0s = nullptr;
  ASSERT_EQ(cudaMalloc(&device_s0s, sizeof(batch_s0s)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_s0s, batch_s0s, sizeof(batch_s0s),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  int4 *device_batch_ys = nullptr;
  ASSERT_EQ(cudaMalloc(&device_batch_ys, kKeys * kBlocks * sizeof(int4)), cudaSuccess);
  const auto span_batch_cws = cuda::std::span<const int4>(device_cws, batch_cws.size());
  const auto span_batch_ys = cuda::std::span<int4>(device_batch_ys, kKeys * kBlocks);
  for (int use_sentinel = 0; use_sentinel < 2; ++use_sentinel) {
    SCOPED_TRACE(use_sentinel);
    const size_t stride = use_sentinel == 0 ? fss::kDenseStride : kKeys;
    ASSERT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false,
        cuda::std::span<const int4>(device_s0s, kKeys), span_batch_cws, span_batch_ys,
        stream, stride)), cudaSuccess);
    std::vector<int4> actual(kKeys * kBlocks, kSplitSentinel);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_batch_ys, actual.size() * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t key = 0; key < kKeys; ++key) {
      for (size_t j = 0; j < kBlocks; ++j) {
        const int4 expected =
            scheme.Eval(false, batch_s0s[key], ref_batch[key], static_cast<uint16_t>(j * 128));
        EXPECT_EQ(memcmp(&actual[key * kBlocks + j], &expected, sizeof(int4)), 0)
            << "key=" << key << " block=" << j;
      }
    }
  }
  // A stride below the key count is rejected.
  EXPECT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false,
      cuda::std::span<const int4>(device_s0s, kKeys), span_batch_cws, span_batch_ys,
      stream, 1)), cudaErrorInvalidValue);

  EXPECT_EQ(cudaFree(device_batch_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_s0s), cudaSuccess);
  EXPECT_EQ(cudaFree(device_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_cws), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace

TEST(PackedHalfTreeDpfSplitGpuTest, SplitLaunchesMatchCwHost) {
  CheckPackedHalfTreeSplitGpuLaunches();
}
