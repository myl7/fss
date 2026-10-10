#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;

static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

static bool EqualInt4(int4 a, int4 b) {
  return memcmp(&a, &b, sizeof(int4)) == 0;
}

// ChaCha-based test fixture over all supported output widths
template <int OutBits>
class PackedHalfTreeDpfChaChaTest : public ::testing::Test {
protected:
  using Prg = fss::prg::ChaCha<1>;
  using DpfType = fss::PackedHalfTreeDpf<kAlphaBits, OutBits, Prg, uint16_t>;

  static constexpr uint64_t kMaxBeta = OutBits == 64 ? ~0ULL : (1ULL << OutBits) - 1;

  int4 s0s[2];
  typename DpfType::Cw cws[DpfType::kDepth];
  int4 fcw;
  Prg prg;
  int4 hash_key;

  PackedHalfTreeDpfChaChaTest() : prg(gChaChaDeviceNonces) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  void TestEvalAtAlpha() {
    DpfType dpf{prg, hash_key};
    dpf.Gen(cws, fcw, s0s, kAlpha, kMaxBeta);

    int4 y0 = dpf.Eval(false, s0s[0], cws, fcw, kAlpha);
    int4 y1 = dpf.Eval(true, s0s[1], cws, fcw, kAlpha);

    int4 diff = fss::util::Xor(y0, y1);
    EXPECT_TRUE(EqualInt4(diff, DpfType::PackBeta(kAlpha, kMaxBeta)));
    EXPECT_EQ(DpfType::Extract(diff, kAlpha), kMaxBeta);
  }

  void TestEvalAtNonAlpha() {
    DpfType dpf{prg, hash_key};
    dpf.Gen(cws, fcw, s0s, kAlpha, kMaxBeta);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    int4 zero = {0, 0, 0, 0};

    // Same block as alpha, different lane: the block reconstructs to the
    // packed beta, and x's lane within it is zero.
    uint16_t x = kAlpha ^ 1;
    int4 y0 = dpf.Eval(false, s0s[0], cws, fcw, x);
    int4 y1 = dpf.Eval(true, s0s[1], cws, fcw, x);
    EXPECT_TRUE(EqualInt4(fss::util::Xor(y0, y1), DpfType::PackBeta(kAlpha, kMaxBeta))) << "Failed at x=" << x;
    EXPECT_EQ(DpfType::Extract(fss::util::Xor(y0, y1), x), 0) << "Failed at x=" << x;

    for (int i = 0; i < 100; i++) {
      x = dis(gen);

      y0 = dpf.Eval(false, s0s[0], cws, fcw, x);
      y1 = dpf.Eval(true, s0s[1], cws, fcw, x);
      int4 diff = fss::util::Xor(y0, y1);
      if (x / DpfType::kLanes == kAlpha / DpfType::kLanes) {
        EXPECT_TRUE(EqualInt4(diff, DpfType::PackBeta(kAlpha, kMaxBeta))) << "Failed at x=" << x;
      } else {
        EXPECT_TRUE(EqualInt4(diff, zero)) << "Failed at x=" << x;
      }
      EXPECT_EQ(DpfType::Extract(diff, x), x == kAlpha ? kMaxBeta : 0) << "Failed at x=" << x;
    }
  }

  void TestEvalAll() {
    DpfType dpf{prg, hash_key};
    dpf.Gen(cws, fcw, s0s, kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks), ys1(blocks);

    dpf.EvalAll(false, s0s[0], cws, fcw, ys0.data());
    dpf.EvalAll(true, s0s[1], cws, fcw, ys1.data());

    size_t alpha_block = kAlpha / DpfType::kLanes;
    int4 expected = DpfType::PackBeta(kAlpha, kMaxBeta);
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
    DpfType dpf{prg, hash_key};
    dpf.Gen(cws, fcw, s0s, kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks);
    dpf.EvalAll(false, s0s[0], cws, fcw, ys0.data());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 50; i++) {
      uint16_t x = dis(gen);
      int4 y0 = dpf.Eval(false, s0s[0], cws, fcw, x);
      EXPECT_TRUE(EqualInt4(y0, ys0[x / DpfType::kLanes])) << "Failed at x=" << x;
    }
  }

  void TestAlphaSweep() {
    DpfType dpf{prg, hash_key};

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
      dpf.Gen(cws, fcw, s0s, alpha, beta);
      dpf.EvalAll(false, s0s[0], cws, fcw, ys0.data());
      dpf.EvalAll(true, s0s[1], cws, fcw, ys1.data());

      size_t alpha_block = alpha / DpfType::kLanes;
      int4 expected = DpfType::PackBeta(alpha, beta);

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

      int4 packed = DpfType::PackBeta(static_cast<uint16_t>(lane), kMaxBeta);
      for (int j = 0; j < DpfType::kLanes; ++j) {
        uint64_t v = DpfType::Extract(packed, static_cast<uint16_t>(j));
        EXPECT_EQ(v, j == lane ? kMaxBeta : 0) << "lane=" << lane << " j=" << j;
      }
    }
  }
};

#define PACKED_HALF_TREE_DPF_CHACHA_TEST(OutBits)                                        \
  using PackedHalfTreeDpf##OutBits##BitChaChaTest = PackedHalfTreeDpfChaChaTest<OutBits>; \
  TEST_F(PackedHalfTreeDpf##OutBits##BitChaChaTest, EvalAtAlpha) {                       \
    TestEvalAtAlpha();                                                                   \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitChaChaTest, EvalAtNonAlpha) {                    \
    TestEvalAtNonAlpha();                                                                \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitChaChaTest, EvalAll) {                           \
    TestEvalAll();                                                                       \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitChaChaTest, EvalMatchesEvalAll) {                \
    TestEvalMatchesEvalAll();                                                            \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitChaChaTest, AlphaSweep) {                        \
    TestAlphaSweep();                                                                    \
  }                                                                                      \
  TEST_F(PackedHalfTreeDpf##OutBits##BitChaChaTest, PackBetaExtract) {                   \
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
  using DpfType = fss::PackedHalfTreeDpf<kAlphaBits, 1, Prg, uint16_t>;

  static constexpr uint64_t kMaxBeta = 1;

  int4 s0s[2];
  typename DpfType::Cw cws[DpfType::kDepth];
  int4 fcw;
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

  Prg MakePrg() {
    uint32_t te0[256];
    uint8_t sbox[256];
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    return Prg(aes_key, te0, sbox);
  }

  void TestEvalAll() {
    Prg prg = MakePrg();
    DpfType dpf{prg, hash_key};
    dpf.Gen(cws, fcw, s0s, kAlpha, kMaxBeta);

    constexpr size_t blocks = 1ULL << DpfType::kDepth;
    std::vector<int4> ys0(blocks), ys1(blocks);

    dpf.EvalAll(false, s0s[0], cws, fcw, ys0.data());
    dpf.EvalAll(true, s0s[1], cws, fcw, ys1.data());

    size_t alpha_block = kAlpha / DpfType::kLanes;
    int4 expected = DpfType::PackBeta(kAlpha, kMaxBeta);
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

// Edge cases: kDepth == 1, the shallowest legal trees
template <int InBits, int OutBits>
class PackedHalfTreeDpfOneLevelTest : public ::testing::Test {
protected:
  using Prg = fss::prg::ChaCha<1>;
  using In = uint8_t;
  using DpfType = fss::PackedHalfTreeDpf<InBits, OutBits, Prg, In>;

  static constexpr uint64_t kMaxBeta = OutBits == 64 ? ~0ULL : (1ULL << OutBits) - 1;
  static_assert(DpfType::kDepth == 1);

  int4 s0s[2];
  typename DpfType::Cw cws[DpfType::kDepth];
  int4 fcw;
  Prg prg;
  int4 hash_key;

  PackedHalfTreeDpfOneLevelTest() : prg(gChaChaDeviceNonces) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  void TestEvalAll() {
    DpfType dpf{prg, hash_key};

    int4 zero = {0, 0, 0, 0};
    for (int alpha = 0; alpha < (1 << InBits); ++alpha) {
      dpf.Gen(cws, fcw, s0s, static_cast<In>(alpha), kMaxBeta);

      int4 ys0[2], ys1[2];
      dpf.EvalAll(false, s0s[0], cws, fcw, ys0);
      dpf.EvalAll(true, s0s[1], cws, fcw, ys1);

      size_t alpha_block = alpha / DpfType::kLanes;
      int4 expected = DpfType::PackBeta(static_cast<In>(alpha), kMaxBeta);
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

// Layout constants and lane known vectors
TEST(PackedHalfTreeDpfLayout, Constants) {
  using D1 = fss::PackedHalfTreeDpf<16, 1, fss::prg::ChaCha<1>, uint16_t>;
  static_assert(D1::kLanes == 128);
  static_assert(D1::kDepth == 9);
  static_assert(sizeof(D1::Cw) == 16);

  using D8 = fss::PackedHalfTreeDpf<16, 8, fss::prg::ChaCha<1>, uint16_t>;
  static_assert(D8::kLanes == 16);
  static_assert(D8::kDepth == 12);

  using D64 = fss::PackedHalfTreeDpf<20, 64, fss::prg::ChaCha<1>, uint32_t>;
  static_assert(D64::kLanes == 2);
  static_assert(D64::kDepth == 19);

  SUCCEED();
}

TEST(PackedHalfTreeDpfLayout, LaneKnownVectors) {
  using D1 = fss::PackedHalfTreeDpf<16, 1, fss::prg::ChaCha<1>, uint16_t>;
  int4 p = D1::PackBeta(5, 1);
  EXPECT_EQ(p.x, 1 << 5);
  EXPECT_EQ(p.y, 0);
  EXPECT_EQ(p.z, 0);
  EXPECT_EQ(p.w, 0);

  using D8 = fss::PackedHalfTreeDpf<16, 8, fss::prg::ChaCha<1>, uint16_t>;
  // Lane 3 of 16: byte lane 3 lives at bits 24..31 of the low word
  p = D8::PackBeta(3, 0xAB);
  EXPECT_EQ(p.x, 0xAB << 24);
  EXPECT_EQ(D8::Extract(p, 3), 0xAB);
  EXPECT_EQ(D8::Extract(p, 2), 0);

  using D64 = fss::PackedHalfTreeDpf<16, 64, fss::prg::ChaCha<1>, uint16_t>;
  int4 q = D64::PackBeta(1, 0x1122334455667788ULL);
  EXPECT_EQ(q.x, 0);
  EXPECT_EQ(q.y, 0);
  EXPECT_EQ(q.z, static_cast<int>(0x55667788u));
  EXPECT_EQ(q.w, static_cast<int>(0x11223344u));
  EXPECT_EQ(D64::Extract(q, 1), 0x1122334455667788ULL);
  EXPECT_EQ(D64::Extract(q, 0), 0);
}
