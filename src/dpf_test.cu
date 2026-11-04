#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prg/aes128_mmo_raw.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include "core_api_test_helpers.cuh"
#include "range_gpu_test_helpers.cuh"
#include <atomic>
#include <thread>

static_assert(!fss::ValidForInputBits<bool, 1>);
static_assert(!fss::ValidForInputBits<const bool, 1>);
static_assert(!fss::ValidForInputBits<volatile bool, 1>);
static_assert(!fss::ValidForInputBits<const volatile bool, 1>);
static_assert(!fss::ValidForInputBits<bool, 8>);
static_assert(fss::ValidForInputBits<uint8_t, 8>);
static_assert(fss::ValidForInputBits<uint64_t, 64>);
static_assert(fss::ValidForInputBits<__uint128_t, 128>);
static_assert(!fss::ValidForInputBits<__uint128_t, 129>);

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
using Uint64Group = fss::group::Uint<uint64_t>;

// Helper to determine comparison size for each group type
template <typename Group>
constexpr size_t GroupCompareSize() {
  if constexpr (std::is_same_v<Group, Uint64Group>) {
    return 8;  // Compare only first 8 bytes for uint64
  } else if constexpr (std::is_same_v<Group, Uint127Group>) {
    return 16;  // Compare 16 bytes (127 bits + control bit which is always 0)
  } else {
    return sizeof(int4);  // Full 16 bytes for Bytes group
  }
}

// Test parameters
constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;
constexpr __uint128_t kBeta = 604;

// Random nonce for ChaCha
constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// Test fixture template for DPF
template <typename Group, typename Prg>
class DpfTestBase : public ::testing::Test {
protected:
  using DpfType = fss::Dpf<kAlphaBits, Group, Prg, uint16_t>;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;

  DpfTestBase(Prg p) : prg(p) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void TestEvalAtAlpha() {
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    // Eval at alpha
    int4 y0 = fss_test::EvalKey(dpf, false, k0, kAlpha);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, kAlpha);

    auto g0 = Group::From(y0);
    auto g1 = Group::From(y1);
    auto result = g0 + g1;
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();

    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAll() {
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);

    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    auto expected = Group::From(b_buf);
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto g0 = Group::From(ys0[x]);
      auto g1 = Group::From(ys1[x]);
      auto result = g0 + g1;
      int4 result_buf = result.Into();

      if (x == kAlpha) {
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at alpha=" << x;
      } else {
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
      }
    }
  }

  void TestEvalAtNonAlpha() {
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 100; i++) {
      uint16_t x = dis(gen);
      if (x == kAlpha) continue;

      int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
      int4 y1 = fss_test::EvalKey(dpf, true, k1, x);

      auto g0 = Group::From(y0);
      auto g1 = Group::From(y1);
      auto result = g0 + g1;

      // Check that result equals identity (0)
      int4 result_buf = result.Into();
      int4 zero = {0, 0, 0, 0};

      EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }
  }
};

// ChaCha PRG tests
class DpfBytesChaChaTest : public DpfTestBase<BytesGroup, fss::prg::ChaCha<2>> {
public:
  DpfBytesChaChaTest() : DpfTestBase(fss::prg::ChaCha<2>(kNonce0, kNonce1)) {}
};

class DpfUint128ChaChaTest : public DpfTestBase<Uint127Group, fss::prg::ChaCha<2>> {
public:
  DpfUint128ChaChaTest() : DpfTestBase(fss::prg::ChaCha<2>(kNonce0, kNonce1)) {}
};

class DpfUint64ChaChaTest : public DpfTestBase<Uint64Group, fss::prg::ChaCha<2>> {
public:
  DpfUint64ChaChaTest() : DpfTestBase(fss::prg::ChaCha<2>(kNonce0, kNonce1)) {}
};

// AES PRG tests
class DpfBytesAesTest : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128Mmo<2>;
  using DpfType = fss::Dpf<kAlphaBits, Group, Prg, uint16_t>;

  int4 s0s[2];
  DpfType::Key k0, k1;
  cuda::std::array<EVP_CIPHER_CTX *, 2> ctxs;

  void SetUp() override {
    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const unsigned char *keys[2] = {key0, key1};
    ctxs = Prg::CreateCtxs(keys);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void TearDown() override {
    Prg::FreeCtxs(ctxs);
  }
};

// Tests for ChaCha
TEST_F(DpfBytesChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DpfBytesChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DpfUint128ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DpfUint128ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DpfUint64ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DpfUint64ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DpfBytesChaChaTest, EvalAll) {
  TestEvalAll();
}
TEST_F(DpfUint128ChaChaTest, EvalAll) {
  TestEvalAll();
}
TEST_F(DpfUint64ChaChaTest, EvalAll) {
  TestEvalAll();
}

// Tests for AES
TEST_F(DpfBytesAesTest, EvalAtAlpha) {
  Prg prg(ctxs);
  DpfType dpf{prg};

  int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
      static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

  int4 y0 = fss_test::EvalKey(dpf, false, k0, kAlpha);
  int4 y1 = fss_test::EvalKey(dpf, true, k1, kAlpha);

  auto g0 = Group::From(y0);
  auto g1 = Group::From(y1);
  auto result = g0 + g1;
  auto expected = Group::From(b_buf);

  int4 result_buf = result.Into();
  int4 expected_buf = expected.Into();
  EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
}

TEST_F(DpfBytesAesTest, EvalAtNonAlpha) {
  Prg prg(ctxs);
  DpfType dpf{prg};

  int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
      static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

  for (int i = 0; i < 100; i++) {
    uint16_t x = dis(gen);
    if (x == kAlpha) continue;

    int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, x);

    auto g0 = Group::From(y0);
    auto g1 = Group::From(y1);
    auto result = g0 + g1;

    int4 result_buf = result.Into();
    int4 zero = {0, 0, 0, 0};
    EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
  }
}

// Aes128Mmo has no Gen8(), so EvalAll() runs the frontier through the
// detail::GenBatch() loop; this pins the OpenSSL PRG cell of the EvalAll matrix.
TEST_F(DpfBytesAesTest, EvalAll) {
  Prg prg(ctxs);
  DpfType dpf{prg};

  int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
      static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

  constexpr size_t n = 1ULL << kAlphaBits;
  std::vector<int4> ys0(n), ys1(n);
  fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
  fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

  auto expected = Group::From(b_buf);
  int4 expected_buf = expected.Into();
  int4 zero = {0, 0, 0, 0};

  for (size_t x = 0; x < n; ++x) {
    auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
    int4 result_buf = result.Into();

    if (x == kAlpha) {
      EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at alpha=" << x;
    } else {
      EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }
  }
}

// These tests exercise the 8-wide batched EvalAll() frontier at its in_bits
// and par_depth boundaries with the PRG whose Gen8() the batch pipelines
// through. in_bits below kBatchPrefix + kBatchTail (6) falls back to the
// per-input point walk, in_bits = 6 is the smallest batched domain and clamps
// the split depth to 0, and larger in_bits run both the sequential batched
// walk (par_depth = 0) and the frontier-width split (par_depth > 0).
template <int in_bits, int par_depth = 0>
class DpfAesRawTestBase : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128MmoRaw<2>;
  using DpfType = fss::Dpf<in_bits, Group, Prg, uint16_t>;

  static constexpr uint8_t kKeys[2][16] = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
  };

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;

  DpfAesRawTestBase() : prg(kKeys) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  int4 MakeBBuf() {
    return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  }

  // Full-domain share sum: beta at alpha, identity elsewhere.
  void TestEvalAll(uint16_t alpha) {
    DpfType dpf{prg};

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    constexpr size_t n = 1ULL << in_bits;
    std::vector<int4> ys0(n), ys1(n);
    fss_test::EvalAllKey<par_depth>(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey<par_depth>(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    auto expected = Group::From(b_buf);
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
      int4 result_buf = result.Into();

      if (x == alpha) {
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at alpha=" << x;
      } else {
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
      }
    }
  }

  // Bit-exactness of the batched EvalAll() against the scalar point Eval().
  void TestEvalAllMatchesEval(uint16_t alpha) {
    DpfType dpf{prg};

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    constexpr size_t n = 1ULL << in_bits;
    std::vector<int4> ys0(n), ys1(n);
    fss_test::EvalAllKey<par_depth>(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey<par_depth>(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    for (size_t x = 0; x < n; ++x) {
      int4 e0 = fss_test::EvalKey(dpf, false, k0, static_cast<uint16_t>(x));
      int4 e1 = fss_test::EvalKey(dpf, true, k1, static_cast<uint16_t>(x));
      EXPECT_EQ(memcmp(&ys0[x], &e0, sizeof(int4)), 0) << "Party 0 mismatch at x=" << x;
      EXPECT_EQ(memcmp(&ys1[x], &e1, sizeof(int4)), 0) << "Party 1 mismatch at x=" << x;
    }
  }
};

class DpfBytesAesRaw5BitsTest : public DpfAesRawTestBase<5> {};
class DpfBytesAesRaw6BitsSeqTest : public DpfAesRawTestBase<6, 0> {};
class DpfBytesAesRaw6BitsClampedSplitTest : public DpfAesRawTestBase<6, 2> {};
class DpfBytesAesRaw8BitsSeqTest : public DpfAesRawTestBase<8, 0> {};
class DpfBytesAesRaw8BitsSplitTest : public DpfAesRawTestBase<8, 2> {};
class DpfBytesAesRaw16BitsDefaultTest : public DpfAesRawTestBase<16> {};

TEST_F(DpfBytesAesRaw5BitsTest, EvalAll) {
  constexpr uint16_t kAlpha5 = 11;
  TestEvalAll(kAlpha5);
  TestEvalAllMatchesEval(kAlpha5);
}
TEST_F(DpfBytesAesRaw6BitsSeqTest, EvalAll) {
  constexpr uint16_t kAlpha6 = 43;
  TestEvalAll(kAlpha6);
  TestEvalAllMatchesEval(kAlpha6);
}
TEST_F(DpfBytesAesRaw6BitsClampedSplitTest, EvalAll) {
  constexpr uint16_t kAlpha6 = 43;
  TestEvalAll(kAlpha6);
  TestEvalAllMatchesEval(kAlpha6);
}
TEST_F(DpfBytesAesRaw8BitsSeqTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DpfBytesAesRaw8BitsSplitTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DpfBytesAesRaw16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}

// Alpha at the domain edges: 0 and 2 ** in_bits - 1.
TEST_F(DpfBytesAesRaw8BitsSeqTest, EvalAllEdgeAlpha) {
  TestEvalAll(0);
  TestEvalAllMatchesEval(0);
  TestEvalAll((1U << 8) - 1);
  TestEvalAllMatchesEval((1U << 8) - 1);
}

// Aes128Soft has no Gen8(), so EvalAll() runs the frontier through the
// detail::GenBatch() loop; this pins the software AES cell of the EvalAll matrix.
class DpfBytesAesSoftTest : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128Soft<2>;
  using DpfType = fss::Dpf<kAlphaBits, Group, Prg, uint16_t>;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  uint8_t aes_key[2][16] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}};

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void TestEvalAll() {
    uint32_t te0[256];
    uint8_t sbox[256];
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    Prg prg(aes_key, te0, sbox);
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);
    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    auto expected = Group::From(b_buf);
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
      int4 result_buf = result.Into();

      if (x == kAlpha) {
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at alpha=" << x;
      } else {
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
      }
    }
  }
};

TEST_F(DpfBytesAesSoftTest, EvalAll) {
  TestEvalAll();
}

// Below kBatchPrefix + kBatchTail (6) bits, EvalAll() evaluates per input; cover
// the point-walk fallback with a PRG that has no batched entry.
TEST(DpfSmallDomainTest, EvalAllChaChaBelowBatchDepth) {
  using Prg = fss::prg::ChaCha<2>;
  using DpfType = fss::Dpf<5, BytesGroup, Prg, uint8_t>;
  constexpr uint8_t kAlpha5 = 11;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis;
  int4 s0s[2] = {{dis(gen), dis(gen), dis(gen), dis(gen) & ~1},
      {dis(gen), dis(gen), dis(gen), dis(gen) & ~1}};

  Prg prg(kNonce0, kNonce1);
  DpfType dpf{prg};

  int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
      static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  typename DpfType::Key k0, k1;
  fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha5, b_buf);

  constexpr size_t n = 1ULL << 5;
  int4 ys0[n], ys1[n];
  fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0, n));
  fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1, n));

  int4 zero = {0, 0, 0, 0};
  for (size_t x = 0; x < n; ++x) {
    int4 e0 = fss_test::EvalKey(dpf, false, k0, static_cast<uint8_t>(x));
    int4 e1 = fss_test::EvalKey(dpf, true, k1, static_cast<uint8_t>(x));
    EXPECT_EQ(memcmp(&ys0[x], &e0, sizeof(int4)), 0) << "Party 0 mismatch at x=" << x;
    EXPECT_EQ(memcmp(&ys1[x], &e1, sizeof(int4)), 0) << "Party 1 mismatch at x=" << x;

    auto result = BytesGroup::From(ys0[x]) + BytesGroup::From(ys1[x]);
    int4 result_buf = result.Into();
    int4 expected_buf = x == kAlpha5 ? BytesGroup::From(b_buf).Into() : zero;
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, sizeof(int4)), 0) << "Reconstruction mismatch at x=" << x;
  }
}


namespace {

template <int bits, typename Group, typename Prg, typename In>
void CheckDpfRange(Prg prg, In first, In last, In alpha, int par_depth) {
  using Scheme = fss::Dpf<bits, Group, Prg, In>;
  Scheme scheme(prg);
  typename Scheme::Cw cws[bits + 1];
  const int4 seeds[2] = {{0x12345678, 0x1234abcd, 0x13579bdf, 0x2468ace0},
      {0x7abc1234, 0x01bc3456, 0x4567abcd, 0x76892344}};
  const int4 beta = Group::From(make_int4(0x5abc0123, 0x44556677, 0x12345678, 0x6abc1234)).Into();
  scheme.Gen(cws, seeds, alpha, beta);
  const size_t count = static_cast<size_t>(static_cast<__uint128_t>(last) - first) + 1;
  const int4 sentinel = make_int4(0x9876, 0x1234, 0x5678, 0xabcd);
  std::vector<int4> outputs[2] = {
      std::vector<int4>(count + 3, sentinel), std::vector<int4>(count + 3, sentinel)};
  for (int party = 0; party < 2; ++party) {
    scheme.EvalRange(party, first, last, seeds[party], cws,
        cuda::std::span<int4>(outputs[party].data() + 1, count + 2), par_depth);
    EXPECT_EQ(memcmp(&outputs[party][0], &sentinel, sizeof(int4)), 0);
    EXPECT_EQ(memcmp(&outputs[party][count + 1], &sentinel, sizeof(int4)), 0);
    EXPECT_EQ(memcmp(&outputs[party][count + 2], &sentinel, sizeof(int4)), 0);
    for (size_t offset = 0; offset < count; ++offset) {
      const In point = first + static_cast<In>(offset);
      const int4 scalar = scheme.Eval(party, seeds[party], cws, point);
      EXPECT_EQ(memcmp(&outputs[party][offset + 1], &scalar, sizeof(int4)), 0)
          << "party=" << party << " offset=" << offset << " par_depth=" << par_depth;
    }
  }
  for (size_t offset = 0; offset < count; ++offset) {
    const In point = first + static_cast<In>(offset);
    const int4 actual = (Group::From(outputs[0][offset + 1]) + Group::From(outputs[1][offset + 1])).Into();
    const int4 expected = point == alpha ? beta : Group().Into();
    EXPECT_EQ(memcmp(&actual, &expected, sizeof(int4)), 0) << "offset=" << offset;
  }
}

template <typename Group, typename Prg>
void CheckDpfRangeSuite(Prg prg) {
  for (unsigned first = 0; first < 32; ++first) {
    for (unsigned last = first; last < 32; ++last) {
      CheckDpfRange<5, Group>(prg, static_cast<uint8_t>(first),
          static_cast<uint8_t>(last), uint8_t{11}, 0);
    }
  }
  using Wide = __uint128_t;
  const Wide boundary = Wide{1} << 64;
  const Wide high = (Wide{1} << 100) + 3;
  const Wide maximum = ~Wide{0};
  for (int depth : {0, 1, 3, 100}) {
    CheckDpfRange<8, Group>(prg, uint16_t{64}, uint16_t{127}, uint16_t{96}, depth);
    CheckDpfRange<8, Group>(prg, uint16_t{243}, uint16_t{255}, uint16_t{242}, depth);
    CheckDpfRange<65, Group>(prg, boundary - 6, boundary + 6, boundary, depth);
    CheckDpfRange<128, Group>(prg, high, high + 12, high + 6, depth);
    CheckDpfRange<128, Group>(prg, maximum - 12, maximum, maximum, depth);
    CheckDpfRange<128, Group>(prg, maximum, maximum, maximum, depth);
  }
}

struct DpfRangeCalls {
  std::atomic<int> inside{0};
  std::atomic<int> maximum{0};
  std::atomic<bool> other_thread{false};
  std::atomic<bool> active_team{false};
  const std::thread::id caller = std::this_thread::get_id();
};

template <typename Prg>
struct DpfRangeObservedPrg {
  Prg prg;
  DpfRangeCalls *calls;

  __host__ __device__ cuda::std::array<int4, 2> Gen(int4 seed) {
#ifndef __CUDA_ARCH__
    if (std::this_thread::get_id() != calls->caller) calls->other_thread = true;
#ifdef _OPENMP
    if (omp_in_parallel()) calls->active_team = true;
#endif
    const int current = calls->inside.fetch_add(1) + 1;
    int previous = calls->maximum.load();
    while (current > previous && !calls->maximum.compare_exchange_weak(previous, current)) {}
#endif
    auto result = prg.Gen(seed);
#ifndef __CUDA_ARCH__
    --calls->inside;
#endif
    return result;
  }
};

}  // namespace

TEST(DpfRangeTest, ChaChaBytesScalarAndPointOracles) {
  CheckDpfRangeSuite<BytesGroup>(fss::prg::ChaCha<2>(kNonce0, kNonce1));
}

TEST(DpfRangeTest, ChaChaUint127ScalarAndPointOracles) {
  CheckDpfRangeSuite<Uint127Group>(fss::prg::ChaCha<2>(kNonce0, kNonce1));
}

TEST(DpfRangeTest, AesSoftScalarAndPointOracles) {
  uint32_t table[256];
  uint8_t sbox[256];
  uint8_t keys[2][16];
  for (int key = 0; key < 2; ++key) {
    for (int byte = 0; byte < 16; ++byte) keys[key][byte] = static_cast<uint8_t>(key * 29 + byte * 7);
  }
  fss::prg::aes_detail::InitTe0(table);
  fss::prg::aes_detail::InitSbox(sbox);
  fss::prg::Aes128Soft<2> prg(keys, table, sbox);
  CheckDpfRangeSuite<BytesGroup>(prg);
  CheckDpfRangeSuite<Uint127Group>(prg);
}

TEST(DpfRangeTest, ZeroDepthConfinesUnsafeAesBackendToCaller) {
  using Prg = fss::prg::Aes128Mmo<2>;
  const unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8};
  const unsigned char key1[16] = {8, 7, 6, 5, 4, 3, 2, 1};
  const unsigned char *keys[2] = {key0, key1};
  auto contexts = Prg::CreateCtxs(keys);
  DpfRangeCalls calls;
  DpfRangeObservedPrg observed{Prg(contexts), &calls};
  CheckDpfRange<5, BytesGroup>(observed, uint8_t{3}, uint8_t{29}, uint8_t{11}, 0);
  const __uint128_t maximum = ~__uint128_t{0};
  CheckDpfRange<128, Uint127Group>(observed, maximum - 12, maximum, maximum, 0);
  EXPECT_EQ(calls.inside.load(), 0);
  EXPECT_EQ(calls.maximum.load(), 1);
  EXPECT_FALSE(calls.other_thread.load());
  EXPECT_FALSE(calls.active_team.load());
  Prg::FreeCtxs(contexts);
}

TEST(DpfRangeTest, PositiveDepthUsesConcurrentSafeBackend) {
  DpfRangeCalls calls;
  DpfRangeObservedPrg observed{fss::prg::ChaCha<2>(kNonce0, kNonce1), &calls};
#ifdef _OPENMP
  const int old_dynamic = omp_get_dynamic();
  const int old_threads = omp_get_max_threads();
  omp_set_dynamic(0);
  omp_set_num_threads(2);
#endif
  CheckDpfRange<8, BytesGroup>(observed, uint16_t{3}, uint16_t{244}, uint16_t{111}, 3);
  EXPECT_EQ(calls.inside.load(), 0);
#ifdef _OPENMP
  EXPECT_TRUE(calls.other_thread.load());
  EXPECT_TRUE(calls.active_team.load());
  omp_set_num_threads(old_threads);
  omp_set_dynamic(old_dynamic);
#endif
}


TEST(DpfGpuRangeTest, ClosedIntervalsAndFrontiers) {
  fss::Dpf<8, BytesGroup, fss::prg::ChaCha<2>, uint16_t> scheme{fss::prg::ChaCha<2>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t, 2, 4>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 0);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 255, 255);
}

TEST(DpfGpuRangeTest, WideDomainAndBoundedSubdivision) {
  fss::Dpf<128, BytesGroup, fss::prg::ChaCha<2>, __uint128_t> wide{fss::prg::ChaCha<2>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<128, __uint128_t>(wide, ~__uint128_t{0} - 128, ~__uint128_t{0});
  fss_test::CheckGpuRange<128, __uint128_t, 2, 4>(wide,
      (__uint128_t{1} << 100) - 17, (__uint128_t{1} << 100) + 19);
  fss::Dpf<12, BytesGroup, fss::prg::ChaCha<2>, uint16_t> split{fss::prg::ChaCha<2>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<12, uint16_t, 0, 1>(split, 0, 4095);
  fss_test::CheckGpuRange<12, uint16_t>(split, 0, 4095);
}

TEST(DpfGpuRangeTest, ArithmeticGroupNarrowInputAndCappedFrontier) {
  fss::Dpf<8, fss_test::RangeUint127Group, fss::prg::ChaCha<2>, uint8_t> scheme{fss::prg::ChaCha<2>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<8, uint8_t, 999, 1024>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint8_t, 2, 4>(scheme, 13, 201);
  fss::Dpf<1, fss_test::RangeUint127Group, fss::prg::ChaCha<2>, uint8_t> tiny{fss::prg::ChaCha<2>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<1, uint8_t, 999, 1024>(tiny, 0, 1);
  fss_test::CheckGpuRange<1, uint8_t>(tiny, 1, 1);
}


template <typename Group, bool has_half, typename Backend = fss::prg::Aes128Soft<2>>
void CheckDpfGenHalf() {
  constexpr int bits = 5;
  using Prg = fss_test::CountedPrg<Backend, 2, has_half>;
  using FullPrg = fss_test::CountedPrg<Backend, 2, false>;
  using Scheme = fss::Dpf<bits, Group, Prg, uint8_t>;
  using FullScheme = fss::Dpf<bits, Group, FullPrg, uint8_t>;
  fss_test::GenHalfAesFixture<2> aes;
  fss_test::PrgCallCounts calls, full_calls;
  auto backend = [&] {
    if constexpr (std::is_same_v<Backend, fss::prg::Aes128Soft<2>>) return aes.Backend();
    else return Backend(aes.keys);
  }();
  Scheme scheme{Prg(backend, calls)};
  FullScheme reference{FullPrg(backend, full_calls)};
  int4 seeds[2] = {{1, 3, 5, 6}, {7, 9, 11, 12}};
  int4 beta = {42, 6, 8, 10};
  typename Scheme::Key keys[2];
  typename FullScheme::Key full_keys[2];
  fss_test::GenPair(scheme, keys[0], keys[1], seeds, uint8_t{13}, beta);
  fss_test::GenPair(reference, full_keys[0], full_keys[1], seeds, uint8_t{13}, beta);
  EXPECT_EQ(calls.full, 2 * bits);
  EXPECT_EQ(calls.half, 0);
  for (uint8_t x = 0; x < 32; ++x) {
    int4 ys[2];
    for (int b = 0; b < 2; ++b) {
      calls = {};
      ys[b] = fss_test::EvalKey(scheme, b, keys[b], x);
      int4 expected = fss_test::EvalKey(reference, b, full_keys[b], x);
      EXPECT_EQ(memcmp(&ys[b], &expected, sizeof(int4)), 0) << "party=" << b << " x=" << +x;
      EXPECT_EQ(calls.full, has_half ? 0 : bits);
      EXPECT_EQ(calls.half, has_half ? bits : 0);
      EXPECT_EQ(calls.right, has_half ? __builtin_popcount(x) : 0);
    }
    int4 result = (Group::From(ys[0]) + Group::From(ys[1])).Into();
    int4 expected = x == 13 ? Group::From(beta).Into() : Group{}.Into();
    EXPECT_EQ(memcmp(&result, &expected, GroupCompareSize<Group>()), 0);
  }
}

TEST(DpfGenHalf, BytesSelectedAndFallback) {
  CheckDpfGenHalf<BytesGroup, true>();
  CheckDpfGenHalf<BytesGroup, false>();
}
TEST(DpfGenHalf, Uint127SelectedAndFallback) {
  CheckDpfGenHalf<Uint127Group, true>();
  CheckDpfGenHalf<Uint127Group, false>();
}
TEST(DpfGenHalf, GenericGroupSelectedAndFallback) {
  CheckDpfGenHalf<Uint64Group, true>();
  CheckDpfGenHalf<Uint64Group, false>();
}


TEST(DpfGenHalf, AesRawSelectedAndFallback) {
  CheckDpfGenHalf<BytesGroup, true, fss::prg::Aes128MmoRaw<2>>();
  CheckDpfGenHalf<BytesGroup, false, fss::prg::Aes128MmoRaw<2>>();
}

TEST(DpfGenHalf, GpuAesSoftSelectedRange) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "no CUDA device available";
  fss_test::GenHalfDeviceAesFixture<2> aes;
  ASSERT_EQ(aes.Allocate(), cudaSuccess);
  fss::Dpf<16, Uint64Group, fss::prg::Aes128Soft<2>, uint16_t> scheme{aes.Backend()};
  fss_test::CheckGpuRange<16, uint16_t, 8, 64>(scheme, 33, 47);
}

namespace {

constexpr int4 kComponentBeta = {0x11223344, 0x55667788,
    static_cast<int>(0x99aabbccu), 0x0deeff00};
constexpr int4 kComponentSentinel = {0x13579bdf, 0x2468ace0,
    static_cast<int>(0xbadc0de0u), 0x12345678};
constexpr int4 kComponentSeeds[2] = {
    {static_cast<int>(0x243f6a88u), static_cast<int>(0x85a308d3u),
        static_cast<int>(0x13198a2eu), static_cast<int>(0x03707344u)},
    {static_cast<int>(0xa4093822u), static_cast<int>(0x299f31d0u),
        static_cast<int>(0x082efa98u), static_cast<int>(0xec4e6c89u)}};

template <typename Group, int bits, typename In>
void CheckComponentEval(const std::vector<In> &queries, In alpha, int4 beta,
    bool use_default_stride, size_t stride) {
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::Dpf<bits, Group, Prg, In>;

  Scheme scheme{Prg(kNonce0, kNonce1)};
  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, kComponentSeeds, alpha, beta);

  std::vector<int4> cws_s(static_cast<size_t>(bits) + 1);
  std::vector<char> cws_tr(static_cast<size_t>(bits) + 1);
  for (int i = 0; i <= bits; ++i) {
    cws_s[i] = k0.cws[i].s;
    cws_tr[i] = k0.cws[i].tr;
  }

  const int4 terminal = cws_s[bits];
  ASSERT_EQ(terminal.w & 1, 0);
  In packed_tr = 0;
  for (int i = 0; i < bits; ++i) {
    if (cws_tr[i]) packed_tr |= static_cast<In>(In{1} << i);
  }
  // Bits at and above the domain depth are outside the packed correction word.
  constexpr int kInBits = static_cast<int>(sizeof(In) * 8);
  if constexpr (bits < kInBits) {
    for (int i = bits; i < kInBits; ++i) {
      packed_tr |= static_cast<In>(In{1} << i);
    }
  }

  const size_t level_stride = use_default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(bits) * level_stride + 1;
  constexpr size_t kExtra = 3;
  std::vector<int4> storage(used + kExtra, kComponentSentinel);
  std::vector<int4> expected_storage = storage;
  for (int i = 0; i < bits; ++i) {
    storage[static_cast<size_t>(i) * level_stride] = cws_s[i];
    expected_storage[static_cast<size_t>(i) * level_stride] = cws_s[i];
  }
  storage[static_cast<size_t>(bits) * level_stride] = terminal;
  expected_storage[static_cast<size_t>(bits) * level_stride] = terminal;
  const cuda::std::span<const int4> exact(storage.data(), used);
  const cuda::std::span<const int4> extra(storage.data(), used + kExtra);

  for (const In x : queries) {
    int4 aos[2];
    int4 got_exact[2];
    int4 got_extra[2];
    for (int party = 0; party < 2; ++party) {
      const bool is_party1 = party != 0;
      const int4 s = is_party1 ? k1.s0 : k0.s0;
      aos[party] = scheme.Eval(is_party1, s, k0.cws, x);
      auto eval = [&](cuda::std::span<const int4> cw_s) {
        if (use_default_stride) return scheme.Eval(is_party1, s, cw_s, packed_tr, x);
        return scheme.Eval(is_party1, s, cw_s, packed_tr, x, stride);
      };
      got_exact[party] = eval(exact);
      got_extra[party] = eval(extra);
      EXPECT_EQ(memcmp(&aos[party], &got_exact[party], sizeof(int4)), 0)
          << "party=" << party << " x=" << static_cast<uint64_t>(x) << " stride=" << level_stride;
      EXPECT_EQ(memcmp(&aos[party], &got_extra[party], sizeof(int4)), 0)
          << "party=" << party << " x=" << static_cast<uint64_t>(x)
          << " stride=" << level_stride << " extra";
    }
    const int4 sum = (Group::From(got_exact[0]) + Group::From(got_exact[1])).Into();
    const int4 truth = x == alpha ? Group::From(beta).Into() : Group{}.Into();
    EXPECT_EQ(memcmp(&sum, &truth, GroupCompareSize<Group>()), 0)
        << "sum x=" << static_cast<uint64_t>(x) << " stride=" << level_stride;
  }

  for (size_t i = 0; i < storage.size(); ++i) {
    EXPECT_EQ(memcmp(&storage[i], &expected_storage[i], sizeof(int4)), 0)
        << "component storage changed at " << i << " stride=" << level_stride;
  }
  for (int i = 0; i <= bits; ++i) {
    EXPECT_EQ(memcmp(&k0.cws[i].s, &cws_s[i], sizeof(int4)), 0)
        << "AoS s changed at " << i << " stride=" << level_stride;
    EXPECT_EQ(k0.cws[i].tr, cws_tr[i] != 0) << "AoS tr changed at " << i;
  }
}

template <typename Group, int bits, typename In>
void CheckComponentEvalStrides(const std::vector<In> &queries, In alpha, int4 beta) {
  CheckComponentEval<Group, bits, In>(queries, alpha, beta, true, 1);
  CheckComponentEval<Group, bits, In>(queries, alpha, beta, false, 1);
  CheckComponentEval<Group, bits, In>(queries, alpha, beta, false, 2);
  CheckComponentEval<Group, bits, In>(queries, alpha, beta, false, 7);
}

std::vector<__uint128_t> WideComponentQueries(__uint128_t alpha) {
  return {__uint128_t{0}, alpha, alpha ^ __uint128_t{1}, alpha ^ (__uint128_t{1} << 127),
      alpha ^ (__uint128_t{1} << 126), ~__uint128_t{0}};
}

}  // namespace

TEST(DpfComponentEval, BytesDepths) {
  CheckComponentEvalStrides<BytesGroup, 1, uint8_t>({0, 1}, uint8_t{1}, kComponentBeta);
  CheckComponentEvalStrides<BytesGroup, 3, uint8_t>({0, 1, 2, 3, 4, 5, 6, 7}, uint8_t{5},
      kComponentBeta);
  CheckComponentEvalStrides<BytesGroup, 8, uint16_t>(
      {0, 1, 37, 164, 165, 166, 254, 255}, uint16_t{165}, kComponentBeta);
}

TEST(DpfComponentEval, BytesDepth128) {
  const __uint128_t alpha = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  CheckComponentEvalStrides<BytesGroup, 128, __uint128_t>(
      WideComponentQueries(alpha), alpha, kComponentBeta);
}

TEST(DpfComponentEval, Uint127Depth128) {
  const __uint128_t alpha = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  CheckComponentEvalStrides<Uint127Group, 128, __uint128_t>(
      WideComponentQueries(alpha), alpha, kComponentBeta);
}

TEST(DpfComponentEval, LargeStrideDepthOne) {
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::Dpf<1, BytesGroup, Prg, uint8_t>;
  Scheme scheme{Prg(kNonce0, kNonce1)};
  const uint8_t alpha = 1;
  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, kComponentSeeds, alpha, kComponentBeta);

  // The terminal sits at cws[in_bits*stride], so a legal stride needs real
  // storage for it. The stride stays at most 2^20, the largest domain the
  // library evaluates, because a larger stride is not a case worth testing.
  // Strides past the guard are rejected by the debug test.
  constexpr size_t kStride = size_t{1} << 20;
  std::vector<int4> storage(kStride + 1, kComponentSentinel);
  storage[0] = k0.cws[0].s;
  storage[kStride] = k0.cws[1].s;
  uint8_t packed_tr = k0.cws[0].tr ? 1 : 0;
  for (int i = 1; i < 8; ++i) packed_tr |= static_cast<uint8_t>(1u << i);
  const cuda::std::span<const int4> exact(storage.data(), storage.size());

  for (int party = 0; party < 2; ++party) {
    const bool is_party1 = party != 0;
    const int4 s = is_party1 ? k1.s0 : k0.s0;
    for (int x = 0; x < 2; ++x) {
      const uint8_t point = static_cast<uint8_t>(x);
      const int4 aos = scheme.Eval(is_party1, s, k0.cws, point);
      const int4 got = scheme.Eval(is_party1, s, exact, packed_tr, point, kStride);
      EXPECT_EQ(memcmp(&aos, &got, sizeof(int4)), 0) << "party=" << party << " x=" << x;
    }
  }
}

TEST(DpfComponentEval, AsymmetricPackedRightControls) {
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::Dpf<3, BytesGroup, Prg, uint8_t>;
  Scheme scheme{Prg(kNonce0, kNonce1)};
  const uint8_t alpha = 5;
  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, kComponentSeeds, alpha, kComponentBeta);

  // 110 is not a palindrome, so reading root-order bits backwards is observable.
  const bool tr_pattern[3] = {true, true, false};
  const bool tl_pattern[3] = {true, false, true};
  for (int i = 0; i < 3; ++i) {
    k0.cws[i].tr = k1.cws[i].tr = tr_pattern[i];
    k0.cws[i].s = k1.cws[i].s = fss::util::SetLsb(k0.cws[i].s, tl_pattern[i]);
  }
  const int4 terminal = make_int4(0x0f1e2d3c, 0x4b5a6978, 0x07162534, 0x0a0b0c0e);
  ASSERT_EQ(terminal.w & 1, 0);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(memcmp(&terminal, &k0.cws[i].s, sizeof(int4)), 0);
  }
  k0.cws[3].s = terminal;
  k0.cws[3].tr = false;
  k1.cws[3].s = terminal;
  k1.cws[3].tr = false;

  uint8_t packed_tr = 0;
  for (int i = 0; i < 3; ++i) {
    if (k0.cws[i].tr) packed_tr |= static_cast<uint8_t>(1u << i);
  }
  EXPECT_EQ(static_cast<uint8_t>(packed_tr & 0x7), uint8_t{0x3});
  for (int i = 3; i < 8; ++i) packed_tr |= static_cast<uint8_t>(1u << i);

  std::vector<int4> cw_s(4, kComponentSentinel);
  for (int i = 0; i < 3; ++i) cw_s[i] = k0.cws[i].s;
  cw_s[3] = terminal;
  const cuda::std::span<const int4> cw_span(cw_s.data(), cw_s.size());

  for (int party = 0; party < 2; ++party) {
    const bool is_party1 = party != 0;
    const int4 s = is_party1 ? k1.s0 : k0.s0;
    for (int x = 0; x < 8; ++x) {
      const uint8_t point = static_cast<uint8_t>(x);
      const int4 aos = scheme.Eval(is_party1, s, k0.cws, point);
      const int4 got = scheme.Eval(is_party1, s, cw_span, packed_tr, point);
      EXPECT_EQ(memcmp(&aos, &got, sizeof(int4)), 0) << "party=" << party << " x=" << x;
    }
  }
}

namespace {

struct ObservedPrgCounts {
  int full = 0;
  int half = 0;
  int copies = 0;
  int live = 0;
  int destroyed = 0;

  void ResetCalls() {
    full = 0;
    half = 0;
    copies = 0;
    destroyed = 0;
  }
};

template <bool has_half>
class ObservedChaChaPrg {
public:
  __host__ __device__ explicit ObservedChaChaPrg(ObservedPrgCounts &counts)
      : chacha_(kNonce0, kNonce1), counts_(&counts) {
    ++counts_->live;
  }

  __host__ __device__ ObservedChaChaPrg(const ObservedChaChaPrg &other)
      : chacha_(other.chacha_), counts_(other.counts_) {
    ++counts_->copies;
    ++counts_->live;
  }

  __host__ __device__ ObservedChaChaPrg &operator=(const ObservedChaChaPrg &) = delete;

  __host__ __device__ ~ObservedChaChaPrg() {
    --counts_->live;
    ++counts_->destroyed;
  }

  __host__ __device__ cuda::std::array<int4, 2> Gen(int4 seed) {
    ++counts_->full;
    return chacha_.Gen(seed);
  }

  __host__ __device__ cuda::std::array<int4, 1> GenHalf(int4 seed, bool right)
    requires(has_half) {
    ++counts_->half;
    // Select from the inner ChaCha directly so the full-Gen counter stays untouched.
    const auto full = chacha_.Gen(seed);
    return cuda::std::array<int4, 1>{full[right ? 1 : 0]};
  }

private:
  fss::prg::ChaCha<2> chacha_;
  ObservedPrgCounts *counts_;
};

template <bool has_half, int bits>
void CheckComponentEvalBackendCalls() {
  using Backend = ObservedChaChaPrg<has_half>;
  using Scheme = fss::Dpf<bits, BytesGroup, Backend, uint8_t>;
  ObservedPrgCounts counts;

  {
    ObservedPrgCounts probe;
    {
      Backend first(probe);
      Backend second(first);
      EXPECT_EQ(probe.copies, 1);
      EXPECT_EQ(probe.live, 2);
    }
    EXPECT_EQ(probe.live, 0);
  }

  {
    Scheme scheme{Backend(counts)};
    EXPECT_GE(counts.copies, 1);
    EXPECT_EQ(counts.live, 1);

    constexpr uint8_t kX = 3;
    typename Scheme::Key k0, k1;
    fss_test::GenPair(scheme, k0, k1, kComponentSeeds, kX, kComponentBeta);
    EXPECT_EQ(counts.live, 1);
    EXPECT_EQ(counts.full, 2 * bits);
    EXPECT_EQ(counts.half, 0);

    std::vector<int4> cw_s(static_cast<size_t>(bits) + 1);
    uint8_t packed_tr = 0;
    for (int i = 0; i < bits; ++i) {
      cw_s[i] = k0.cws[i].s;
      if (k0.cws[i].tr) packed_tr |= static_cast<uint8_t>(1u << i);
    }
    for (int i = bits; i < 8; ++i) packed_tr |= static_cast<uint8_t>(1u << i);
    cw_s[bits] = k0.cws[bits].s;
    const cuda::std::span<const int4> cw_span(cw_s.data(), cw_s.size());

    for (int method = 0; method < 2; ++method) {
      for (int party = 0; party < 2; ++party) {
        const bool is_party1 = party != 0;
        const int4 s = is_party1 ? k1.s0 : k0.s0;
        counts.ResetCalls();
        const int live_before = counts.live;
        int4 y;
        if (method == 0) {
          y = scheme.Eval(is_party1, s, k0.cws, kX);
        } else {
          y = scheme.Eval(is_party1, s, cw_span, packed_tr, kX);
        }
        EXPECT_EQ(counts.full, has_half ? 0 : bits);
        EXPECT_EQ(counts.half, has_half ? bits : 0);
        EXPECT_EQ(counts.copies, 0);
        EXPECT_EQ(counts.live, live_before);
        EXPECT_EQ(counts.destroyed, 0);
        (void)y;
      }
    }
  }
  EXPECT_EQ(counts.live, 0);
}

}  // namespace

TEST(DpfComponentEval, BackendCallsSelectedAndFallback) {
  CheckComponentEvalBackendCalls<true, 5>();
  CheckComponentEvalBackendCalls<false, 5>();
}

// The split Eval is a low-level entry shared with the device: layout
// violations are caller preconditions, so Debug builds abort and these invalid
// inputs must not run in Release.
#ifndef NDEBUG
TEST(DpfComponentEval, AbortsOnInvalidComponentInputsInDebug) {
  using Prg = fss::prg::ChaCha<2>;
  using Three = fss::Dpf<3, BytesGroup, Prg, uint8_t>;
  Three three{Prg(kNonce0, kNonce1)};
  const int4 seed = {1, 2, 3, 4};
  int4 tiny[8] = {};
  const cuda::std::span<const int4> tiny_span(tiny);

  EXPECT_DEATH((void)three.Eval(false, seed, tiny_span, uint8_t{0}, uint8_t{0}, size_t{0}),
      "Assertion.*failed");
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_DEATH((void)three.Eval(false, seed, cuda::std::span<const int4>(tiny, 3), uint8_t{0},
                       uint8_t{0}, size_t{1}),
      "Assertion.*failed");
  EXPECT_DEATH((void)three.Eval(false, seed, cuda::std::span<const int4>(tiny, 6), uint8_t{0},
                       uint8_t{0}, size_t{2}),
      "Assertion.*failed");
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / 3 + 1;
  EXPECT_DEATH((void)three.Eval(false, seed, tiny_span, uint8_t{0}, uint8_t{0},
                       overflowing_stride),
      "Assertion.*failed");
}
#endif  // NDEBUG

namespace {

template <int bits, typename In>
using ComponentGpuScheme = fss::Dpf<bits, BytesGroup, fss::prg::ChaCha<2>, In>;

template <int bits, typename In>
__global__ void ComponentEvalGpuKernel(bool b, int4 seed,
    const typename ComponentGpuScheme<bits, In>::Cw *cws, const int4 *cw_s, size_t cw_s_count,
    In packed_tr, const In *queries, int4 *aos, int4 *component, int query_count,
    size_t level_stride) {
  const int index = static_cast<int>(threadIdx.x);
  if (index >= query_count) return;
  using Scheme = ComponentGpuScheme<bits, In>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const In x = queries[index];
  aos[index] = scheme.Eval(b, seed, cws, x);
  component[index] = scheme.Eval(b, seed,
      cuda::std::span<const int4>(cw_s, cw_s_count), packed_tr, x, level_stride);
}

void ExpectComponentInt4(const int4 &actual, const int4 &expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

std::vector<__uint128_t> ComponentGpuDepth128Queries(__uint128_t alpha) {
  return {
      alpha,
      alpha ^ (__uint128_t{1} << 127),
      alpha ^ (__uint128_t{1} << 64),
      alpha ^ (__uint128_t{1} << 63),
      alpha ^ __uint128_t{1},
      ~__uint128_t{0},
  };
}

template <int bits, typename In>
void CheckComponentEvalGpu(const std::vector<In> &queries, In alpha, int4 beta, size_t level_stride) {
  using Scheme = ComponentGpuScheme<bits, In>;
  using Cw = typename Scheme::Cw;
  ASSERT_FALSE(queries.empty());
  ASSERT_LE(queries.size(), size_t{8});
  ASSERT_GT(level_stride, size_t{0});
  if constexpr (bits > 1) {
    ASSERT_LE(level_stride, size_t{7});
  }

  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, kComponentSeeds, alpha, beta);

  ASSERT_EQ(k0.cws[bits].s.w & 1, 0);

  const size_t query_count = queries.size();
  const size_t used = static_cast<size_t>(bits) * level_stride + 1;
  const size_t capacity = used + 3;

  Cw *d_cws = nullptr;
  int4 *d_cw_s = nullptr;
  In *d_queries = nullptr;
  int4 *d_aos = nullptr;
  int4 *d_component = nullptr;
  auto cleanup = [&] {
    if (d_component != nullptr) EXPECT_EQ(cudaFree(d_component), cudaSuccess);
    if (d_aos != nullptr) EXPECT_EQ(cudaFree(d_aos), cudaSuccess);
    if (d_queries != nullptr) EXPECT_EQ(cudaFree(d_queries), cudaSuccess);
    if (d_cw_s != nullptr) EXPECT_EQ(cudaFree(d_cw_s), cudaSuccess);
    if (d_cws != nullptr) EXPECT_EQ(cudaFree(d_cws), cudaSuccess);
  };

  cudaError_t alloc = cudaMallocManaged(&d_cws, (static_cast<size_t>(bits) + 1) * sizeof(Cw));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_cw_s, capacity * sizeof(int4));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_queries, query_count * sizeof(In));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_aos, 2 * query_count * sizeof(int4));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_component, 2 * query_count * sizeof(int4));
  if (alloc != cudaSuccess) {
    cleanup();
    FAIL() << "cudaMallocManaged failed: " << cudaGetErrorString(alloc);
    return;
  }

  for (int i = 0; i <= bits; ++i) {
    d_cws[i].s = k0.cws[i].s;
    d_cws[i].tr = k0.cws[i].tr;
  }
  for (size_t i = 0; i < capacity; ++i) {
    d_cw_s[i].x = kComponentSentinel.x;
    d_cw_s[i].y = kComponentSentinel.y;
    d_cw_s[i].z = kComponentSentinel.z;
    d_cw_s[i].w = kComponentSentinel.w;
  }
  for (int i = 0; i <= bits; ++i) {
    const size_t slot = static_cast<size_t>(i) * level_stride;
    d_cw_s[slot].x = k0.cws[i].s.x;
    d_cw_s[slot].y = k0.cws[i].s.y;
    d_cw_s[slot].z = k0.cws[i].s.z;
    d_cw_s[slot].w = k0.cws[i].s.w;
  }
  In packed_tr = 0;
  for (int i = 0; i < bits; ++i) {
    if (k0.cws[i].tr) packed_tr = static_cast<In>(packed_tr | static_cast<In>(__uint128_t{1} << i));
  }
  for (size_t i = 0; i < query_count; ++i) d_queries[i] = queries[i];
  for (size_t i = 0; i < 2 * query_count; ++i) {
    d_aos[i] = kComponentSentinel;
    d_component[i] = kComponentSentinel;
  }

  for (int party = 0; party < 2; ++party) {
    ComponentEvalGpuKernel<bits, In><<<1, static_cast<unsigned>(query_count)>>>(party != 0,
        party == 0 ? k0.s0 : k1.s0, d_cws, d_cw_s, used, packed_tr, d_queries,
        d_aos + party * query_count, d_component + party * query_count,
        static_cast<int>(query_count), level_stride);
    const cudaError_t launch = cudaGetLastError();
    EXPECT_EQ(launch, cudaSuccess) << cudaGetErrorString(launch);
    const cudaError_t sync = cudaDeviceSynchronize();
    EXPECT_EQ(sync, cudaSuccess) << cudaGetErrorString(sync);
  }

  for (int party = 0; party < 2; ++party) {
    const typename Scheme::Key &key = party == 0 ? k0 : k1;
    for (size_t i = 0; i < query_count; ++i) {
      SCOPED_TRACE(::testing::Message() << "depth " << bits << " stride " << level_stride
                                        << " party " << party << " query " << i);
      const int4 oracle = scheme.Eval(party != 0, key.s0, key.cws, queries[i]);
      ExpectComponentInt4(d_aos[party * query_count + i], oracle);
      ExpectComponentInt4(d_component[party * query_count + i], oracle);
    }
  }

  for (size_t i = 0; i < capacity; ++i) {
    int4 expected = kComponentSentinel;
    if (i < used && i % level_stride == 0) expected = k0.cws[i / level_stride].s;
    SCOPED_TRACE(::testing::Message() << "depth " << bits << " stride " << level_stride
                                      << " component slot " << i);
    ExpectComponentInt4(d_cw_s[i], expected);
  }

  cleanup();
}

}  // namespace

TEST(DpfComponentEvalGpu, DeviceMatchesHostAos) {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(status, cudaSuccess);

  const __uint128_t alpha = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  const std::vector<__uint128_t> wide = ComponentGpuDepth128Queries(alpha);
  for (const size_t stride : {size_t{1}, size_t{2}, size_t{7}}) {
    CheckComponentEvalGpu<1, uint8_t>({0, 1}, uint8_t{1}, kComponentBeta, stride);
    CheckComponentEvalGpu<3, uint8_t>({0, 5, 7}, uint8_t{5}, kComponentBeta, stride);
    CheckComponentEvalGpu<8, uint16_t>({0, 1, 165, 255}, uint16_t{165}, kComponentBeta, stride);
    CheckComponentEvalGpu<128, __uint128_t>(wide, alpha, kComponentBeta, stride);
  }
  CheckComponentEvalGpu<1, uint8_t>({0, 1}, uint8_t{1}, kComponentBeta, size_t{1} << 20);
}

namespace {

constexpr int kInterleavedKeyCount = 2;
constexpr int4 kInterleavedBeta = {0x0badf00d, 0x12345678,
    static_cast<int>(0x76543210u), 0x0abcdef0};
constexpr int4 kInterleavedSeeds[kInterleavedKeyCount][2] = {
    {kComponentSeeds[0], kComponentSeeds[1]},
    {{static_cast<int>(0xdeadbeefu), static_cast<int>(0x0badc0deu),
         static_cast<int>(0xfeedfaceu), static_cast<int>(0x01234566u)},
        {static_cast<int>(0xcafebabeu), static_cast<int>(0x8badf00du),
            static_cast<int>(0x0ddba11u), static_cast<int>(0x0f00d123u)}}};

bool Int4Equal(const int4 &a, const int4 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

// One thread per independent key; all keys are active in the same block.
template <int bits, typename In>
__global__ void InterleavedComponentEvalGpuKernel(bool b, const int4 *seeds,
    const typename ComponentGpuScheme<bits, In>::Cw *cws, const int4 *component,
    const In *packed_tr, const In *queries, int4 *aos_out,
    int4 *component_out, int query_count, size_t component_count,
    size_t exact_span_length) {
  const int key = static_cast<int>(threadIdx.x);
  if (key >= kInterleavedKeyCount) return;
  using Scheme = ComponentGpuScheme<bits, In>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const size_t span_length = exact_span_length == 0
      ? component_count - static_cast<size_t>(key)
      : exact_span_length;
  const typename Scheme::Cw *key_cws = cws + static_cast<size_t>(key) * (bits + 1);
  const In *key_queries = queries + static_cast<size_t>(key) * query_count;
  int4 *key_aos = aos_out + static_cast<size_t>(key) * query_count;
  int4 *key_component = component_out + static_cast<size_t>(key) * query_count;
  for (int i = 0; i < query_count; ++i) {
    const In x = key_queries[i];
    key_aos[i] = scheme.Eval(b, seeds[key], key_cws, x);
    key_component[i] = scheme.Eval(b, seeds[key],
        cuda::std::span<const int4>(component + key, span_length), packed_tr[key],
        x, kInterleavedKeyCount);
  }
}

template <int bits, typename In>
void CheckInterleavedComponentEvalGpu(const std::vector<In> &queries0,
    const std::vector<In> &queries1, In alpha0, int4 beta0, In alpha1, int4 beta1) {
  using Scheme = ComponentGpuScheme<bits, In>;
  using Cw = typename Scheme::Cw;
  static_assert(bits > 0);

  ASSERT_EQ(queries0.size(), queries1.size());
  ASSERT_GT(queries0.size(), size_t{0});
  EXPECT_TRUE(queries0 != queries1) << "the two keys need different query arrays";

  const std::vector<In> *query_sets[kInterleavedKeyCount] = {&queries0, &queries1};
  const In alphas[kInterleavedKeyCount] = {alpha0, alpha1};
  for (int key = 0; key < kInterleavedKeyCount; ++key) {
    bool has_alpha = false;
    bool has_non_alpha = false;
    for (const In x : *query_sets[key]) {
      has_alpha = has_alpha || x == alphas[key];
      has_non_alpha = has_non_alpha || x != alphas[key];
    }
    EXPECT_TRUE(has_alpha) << "key " << key << " has no alpha query";
    EXPECT_TRUE(has_non_alpha) << "key " << key << " has no non-alpha query";
  }

  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  typename Scheme::Key keys[kInterleavedKeyCount][2];
  const int4 betas[kInterleavedKeyCount] = {beta0, beta1};
  for (int key = 0; key < kInterleavedKeyCount; ++key) {
    fss_test::GenPair(scheme, keys[key][0], keys[key][1], kInterleavedSeeds[key],
        alphas[key], betas[key]);
  }

  for (int party = 0; party < 2; ++party) {
    EXPECT_FALSE(Int4Equal(keys[0][party].s0, keys[1][party].s0))
        << "seeds must differ between keys, party " << party;
  }
  for (int level = 0; level < bits; ++level) {
    EXPECT_FALSE(Int4Equal(keys[0][0].cws[level].s, keys[1][0].cws[level].s))
        << "correction words must differ between keys, level " << level;
  }

  int4 terminals[kInterleavedKeyCount];
  In packed_tr[kInterleavedKeyCount] = {In{0}, In{0}};
  constexpr int kInWidth = static_cast<int>(sizeof(In) * 8);
  for (int key = 0; key < kInterleavedKeyCount; ++key) {
    terminals[key] = keys[key][0].cws[bits].s;
    ASSERT_EQ(terminals[key].w & 1, 0);
    for (int level = 0; level < bits; ++level) {
      if (keys[key][0].cws[level].tr) {
        packed_tr[key] = static_cast<In>(packed_tr[key] | static_cast<In>(In{1} << level));
      }
    }
    // Bits at and above the depth are ignored. Tag them per key so the packed
    // correction words differ deterministically and the ignore rule is exercised.
    for (int level = bits; level < kInWidth; ++level) {
      if (key != 0) {
        packed_tr[key] = static_cast<In>(packed_tr[key] | static_cast<In>(In{1} << level));
      }
    }
  }
  EXPECT_FALSE(Int4Equal(terminals[0], terminals[1]))
      << "terminals must differ between keys";
  EXPECT_TRUE(packed_tr[0] != packed_tr[1])
      << "packed right controls must differ between keys";

  const size_t key_count = kInterleavedKeyCount;
  const size_t query_count = queries0.size();
  const size_t component_count = static_cast<size_t>(bits + 1) * key_count;
  const size_t output_count = 2 * key_count * query_count;
  std::vector<int4> expected_component(component_count);
  for (int level = 0; level <= bits; ++level) {
    for (int key = 0; key < kInterleavedKeyCount; ++key) {
      expected_component[static_cast<size_t>(level) * key_count + key] =
          keys[key][0].cws[level].s;
    }
  }

  int4 *d_seeds = nullptr;
  Cw *d_cws = nullptr;
  int4 *d_component = nullptr;
  In *d_packed_tr = nullptr;
  In *d_queries = nullptr;
  int4 *d_aos = nullptr;
  int4 *d_component_out = nullptr;
  auto cleanup = [&] {
    if (d_component_out != nullptr) EXPECT_EQ(cudaFree(d_component_out), cudaSuccess);
    if (d_aos != nullptr) EXPECT_EQ(cudaFree(d_aos), cudaSuccess);
    if (d_queries != nullptr) EXPECT_EQ(cudaFree(d_queries), cudaSuccess);
    if (d_packed_tr != nullptr) EXPECT_EQ(cudaFree(d_packed_tr), cudaSuccess);
    if (d_component != nullptr) EXPECT_EQ(cudaFree(d_component), cudaSuccess);
    if (d_cws != nullptr) EXPECT_EQ(cudaFree(d_cws), cudaSuccess);
    if (d_seeds != nullptr) EXPECT_EQ(cudaFree(d_seeds), cudaSuccess);
  };

  cudaError_t alloc = cudaMallocManaged(&d_seeds, 2 * key_count * sizeof(int4));
  if (alloc == cudaSuccess) {
    alloc = cudaMallocManaged(&d_cws,
        key_count * (static_cast<size_t>(bits) + 1) * sizeof(Cw));
  }
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_component, component_count * sizeof(int4));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_packed_tr, key_count * sizeof(In));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_queries, key_count * query_count * sizeof(In));
  if (alloc == cudaSuccess) alloc = cudaMallocManaged(&d_aos, output_count * sizeof(int4));
  if (alloc == cudaSuccess) {
    alloc = cudaMallocManaged(&d_component_out, output_count * sizeof(int4));
  }
  if (alloc != cudaSuccess) {
    cleanup();
    FAIL() << "cudaMallocManaged failed: " << cudaGetErrorString(alloc);
    return;
  }

  for (int party = 0; party < 2; ++party) {
    for (int key = 0; key < kInterleavedKeyCount; ++key) {
      d_seeds[static_cast<size_t>(party) * key_count + key] = keys[key][party].s0;
    }
  }
  for (int key = 0; key < kInterleavedKeyCount; ++key) {
    const size_t cw_base = static_cast<size_t>(key) * (bits + 1);
    for (int level = 0; level <= bits; ++level) {
      d_cws[cw_base + level].s = keys[key][0].cws[level].s;
      d_cws[cw_base + level].tr = keys[key][0].cws[level].tr;
    }
    d_packed_tr[key] = packed_tr[key];
    for (size_t i = 0; i < query_count; ++i) {
      d_queries[static_cast<size_t>(key) * query_count + i] = (*query_sets[key])[i];
    }
  }
  for (int level = 0; level <= bits; ++level) {
    for (int key = 0; key < kInterleavedKeyCount; ++key) {
      d_component[static_cast<size_t>(level) * key_count + key] = keys[key][0].cws[level].s;
    }
  }
  for (size_t i = 0; i < output_count; ++i) {
    d_aos[i] = kComponentSentinel;
    d_component_out[i] = kComponentSentinel;
  }

  const size_t exact_span_length = static_cast<size_t>(bits) * key_count + 1;
  for (const size_t span_length : {size_t{0}, exact_span_length}) {
    for (int party = 0; party < 2; ++party) {
      const size_t output_offset = static_cast<size_t>(party) * key_count * query_count;
      InterleavedComponentEvalGpuKernel<bits, In>
          <<<1, static_cast<unsigned>(kInterleavedKeyCount)>>>(party != 0,
              d_seeds + static_cast<size_t>(party) * key_count, d_cws, d_component,
              d_packed_tr, d_queries, d_aos + output_offset,
              d_component_out + output_offset, static_cast<int>(query_count),
              component_count, span_length);
      const cudaError_t launch = cudaGetLastError();
      EXPECT_EQ(launch, cudaSuccess) << cudaGetErrorString(launch);
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    EXPECT_EQ(sync, cudaSuccess) << cudaGetErrorString(sync);

    for (int party = 0; party < 2; ++party) {
      for (int key = 0; key < kInterleavedKeyCount; ++key) {
        const typename Scheme::Key &oracle_key = keys[key][party];
        const size_t key_offset =
            (static_cast<size_t>(party) * key_count + key) * query_count;
        for (size_t i = 0; i < query_count; ++i) {
          SCOPED_TRACE(::testing::Message() << "depth " << bits << " key " << key
                                            << " party " << party << " query " << i);
          const int4 oracle = scheme.Eval(party != 0, oracle_key.s0, oracle_key.cws,
              (*query_sets[key])[i]);
          ExpectComponentInt4(d_aos[key_offset + i], oracle);
          ExpectComponentInt4(d_component_out[key_offset + i], oracle);
        }
      }
    }
  }

  for (size_t i = 0; i < component_count; ++i) {
    SCOPED_TRACE(::testing::Message() << "depth " << bits << " component slot " << i);
    ExpectComponentInt4(d_component[i], expected_component[i]);
  }

  cleanup();
}

}  // namespace

TEST(DpfComponentEvalGpu, InterleavedKeys) {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(status, cudaSuccess);

  CheckInterleavedComponentEvalGpu<3, uint8_t>(
      {2, 3, 7}, {5, 1, 6}, uint8_t{2}, kComponentBeta, uint8_t{5}, kInterleavedBeta);
  CheckInterleavedComponentEvalGpu<8, uint16_t>(
      {165, 166, 0, 254}, {90, 89, 255, 7}, uint16_t{165}, kComponentBeta,
      uint16_t{90}, kInterleavedBeta);

  const __uint128_t alpha0 = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  const __uint128_t alpha1 = (__uint128_t{1} << 126) | (__uint128_t{1} << 65) | 0x99;
  CheckInterleavedComponentEvalGpu<128, __uint128_t>(
      ComponentGpuDepth128Queries(alpha0), ComponentGpuDepth128Queries(alpha1), alpha0,
      kComponentBeta, alpha1, kInterleavedBeta);
}

namespace {

constexpr size_t kGenMaxStride = 7;
constexpr int kGenQueryCap = 6;
constexpr size_t GenStridedSlots(int bits) { return static_cast<size_t>(bits) * kGenMaxStride + 5; }

template <int bits, typename Group, typename In>
struct GenRun {
  using Scheme = fss::Dpf<bits, Group, fss::prg::ChaCha<2>, In>;
  using Cw = typename Scheme::Cw;
  int4 seeds[2];
  In alpha;
  int4 beta;
  size_t stride;
  int query_count;
  In fixed_tr, strided_tr;
  In queries[kGenQueryCap];
  Cw legacy[bits + 1];
  int4 fixed[bits + 3];
  int4 legacy_out[2][kGenQueryCap], fixed_out[2][kGenQueryCap], strided_out[2][kGenQueryCap];
};

template <int bits, typename Group, typename In>
void PrepareGenRun(GenRun<bits, Group, In> &run, const int4 (&seeds)[2], In alpha,
    int4 beta, size_t stride, const std::vector<In> &queries) {
  EXPECT_LE(queries.size(), static_cast<size_t>(kGenQueryCap));
  run = GenRun<bits, Group, In>{};
  run.seeds[0] = seeds[0];
  run.seeds[1] = seeds[1];
  run.alpha = alpha;
  run.beta = beta;
  run.stride = stride;
  run.query_count = static_cast<int>(
      queries.size() > static_cast<size_t>(kGenQueryCap) ? kGenQueryCap : queries.size());
  for (int q = 0; q < run.query_count; ++q) run.queries[q] = queries[q];
  for (auto &cw : run.legacy) cw = {kComponentSentinel, false};
  for (auto &slot : run.fixed) slot = kComponentSentinel;
  const In ones = static_cast<In>(~static_cast<In>(0));
  run.fixed_tr = run.strided_tr = ones;
  for (int p = 0; p < 2; ++p) for (int q = 0; q < kGenQueryCap; ++q) {
    run.legacy_out[p][q] = run.fixed_out[p][q] = run.strided_out[p][q] = kComponentSentinel;
  }
}

template <int bits, typename Group, typename In>
__host__ __device__ void RunComponentGen(GenRun<bits, Group, In> &r, int4 *strided, size_t cap) {
  using Scheme = fss::Dpf<bits, Group, fss::prg::ChaCha<2>, In>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const cuda::std::span<const int4, 2> seeds(r.seeds, 2);
  scheme.Gen(r.legacy, r.seeds, r.alpha, r.beta);
  scheme.Gen(cuda::std::span<int4>(r.fixed + 1, bits + 1), r.fixed_tr, seeds,
      r.alpha, r.beta);
  scheme.Gen(cuda::std::span<int4>(strided, cap), r.strided_tr, seeds, r.alpha, r.beta,
      r.stride);
  const cuda::std::span<const int4> tree(r.fixed + 1, bits + 1);
  const cuda::std::span<const int4> stree(strided, cap);
  for (int p = 0; p < 2; ++p) for (int q = 0; q < r.query_count; ++q) {
    const bool b = p != 0;
    const In x = r.queries[q];
    r.legacy_out[p][q] = scheme.Eval(b, r.seeds[p], r.legacy, x);
    r.fixed_out[p][q] = scheme.Eval(b, r.seeds[p], tree, r.fixed_tr, x);
    r.strided_out[p][q] = scheme.Eval(b, r.seeds[p], stree, r.strided_tr, x, r.stride);
  }
}

template <int key_count, int bits, typename Group, typename In>
__global__ void GenGpuKernel(GenRun<bits, Group, In> *runs, int4 *shared, size_t total) {
  const int key = static_cast<int>(threadIdx.x);
  if (key < key_count) RunComponentGen<bits, Group, In>(runs[key], shared + 1 + key,
      total - 1 - key);
}

template <int bits, typename Group, typename In>
void CheckGenCws(const GenRun<bits, Group, In> &g, const int4 *s,
    const GenRun<bits, Group, In> &o) {
  for (int i = 0; i <= bits; ++i) {
    ExpectComponentInt4(g.legacy[i].s, o.legacy[i].s);
    EXPECT_EQ(g.legacy[i].tr, o.legacy[i].tr);
  }
  for (int i = 0; i < bits; ++i) {
    ExpectComponentInt4(g.fixed[1 + i], o.legacy[i].s);
    ExpectComponentInt4(s[static_cast<size_t>(i) * g.stride], o.legacy[i].s);
    EXPECT_EQ(((g.fixed_tr >> i) & 1) != 0, o.legacy[i].tr);
    EXPECT_EQ(((g.strided_tr >> i) & 1) != 0, o.legacy[i].tr);
  }
  const int4 ft = g.fixed[1 + bits], st = s[static_cast<size_t>(bits) * g.stride];
  ExpectComponentInt4(ft, o.legacy[bits].s);
  ExpectComponentInt4(st, o.legacy[bits].s);
  EXPECT_EQ(ft.w & 1, 0);
  EXPECT_EQ(st.w & 1, 0);
  EXPECT_EQ(g.legacy[bits].s.w & 1, 0);
  EXPECT_FALSE(g.legacy[bits].tr);
  EXPECT_TRUE(Int4Equal(g.fixed[0], kComponentSentinel));
  EXPECT_TRUE(Int4Equal(g.fixed[bits + 2], kComponentSentinel));
  if constexpr (bits < static_cast<int>(sizeof(In) * 8)) {
    EXPECT_FALSE(static_cast<In>(g.fixed_tr >> bits) != In{0});
    EXPECT_FALSE(static_cast<In>(g.strided_tr >> bits) != In{0});
  }
  for (int p = 0; p < 2; ++p) ExpectComponentInt4(g.seeds[p], o.seeds[p]);
  ExpectComponentInt4(g.beta, o.beta);
  EXPECT_EQ(g.stride, o.stride);
}

template <int bits, typename Group, typename In>
void CheckGenEval(const GenRun<bits, Group, In> &g, const GenRun<bits, Group, In> &o) {
  for (int p = 0; p < 2; ++p) for (int q = 0; q < g.query_count; ++q) {
    ExpectComponentInt4(g.legacy_out[p][q], o.legacy_out[p][q]);
    ExpectComponentInt4(g.fixed_out[p][q], o.legacy_out[p][q]);
    ExpectComponentInt4(g.strided_out[p][q], o.legacy_out[p][q]);
  }
  const int4 beta = Group::From(fss::util::SetLsb(g.beta, false)).Into();
  const int4 zero = Group{}.Into();
  auto sums = [&](const int4 (&out)[2][kGenQueryCap]) {
    for (int q = 0; q < g.query_count; ++q) {
      const int4 sum = (Group::From(out[0][q]) + Group::From(out[1][q])).Into();
      const int4 expected = g.queries[q] == g.alpha ? beta : zero;
      EXPECT_EQ(memcmp(&sum, &expected, GroupCompareSize<Group>()), 0);
    }
  };
  sums(g.legacy_out);
  sums(g.fixed_out);
  sums(g.strided_out);
}

template <int key_count, int bits, typename Group, typename In>
void CheckSharedLayout(const int4 *shared, size_t total, size_t stride,
    const GenRun<bits, Group, In> (&runs)[key_count]) {
  EXPECT_TRUE(Int4Equal(shared[0], kComponentSentinel));
  for (int i = 0; i <= bits; ++i) {
    const size_t plane = 1 + static_cast<size_t>(i) * stride;
    for (int key = 0; key < key_count; ++key) {
      ExpectComponentInt4(shared[plane + key], runs[key].legacy[i].s);
    }
    for (size_t gap = key_count; gap < stride; ++gap) {
      EXPECT_TRUE(Int4Equal(shared[plane + gap], kComponentSentinel));
    }
  }
  for (size_t j = 1 + static_cast<size_t>(bits + 1) * stride; j < total; ++j) {
    EXPECT_TRUE(Int4Equal(shared[j], kComponentSentinel));
  }
}

template <int key_count, typename Group, int bits, typename In>
void CheckGenSharedHost(const int4 (&seeds)[key_count][2], const In (&alphas)[key_count],
    const int4 (&betas)[key_count], const std::vector<In> (&queries)[key_count], size_t stride) {
  using Run = GenRun<bits, Group, In>;
  const size_t total = 1 + static_cast<size_t>(bits + 1) * stride + 2;
  Run runs[key_count]{};
  std::vector<int4> shared(total, kComponentSentinel);
  for (int key = 0; key < key_count; ++key) {
    PrepareGenRun(runs[key], seeds[key], alphas[key], betas[key], stride, queries[key]);
  }
  for (int key = 0; key < key_count; ++key) {
    RunComponentGen(runs[key], shared.data() + 1 + key, total - 1 - key);
  }
  for (int key = 0; key < key_count; ++key) {
    for (int p = 0; p < 2; ++p) EXPECT_TRUE(Int4Equal(runs[key].seeds[p], seeds[key][p]));
    CheckGenCws(runs[key], shared.data() + 1 + key, runs[key]);
    CheckGenEval(runs[key], runs[key]);
  }
  CheckSharedLayout<key_count>(shared.data(), total, stride, runs);
}

template <int key_count, typename Group, int bits, typename In>
void CheckGenSharedGpu(const int4 (&seeds)[key_count][2], const In (&alphas)[key_count],
    const int4 (&betas)[key_count], const std::vector<In> (&queries)[key_count], size_t stride) {
  using Run = GenRun<bits, Group, In>;
  const size_t total = 1 + static_cast<size_t>(bits + 1) * stride + 2;
  Run *d_runs = nullptr;
  int4 *d_shared = nullptr;
  auto free_all = [&] {
    if (d_shared != nullptr) EXPECT_EQ(cudaFree(d_shared), cudaSuccess);
    if (d_runs != nullptr) EXPECT_EQ(cudaFree(d_runs), cudaSuccess);
  };
  cudaError_t status = cudaMallocManaged(&d_runs, key_count * sizeof(Run));
  if (status == cudaSuccess) status = cudaMallocManaged(&d_shared, total * sizeof(int4));
  if (status != cudaSuccess) {
    free_all();
    FAIL() << cudaGetErrorString(status);
    return;
  }
  for (int key = 0; key < key_count; ++key) {
    PrepareGenRun(d_runs[key], seeds[key], alphas[key], betas[key], stride, queries[key]);
  }
  for (size_t i = 0; i < total; ++i) d_shared[i] = kComponentSentinel;
  Run oracle[key_count]{};
  std::vector<int4> oracle_shared(total, kComponentSentinel);
  for (int key = 0; key < key_count; ++key) {
    PrepareGenRun(oracle[key], seeds[key], alphas[key], betas[key], stride, queries[key]);
    RunComponentGen(oracle[key], oracle_shared.data() + 1 + key, total - 1 - key);
  }
  GenGpuKernel<key_count, bits, Group, In>
      <<<1, static_cast<unsigned>(key_count)>>>(d_runs, d_shared, total);
  const cudaError_t launch = cudaGetLastError();
  const cudaError_t sync = cudaDeviceSynchronize();
  EXPECT_EQ(launch, cudaSuccess) << cudaGetErrorString(launch);
  EXPECT_EQ(sync, cudaSuccess) << cudaGetErrorString(sync);
  for (int key = 0; key < key_count; ++key) {
    CheckGenCws(d_runs[key], d_shared + 1 + key, oracle[key]);
    CheckGenEval(d_runs[key], oracle[key]);
    for (int p = 0; p < 2; ++p) EXPECT_TRUE(Int4Equal(d_runs[key].seeds[p], seeds[key][p]));
  }
  CheckSharedLayout<key_count>(d_shared, total, stride, oracle);
  free_all();
}

template <bool has_half, int bits>
void CheckGenBackendCalls() {
  using Backend = ObservedChaChaPrg<has_half>;
  using Scheme = fss::Dpf<bits, BytesGroup, Backend, uint8_t>;
  ObservedPrgCounts counts;
  {
    Scheme scheme{Backend(counts)};
    EXPECT_EQ(counts.live, 1);
    const int4 seeds[2] = {kInterleavedSeeds[0][0], kInterleavedSeeds[0][1]};
    const cuda::std::span<const int4, 2> seed_span(seeds, 2);
    typename Scheme::Cw legacy[bits + 1];
    int4 fixed[bits + 1];
    std::vector<int4> strided(static_cast<size_t>(bits) * 2 + 1, kComponentSentinel);
    const uint8_t alpha = 3;
    for (int method = 0; method < 3; ++method) {
      counts.ResetCalls();
      const int live_before = counts.live;
      uint8_t tr = 0xff;
      if (method == 0) {
        scheme.Gen(legacy, seed_span.data(), alpha, kComponentBeta);
      } else if (method == 1) {
        scheme.Gen(cuda::std::span<int4>(fixed, bits + 1), tr, seed_span, alpha,
            kComponentBeta);
      } else {
        scheme.Gen(cuda::std::span<int4>(strided.data(), strided.size()), tr, seed_span,
            alpha, kComponentBeta, size_t{2});
      }
      EXPECT_EQ(counts.full, 2 * bits);
      EXPECT_EQ(counts.half, 0);
      EXPECT_EQ(counts.copies, 0);
      EXPECT_EQ(counts.destroyed, 0);
      EXPECT_EQ(counts.live, live_before);
    }
  }
  EXPECT_EQ(counts.live, 0);
}

template <int key_count, typename Group, int bits, typename In>
void CheckGenSharedCase(bool gpu, const int4 (&seeds)[key_count][2], const In (&alphas)[key_count],
    const int4 (&betas)[key_count], const std::vector<In> (&queries)[key_count], size_t stride) {
  if (gpu) {
    CheckGenSharedGpu<key_count, Group, bits, In>(seeds, alphas, betas, queries, stride);
  } else {
    CheckGenSharedHost<key_count, Group, bits, In>(seeds, alphas, betas, queries, stride);
  }
}

template <typename Group, int bits, typename In>
void CheckSingleGenCase(bool gpu, const std::vector<In> &q, In alpha, int4 beta, size_t stride) {
  const int4 seeds[1][2] = {{kInterleavedSeeds[0][0], kInterleavedSeeds[0][1]}};
  const In alphas[1] = {alpha};
  const int4 betas[1] = {beta};
  const std::vector<In> queries[1] = {q};
  CheckGenSharedCase<1, Group, bits, In>(gpu, seeds, alphas, betas, queries, stride);
}

void CheckSingleGenCases(bool gpu) {
  const int4 beta = make_int4(kComponentBeta.x, kComponentBeta.y, kComponentBeta.z,
      kComponentBeta.w | 1);
  const __uint128_t alpha = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  for (const size_t stride : {size_t{1}, size_t{2}, size_t{7}}) {
    CheckSingleGenCase<BytesGroup, 1, uint8_t>(gpu, {1, 0}, uint8_t{1}, beta, stride);
    CheckSingleGenCase<BytesGroup, 3, uint8_t>(gpu, {5, 4, 0, 7}, uint8_t{5}, beta, stride);
    CheckSingleGenCase<BytesGroup, 8, uint16_t>(gpu, {165, 164, 0, 255}, uint16_t{165},
        beta, stride);
    CheckSingleGenCase<BytesGroup, 128, __uint128_t>(gpu, WideComponentQueries(alpha),
        alpha, beta, stride);
  }
}

void CheckInterleavedGenCases(bool gpu) {
  const int4 betas[2] = {kComponentBeta, make_int4(kInterleavedBeta.x, kInterleavedBeta.y,
      kInterleavedBeta.z, kInterleavedBeta.w | 1)};
  const uint8_t alphas1[2] = {0, 1};
  const std::vector<uint8_t> q1 = {0, 1};
  const std::vector<uint8_t> queries1[2] = {q1, q1};
  CheckGenSharedCase<2, BytesGroup, 1, uint8_t>(gpu, kInterleavedSeeds, alphas1, betas,
      queries1, 3);
  const uint8_t alphas8[2] = {2, 5};
  const std::vector<uint8_t> q80 = {2, 3, 7}, q81 = {5, 4, 0};
  const std::vector<uint8_t> queries8[2] = {q80, q81};
  CheckGenSharedCase<2, BytesGroup, 3, uint8_t>(gpu, kInterleavedSeeds, alphas8, betas,
      queries8, 3);
  const uint16_t alphas16[2] = {165, 90};
  const std::vector<uint16_t> q160 = {165, 164, 0, 255}, q161 = {90, 89, 255, 7};
  const std::vector<uint16_t> queries16[2] = {q160, q161};
  CheckGenSharedCase<2, BytesGroup, 8, uint16_t>(gpu, kInterleavedSeeds, alphas16, betas,
      queries16, 3);
  const __uint128_t alpha0 = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  const __uint128_t alpha1 = (__uint128_t{1} << 126) | (__uint128_t{1} << 63) | 0x91;
  const __uint128_t alphas128[2] = {alpha0, alpha1};
  const std::vector<__uint128_t> q1280 = WideComponentQueries(alpha0);
  const std::vector<__uint128_t> q1281 = WideComponentQueries(alpha1);
  const std::vector<__uint128_t> queries128[2] = {q1280, q1281};
  CheckGenSharedCase<2, BytesGroup, 128, __uint128_t>(gpu, kInterleavedSeeds, alphas128,
      betas, queries128, 3);
}

}  // namespace

TEST(DpfComponentGen, HostFixedAndStridedMatchLegacy) {
  CheckSingleGenCases(false);
}

TEST(DpfComponentGen, Uint127HostTerminal) {
  const int4 beta = make_int4(kComponentBeta.x, kComponentBeta.y, kComponentBeta.z,
      kComponentBeta.w | 1);
  const __uint128_t alpha = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  const int4 seeds[1][2] = {{kInterleavedSeeds[0][0], kInterleavedSeeds[0][1]}};
  const __uint128_t alphas[1] = {alpha};
  const int4 betas[1] = {beta};
  const std::vector<__uint128_t> q = WideComponentQueries(alpha);
  const std::vector<__uint128_t> queries[1] = {q};
  for (const size_t stride : {size_t{1}, size_t{2}, size_t{7}}) {
    CheckGenSharedHost<1, Uint127Group, 128, __uint128_t>(seeds, alphas, betas, queries,
        stride);
  }
}

TEST(DpfComponentGen, InterleavedSharedStrideThree) {
  CheckInterleavedGenCases(false);
}

TEST(DpfComponentGen, BackendCalls) {
  CheckGenBackendCalls<true, 5>();
  CheckGenBackendCalls<false, 5>();
}

// The split Gen is a low-level entry shared with the device, like split Eval.
#ifndef NDEBUG
TEST(DpfComponentGen, AbortsOnInvalidInputsInDebug) {
  using Prg = fss::prg::ChaCha<2>;
  using One = fss::Dpf<1, BytesGroup, Prg, uint8_t>;
  using Three = fss::Dpf<3, BytesGroup, Prg, uint8_t>;
  using Wide = fss::Dpf<128, BytesGroup, Prg, __uint128_t>;
  One one{Prg(kNonce0, kNonce1)};
  Three three{Prg(kNonce0, kNonce1)};
  Wide wide{Prg(kNonce0, kNonce1)};
  const int4 seeds[2] = {kInterleavedSeeds[0][0], kInterleavedSeeds[0][1]};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  int4 slot[4] = {kComponentSentinel, kComponentSentinel, kComponentSentinel,
      kComponentSentinel};
  const cuda::std::span<int4> one_slot(slot, 1);
  uint8_t tr8 = 0;
  __uint128_t tr128 = 0;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  EXPECT_DEATH(three.Gen(one_slot, tr8, seed_span, uint8_t{0}, kComponentBeta, size_t{0}),
      "Assertion.*failed");
  EXPECT_DEATH(one.Gen(one_slot, tr8, seed_span, uint8_t{0}, kComponentBeta, size_t{1}),
      "Assertion.*failed");
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_DEATH(three.Gen(cuda::std::span<int4>(slot, 3), tr8, seed_span, uint8_t{0},
                       kComponentBeta),
      "Assertion.*failed");
  EXPECT_DEATH(three.Gen(cuda::std::span<int4>(slot, 4), tr8, seed_span, uint8_t{0},
                       kComponentBeta, size_t{2}),
      "Assertion.*failed");
  EXPECT_DEATH(three.Gen(one_slot, tr8, seed_span, uint8_t{0}, kComponentBeta,
                       (kMaxElements - 1) / 3 + 1),
      "Assertion.*failed");
  EXPECT_DEATH(one.Gen(one_slot, tr8, seed_span, uint8_t{0}, kComponentBeta,
                       std::numeric_limits<size_t>::max()),
      "Assertion.*failed");
  EXPECT_DEATH(one.Gen(one_slot, tr8, seed_span, uint8_t{0}, kComponentBeta, kMaxElements),
      "Assertion.*failed");
  EXPECT_DEATH(wide.Gen(one_slot, tr128, seed_span, __uint128_t{0}, kComponentBeta,
                       (kMaxElements - 1) / 128 + 1),
      "Assertion.*failed");
}
#endif  // NDEBUG

TEST(DpfComponentGenGpu, DeviceGenMatchesHostLegacy) {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(status, cudaSuccess);
  CheckSingleGenCases(true);
}

TEST(DpfComponentGenGpu, InterleavedKeys) {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(status, cudaSuccess);
  CheckInterleavedGenCases(true);
}

// -----------------------------------------------------------------------------
// Split-Cws full-domain and range API: bit-exactness against the Cw forms.

namespace {

// Checks the split-Cws full-domain API against the Cw forms for one
// configuration: Gen() correction words, every EvalAll and EvalRange output, and the
// two-party reconstruction. stride == 0 selects the default-argument call.
template <int in_bits, typename Group, typename In>
void CheckDpfSplitFullDomain(In alpha, int4 beta, size_t stride) {
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::Dpf<in_bits, Group, Prg, In>;
  Scheme scheme{Prg(kNonce0, kNonce1)};
  const bool default_stride = stride == 0;
  const size_t level_stride = default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(in_bits) * level_stride + 1;
  constexpr size_t kExtra = 3;
  constexpr size_t kDomain = size_t{1} << in_bits;

  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, kComponentSeeds, alpha, beta);

  std::vector<int4> cws(used + kExtra, kComponentSentinel);
  const cuda::std::span<int4> cws_out(cws.data(), cws.size());
  const cuda::std::span<const int4, 2> seed_span(kComponentSeeds, 2);
  In tr = 0;
  if (default_stride) scheme.Gen(cws_out, tr, seed_span, alpha, beta);
  else scheme.Gen(cws_out, tr, seed_span, alpha, beta, stride);

  // The split correction words are bit-exact against the Cw layout, including the
  // left-control LSB, and only the described prefix is written.
  In expected_tr = 0;
  for (int i = 0; i < in_bits; ++i) {
    const size_t index = static_cast<size_t>(i) * level_stride;
    EXPECT_EQ(memcmp(&cws[index], &k0.cws[i].s, sizeof(int4)), 0) << "cws layer " << i;
    if (k0.cws[i].tr) expected_tr = static_cast<In>(expected_tr | static_cast<In>(In{1} << i));
  }
  EXPECT_EQ(tr, expected_tr) << "packed right-control bits";
  EXPECT_EQ(memcmp(&cws[used - 1], &k0.cws[in_bits].s, sizeof(int4)), 0) << "cws terminal";
  for (size_t i = 0; i < cws.size(); ++i) {
    if (i % level_stride == 0 && i / level_stride <= static_cast<size_t>(in_bits)) continue;
    EXPECT_EQ(memcmp(&cws[i], &kComponentSentinel, sizeof(int4)), 0) << "cws untouched at " << i;
  }

  const cuda::std::span<const int4> split_cws(cws.data(), used);
  std::vector<int4> split_all[2];
  for (int party = 0; party < 2; ++party) {
    const bool b = party != 0;
    const int4 s0 = party == 0 ? k0.s0 : k1.s0;

    for (int par_depth : {0, 2}) {
      std::vector<int4> aos_all(kDomain + 2, kComponentSentinel);
      split_all[party].assign(kDomain + 2, kComponentSentinel);
      scheme.EvalAll(b, s0, k0.cws, aos_all.data(), par_depth);
      if (default_stride) scheme.EvalAll(b, s0, split_cws, tr, split_all[party].data(), par_depth);
      else scheme.EvalAll(b, s0, split_cws, tr, split_all[party].data(), par_depth, stride);
      EXPECT_EQ(memcmp(aos_all.data(), split_all[party].data(), kDomain * sizeof(int4)), 0)
          << "EvalAll party=" << party << " par_depth=" << par_depth;
      for (size_t i = kDomain; i < split_all[party].size(); ++i) {
        EXPECT_EQ(memcmp(&split_all[party][i], &kComponentSentinel, sizeof(int4)), 0)
            << "EvalAll extra capacity at " << i;
      }
    }

    // A proper prefix, a proper suffix and the full domain, in input order.
    const In domain_last = static_cast<In>(kDomain - 1);
    const In half = static_cast<In>(kDomain / 2);
    const In intervals[3][2] = {{In{0}, domain_last}, {In{0}, half}, {half, domain_last}};
    for (const auto &interval : intervals) {
      const In first = interval[0];
      const In last = interval[1];
      const size_t count = static_cast<size_t>(last - first) + 1;
      for (int par_depth : {0, 2}) {
        std::vector<int4> aos_range(count + 2, kComponentSentinel);
        std::vector<int4> split_range(count + 2, kComponentSentinel);
        scheme.EvalRange(b, first, last, s0, k0.cws,
            cuda::std::span<int4>(aos_range.data(), count), par_depth);
        const auto ys = cuda::std::span<int4>(split_range.data(), count);
        if (default_stride) scheme.EvalRange(b, first, last, s0, split_cws, tr, ys, par_depth);
        else scheme.EvalRange(b, first, last, s0, split_cws, tr, ys, par_depth, stride);
        EXPECT_EQ(memcmp(aos_range.data(), split_range.data(), count * sizeof(int4)), 0)
            << "EvalRange party=" << party << " first=" << static_cast<uint64_t>(first)
            << " last=" << static_cast<uint64_t>(last) << " par_depth=" << par_depth;
        for (size_t i = count; i < split_range.size(); ++i) {
          EXPECT_EQ(memcmp(&split_range[i], &kComponentSentinel, sizeof(int4)), 0)
              << "EvalRange extra capacity at " << i;
        }
      }
    }
  }

  for (size_t x = 0; x < kDomain; ++x) {
    const int4 sum = (Group::From(split_all[0][x]) + Group::From(split_all[1][x])).Into();
    const int4 truth = static_cast<In>(x) == alpha ? Group::From(beta).Into() : Group{}.Into();
    EXPECT_EQ(memcmp(&sum, &truth, GroupCompareSize<Group>()), 0) << "reconstruction x=" << x;
  }
}

template <int in_bits, typename Group, typename In>
void CheckDpfSplitStrides(In alpha) {
  for (size_t stride : {size_t{0}, size_t{1}, size_t{2}, size_t{7}}) {
    CheckDpfSplitFullDomain<in_bits, Group, In>(alpha, kComponentBeta, stride);
  }
}

}  // namespace

// in_bits 1 exercises the single-level tree, 3 keeps the per-input EvalAll()
// path, and 6 and 8 run the batched EvalAll() frontier.
TEST(DpfCwsTest, Depth1) { CheckDpfSplitStrides<1, BytesGroup, uint8_t>(1); }
TEST(DpfCwsTest, Depth3) { CheckDpfSplitStrides<3, BytesGroup, uint8_t>(5); }
TEST(DpfCwsTest, Depth6) { CheckDpfSplitStrides<6, BytesGroup, uint8_t>(43); }
TEST(DpfCwsTest, Depth8) { CheckDpfSplitStrides<8, BytesGroup, uint16_t>(165); }
TEST(DpfCwsTest, Depth8Uint127) { CheckDpfSplitStrides<8, Uint127Group, uint16_t>(165); }
TEST(DpfCwsTest, Depth8GenericGroup) { CheckDpfSplitStrides<8, Uint64Group, uint16_t>(165); }

// A 128-bit input domain keeps EvalRange() in the wide-input instantiation,
// where the input width equals in_bits.
TEST(DpfCwsTest, WideInputRange) {
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::Dpf<128, Uint127Group, Prg, __uint128_t>;
  Scheme scheme{Prg(kNonce0, kNonce1)};
  const __uint128_t alpha = (__uint128_t{1} << 127) | (__uint128_t{1} << 64) | 0x42;
  const __uint128_t first = (__uint128_t{1} << 100) + 3;
  const __uint128_t last = first + 12;
  constexpr size_t kCount = 13;
  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, kComponentSeeds, alpha, kComponentBeta);

  for (const size_t stride : {size_t{1}, size_t{3}}) {
    std::vector<int4> cws(128 * stride + 1, kComponentSentinel);
    __uint128_t tr = 0;
    scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()), tr,
        cuda::std::span<const int4, 2>(kComponentSeeds, 2), alpha, kComponentBeta, stride);
    const cuda::std::span<const int4> split_cws(cws.data(), cws.size());
    for (int party = 0; party < 2; ++party) {
      const bool b = party != 0;
      const int4 s0 = party == 0 ? k0.s0 : k1.s0;
      std::vector<int4> aos(kCount + 1, kComponentSentinel);
      std::vector<int4> split(kCount + 1, kComponentSentinel);
      scheme.EvalRange(b, first, last, s0, k0.cws, cuda::std::span<int4>(aos.data(), kCount));
      scheme.EvalRange(b, first, last, s0, split_cws, tr,
          cuda::std::span<int4>(split.data(), kCount), 0, stride);
      EXPECT_EQ(memcmp(aos.data(), split.data(), kCount * sizeof(int4)), 0)
          << "party=" << party << " stride=" << stride;
      EXPECT_EQ(memcmp(&split[kCount], &kComponentSentinel, sizeof(int4)), 0);
    }
  }
}

// The split EvalAll() and EvalRange() checks throw on the host, so they are
// tested in Release too, which is where the old compiled-out assertions were
// defects. The terminal checks run before any parallel dispatch.
TEST(DpfCwsTest, RejectsInvalidEvalInputs) {
  using Prg = fss::prg::ChaCha<2>;
  using Three = fss::Dpf<3, BytesGroup, Prg, uint8_t>;
  Three three{Prg(kNonce0, kNonce1)};
  const int4 seeds[2] = {kComponentSeeds[0], kComponentSeeds[1]};
  int4 slot[10];
  for (auto &entry : slot) entry = kComponentSentinel;
  const cuda::std::span<const int4> full(slot, 4);
  const cuda::std::span<const int4> short_span(slot, 3);
  int4 ys[8] = {};
  const auto ys_span = cuda::std::span<int4>(ys, 8);
  uint8_t tr8 = 0;
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);

  // A zero stride is rejected on every split method.
  EXPECT_THROW(three.EvalAll(false, seeds[0], full, tr8, ys, 0, size_t{0}),
      std::invalid_argument);
  EXPECT_THROW(three.EvalRange(false, uint8_t{0}, uint8_t{7}, seeds[0], full, tr8,
                    ys_span, 0, size_t{0}),
      std::invalid_argument);
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_THROW(three.EvalAll(false, seeds[0], short_span, tr8, ys), std::invalid_argument);
  EXPECT_THROW(three.EvalRange(false, uint8_t{0}, uint8_t{7}, seeds[0], short_span, tr8,
                    ys_span),
      std::invalid_argument);
  // A stride past the addressable guard is rejected.
  EXPECT_THROW(three.EvalAll(false, seeds[0], full, tr8, ys, 0, (kMaxElements - 1) / 3 + 1),
      std::invalid_argument);
  EXPECT_THROW(three.EvalRange(false, uint8_t{0}, uint8_t{7}, seeds[0], full, tr8, ys_span,
                    0, (kMaxElements - 1) / 3 + 1),
      std::invalid_argument);
#ifndef NDEBUG
  EXPECT_DEBUG_DEATH(three.EvalRange(false, uint8_t{0}, uint8_t{7}, seeds[0], full, tr8,
                      ys_span, -1),
      "par_depth >= 0");
#endif  // NDEBUG
}

// The terminal LSB is a caller precondition, unlike the buffer metadata above:
// Debug builds abort on the per-input and the batched paths, and Release
// builds do not check it.
#ifndef NDEBUG
TEST(DpfCwsTest, AbortsOnBadTerminalInDebug) {
  using Prg = fss::prg::ChaCha<2>;
  using Three = fss::Dpf<3, BytesGroup, Prg, uint8_t>;
  using Eight = fss::Dpf<8, BytesGroup, Prg, uint16_t>;
  Three three{Prg(kNonce0, kNonce1)};
  Eight eight{Prg(kNonce0, kNonce1)};
  const int4 seeds[2] = {kComponentSeeds[0], kComponentSeeds[1]};
  int4 odd[10];
  for (auto &entry : odd) entry = kComponentSentinel;
  int4 ys[8] = {};
  const auto ys_span = cuda::std::span<int4>(ys, 8);
  uint8_t tr8 = 0;
  uint16_t tr16 = 0;
  odd[3].w |= 1;
  EXPECT_DEATH(three.EvalAll(false, seeds[0], cuda::std::span<const int4>(odd, 4), tr8, ys),
      "Assertion.*failed");
  EXPECT_DEATH(three.EvalRange(false, uint8_t{0}, uint8_t{7}, seeds[0],
                       cuda::std::span<const int4>(odd, 4), tr8, ys_span),
      "Assertion.*failed");
  odd[3].w &= ~1;
  odd[8].w |= 1;
  EXPECT_DEATH(eight.EvalAll(false, seeds[0], cuda::std::span<const int4>(odd, 9), tr16, ys),
      "Assertion.*failed");
}
#endif  // NDEBUG

namespace {

// An odd terminal violates a documented precondition: Debug builds must abort
// on the calling thread, before any parallel dispatch, whatever the par_depth.
template <int in_bits>
void CheckDpfOddTerminalRejected(int par_depth) {
  using Scheme = fss::Dpf<in_bits, BytesGroup, fss::prg::ChaCha<2>, uint8_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const int4 s0 = {13, 27, 45, 66};
  typename Scheme::Cw cws[in_bits + 1]{};
  cws[in_bits].s.w |= 1;
  int4 split_cws[in_bits + 1] = {};
  split_cws[in_bits].w = 1;
  const uint8_t tr = 0;
  constexpr size_t n = size_t{1} << in_bits;
  int4 ys_storage[n];
  EXPECT_DEATH(scheme.EvalAll(false, s0, cws, ys_storage, par_depth), "Assertion.*failed");
  EXPECT_DEATH(scheme.EvalRange(false, uint8_t{0}, uint8_t{n - 1}, s0, cws,
      cuda::std::span<int4>(ys_storage, n), par_depth), "Assertion.*failed");
  EXPECT_DEATH(scheme.EvalAll(false, s0, cuda::std::span<const int4>(split_cws, in_bits + 1),
      tr, ys_storage, par_depth), "Assertion.*failed");
  EXPECT_DEATH(scheme.EvalRange(false, uint8_t{0}, uint8_t{n - 1}, s0,
      cuda::std::span<const int4>(split_cws, in_bits + 1), tr,
      cuda::std::span<int4>(ys_storage, n), par_depth), "Assertion.*failed");
}

}  // namespace

// in_bits 3 keeps the per-input walk and in_bits 8 uses the batched path; both
// must abort on an odd terminal on the calling thread, before OpenMP work,
// whatever the par_depth. Release builds do not check the precondition, so the
// invalid inputs run only in Debug.
#ifndef NDEBUG
TEST(DpfTerminalTest, RejectsOddTerminalBeforeParallelDispatch) {
  for (int par_depth = 0; par_depth <= 1; ++par_depth) {
    SCOPED_TRACE(par_depth);
    CheckDpfOddTerminalRejected<3>(par_depth);
    CheckDpfOddTerminalRejected<8>(par_depth);
  }
}
#endif  // NDEBUG

// The batch geometry validation is host code that runs before any CUDA call, so
// these checks need no device and read no buffer.
TEST(DpfSplitGpuTest, RejectsInvalidBatchGeometry) {
  using Wide = fss::Dpf<32, BytesGroup, fss::prg::ChaCha<2>, uint64_t>;
  Wide wide{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  int4 seed_storage[1] = {};
  int4 cws_storage[1] = {};
  int4 ys_storage[1] = {};
  uint64_t tr_storage[1] = {};

  // 2^32 keys of 2^32 outputs make the output product wrap in size_t, so the
  // guard must reject the call before the product is formed. The correction word
  // spans are sized nominally here and the key count is above the launch grid
  // limit for every launch shape, so a variant without the output guard would
  // stop at cudaErrorInvalidConfiguration instead of launching into the
  // one-element output buffer.
  constexpr size_t kKeys = size_t{1} << 32;
  constexpr size_t kMinCws = 32 * kKeys + kKeys;
  EXPECT_EQ((wide.template EvalAllGpuBatch<24, 256>(false,
      cuda::std::span<const int4>(seed_storage, kKeys),
      cuda::std::span<const int4>(cws_storage, kMinCws),
      cuda::std::span<const uint64_t>(tr_storage, kKeys),
      cuda::std::span<int4>(ys_storage, 1), 0)), cudaErrorInvalidValue);

  using Small = fss::Dpf<8, BytesGroup, fss::prg::ChaCha<2>, uint16_t>;
  Small small{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  constexpr size_t kSmallKeys = 2;
  constexpr size_t kSmallRequired = 8 * kSmallKeys + kSmallKeys;
  int4 small_s0s[kSmallKeys] = {};
  int4 small_cws[kSmallRequired] = {};
  int4 small_ys[kSmallKeys * 256] = {};
  uint16_t small_tr[kSmallKeys] = {};
  const auto s0s = cuda::std::span<const int4>(small_s0s, kSmallKeys);
  const auto cws = cuda::std::span<const int4>(small_cws, kSmallRequired);
  const auto tr = cuda::std::span<const uint16_t>(small_tr, kSmallKeys);
  const auto ys = cuda::std::span<int4>(small_ys, kSmallKeys * 256);

  // A stride below the key count is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s, cws, tr, ys, 0, 1)),
      cudaErrorInvalidValue);
  // A stride past the element guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s, cws, tr, ys, 0,
                        kMaxElements + 1)),
      cudaErrorInvalidValue);
  // A span of correction words that omits the terminal column is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s,
                        cuda::std::span<const int4>(small_cws, kSmallRequired - 1), tr, ys)),
      cudaErrorInvalidValue);
  // An empty batch succeeds without reading any buffer.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, cuda::std::span<const int4>(),
                        cuda::std::span<const int4>(), cuda::std::span<const uint16_t>(),
                        cuda::std::span<int4>())),
      cudaSuccess);
}

// The single-key GPU validation is host code that runs before any CUDA call,
// so these checks need no device either.
TEST(DpfSplitGpuTest, RejectsInvalidSingleKeyGeometry) {
  using Scheme = fss::Dpf<8, BytesGroup, fss::prg::ChaCha<2>, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  constexpr size_t kDomain = size_t{1} << 8;
  int4 cws_storage[9] = {};
  int4 ys_storage[kDomain] = {};
  const int4 s0 = {1, 2, 3, 4};
  const uint16_t tr = 0;
  const auto cws = cuda::std::span<const int4>(cws_storage, 9);
  const auto ys = cuda::std::span<int4>(ys_storage, kDomain);
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);

  // A zero stride is valid only in a batch method, so a non-batch method rejects it.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, s0, cws, tr, ys, 0, 0)),
      cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{255}, s0, cws,
                        tr, ys, 0, 0)),
      cudaErrorInvalidValue);
  // A stride past the addressable guard is rejected before the span is read.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, s0, cws, tr, ys, 0,
                        (kMaxElements - 1) / 8 + 1)),
      cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{255}, s0, cws,
                        tr, ys, 0, (kMaxElements - 1) / 8 + 1)),
      cudaErrorInvalidValue);
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, s0,
                        cuda::std::span<const int4>(cws_storage, 8), tr, ys)),
      cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{255}, s0,
                        cuda::std::span<const int4>(cws_storage, 8), tr, ys)),
      cudaErrorInvalidValue);
  // Inverted and out-of-domain endpoints are rejected.
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{7}, uint16_t{3}, s0, cws,
                        tr, ys)),
      cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{300}, s0, cws,
                        tr, ys)),
      cudaErrorInvalidValue);
}

// The GPU split launches are compile-verified here and executed only on a
// machine with a free device, comparing against the host Cw-form outputs. The
// wide Group instantiation also covers the split host stubs.
namespace {

template <typename Group>
void CheckDpfSplitGpuLaunches() {
  int devices = 0;
  cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
      (device_status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(device_status, cudaSuccess);

  constexpr int kBits = 8;
  constexpr size_t kDomain = size_t{1} << kBits;
  using Scheme = fss::Dpf<kBits, Group, fss::prg::ChaCha<2>, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 beta = {7, 19, 23, 42};
  const uint16_t alpha = 107;
  constexpr size_t kStride = 2;

  typename Scheme::Key k0, k1;
  fss_test::GenPair(scheme, k0, k1, cuda::std::span<const int4, 2>(seeds, 2), alpha, beta);

  std::vector<int4> cws((kBits + 1) * kStride + 2, kComponentSentinel);
  uint16_t tr = 0;
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha, beta, kStride);

  int4 *device_cws = nullptr;
  uint16_t *device_tr = nullptr;
  int4 *device_ys = nullptr;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  constexpr size_t kKeys = 2;
  ASSERT_EQ(cudaMalloc(&device_cws, cws.size() * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tr, kKeys * sizeof(uint16_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, kDomain * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_cws, cws.data(), cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_tr, &tr, sizeof(uint16_t), cudaMemcpyHostToDevice, stream),
      cudaSuccess);

  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(party);
    const bool b = party != 0;
    const int4 s0 = party == 0 ? k0.s0 : k1.s0;
    const auto span_cws = cuda::std::span<const int4>(device_cws, cws.size());

    // EvalAllGpu matches the host Cw form block for block.
    std::vector<int4> expected(kDomain, kComponentSentinel);
    scheme.EvalAll(b, s0, k0.cws, expected.data());
    std::vector<int4> actual(kDomain, kComponentSentinel);
    ASSERT_EQ((scheme.template EvalAllGpu<8, 16>(b, s0, span_cws, tr,
        cuda::std::span<int4>(device_ys, kDomain), stream, kStride)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_ys, kDomain * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(memcmp(actual.data(), expected.data(), kDomain * sizeof(int4)), 0);

    // EvalRangeGpu over a partial interval leaves the extra device capacity untouched.
    std::vector<int4> filled(kDomain, kComponentSentinel);
    ASSERT_EQ(cudaMemcpyAsync(device_ys, filled.data(), kDomain * sizeof(int4),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ((scheme.template EvalRangeGpu<8, 16>(b, uint16_t{13}, uint16_t{201}, s0,
        span_cws, tr, cuda::std::span<int4>(device_ys, kDomain), stream, kStride)),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(filled.data(), device_ys, kDomain * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (uint16_t x = 13; x <= 201; ++x) {
      const int4 oracle = scheme.Eval(b, s0, k0.cws, x);
      EXPECT_EQ(memcmp(&filled[x - 13], &oracle, sizeof(int4)), 0) << "range x=" << x;
    }
    for (size_t i = 201 - 13 + 1; i < kDomain; ++i) {
      EXPECT_EQ(memcmp(&filled[i], &kComponentSentinel, sizeof(int4)), 0) << "untouched at " << i;
    }
  }

  // A zero stride and a span that omits the terminal slot are rejected.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, k0.s0,
      cuda::std::span<const int4>(device_cws, cws.size()), tr,
      cuda::std::span<int4>(device_ys, kDomain), stream, 0)), cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{255}, k0.s0,
      cuda::std::span<const int4>(device_cws, kBits), tr,
      cuda::std::span<int4>(device_ys, kDomain), stream)), cudaErrorInvalidValue);

  // Batch launches: the default stride is the dense level-major key count.
  typename Scheme::Cw ref_batch[kKeys][kBits + 1];
  std::vector<int4> batch_cws((kBits + 1) * kKeys, kComponentSentinel);
  int4 batch_s0s[kKeys];
  uint16_t batch_tr[kKeys] = {0, 0};
  for (size_t key = 0; key < kKeys; ++key) {
    batch_s0s[key] = seeds[key];
    scheme.Gen(ref_batch[key], seeds, static_cast<uint16_t>(alpha + key), beta);
    uint16_t key_tr = 0;
    // Offsetting by the key index fills the dense level-major layout.
    scheme.Gen(cuda::std::span<int4>(batch_cws.data() + key, batch_cws.size() - key), key_tr,
        cuda::std::span<const int4, 2>(seeds, 2), static_cast<uint16_t>(alpha + key), beta,
        kKeys);
    batch_tr[key] = key_tr;
  }
  ASSERT_EQ(cudaMemcpyAsync(device_cws, batch_cws.data(), batch_cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_tr, batch_tr, sizeof(batch_tr), cudaMemcpyHostToDevice,
      stream), cudaSuccess);
  int4 *device_s0s = nullptr;
  ASSERT_EQ(cudaMalloc(&device_s0s, sizeof(batch_s0s)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_s0s, batch_s0s, sizeof(batch_s0s),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  int4 *device_batch_ys = nullptr;
  ASSERT_EQ(cudaMalloc(&device_batch_ys, kKeys * kDomain * sizeof(int4)), cudaSuccess);
  const auto span_batch_cws = cuda::std::span<const int4>(device_cws, batch_cws.size());
  const auto span_batch_tr = cuda::std::span<const uint16_t>(device_tr, kKeys);
  const auto span_batch_ys = cuda::std::span<int4>(device_batch_ys, kKeys * kDomain);
  for (int use_default = 0; use_default < 2; ++use_default) {
    SCOPED_TRACE(use_default);
    const size_t stride = use_default == 0 ? 0 : kKeys;
    ASSERT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false,
        cuda::std::span<const int4>(device_s0s, kKeys), span_batch_cws, span_batch_tr,
        span_batch_ys, stream, stride)), cudaSuccess);
    std::vector<int4> actual(kKeys * kDomain, kComponentSentinel);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_batch_ys, actual.size() * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t key = 0; key < kKeys; ++key) {
      for (size_t x = 0; x < kDomain; ++x) {
        const int4 oracle = scheme.Eval(false, batch_s0s[key], ref_batch[key],
            static_cast<uint16_t>(x));
        EXPECT_EQ(memcmp(&actual[key * kDomain + x], &oracle, sizeof(int4)), 0)
            << "key=" << key << " x=" << x;
      }
    }
  }
  // A stride below the key count is rejected.
  EXPECT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false,
      cuda::std::span<const int4>(device_s0s, kKeys), span_batch_cws, span_batch_tr,
      span_batch_ys, stream, 1)), cudaErrorInvalidValue);

  EXPECT_EQ(cudaFree(device_batch_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_s0s), cudaSuccess);
  EXPECT_EQ(cudaFree(device_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_tr), cudaSuccess);
  EXPECT_EQ(cudaFree(device_cws), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace

TEST(DpfSplitGpuTest, SplitLaunchesMatchCwHost) { CheckDpfSplitGpuLaunches<BytesGroup>(); }
TEST(DpfSplitGpuTest, WideGroupSplitLaunchesCompile) { CheckDpfSplitGpuLaunches<Uint127Group>(); }

// The split EvalAll and EvalRange buffer checks stay host-boundary throws; the
// split Gen, split Eval, and terminal requirements are low-level preconditions.
TEST(DpfCwsTest, RejectsInvalidInputs) {
  using Scheme = fss::Dpf<3, BytesGroup, fss::prg::ChaCha<2>, uint8_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  int4 cws[4] = {};
  typename Scheme::Cw raw_cws[4] = {};
  int4 ys[8] = {};
  const int4 s0 = seeds[0];

  EXPECT_THROW((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(cws, 3),
                        uint8_t{0}, ys),
      std::invalid_argument);

  // Range checks: reversed and out-of-domain intervals, short output, null input.
  EXPECT_THROW((void)scheme.EvalRange(false, 3, 2, s0, raw_cws,
                        cuda::std::span<int4>(ys, 8)),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalRange(false, 0, 8, s0, raw_cws,
                        cuda::std::span<int4>(ys, 8)),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalRange(false, 0, 7, s0, raw_cws,
                        cuda::std::span<int4>(ys, 7)),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalRange(false, 0, 7, s0,
                        static_cast<const typename Scheme::Cw *>(nullptr),
                        cuda::std::span<int4>(ys, 8)),
      std::invalid_argument);
}

#ifndef NDEBUG
TEST(DpfCwsTest, AbortsOnInvalidLowLevelInputsInDebug) {
  using Scheme = fss::Dpf<3, BytesGroup, fss::prg::ChaCha<2>, uint8_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  int4 cws[4] = {};
  int4 ys[8] = {};
  uint8_t tr = 0;
  const auto cws_out = cuda::std::span<int4>(cws, 4);
  const auto cws_in = cuda::std::span<const int4>(cws, 4);
  const int4 s0 = seeds[0];

  // A zero stride is valid only in a batch method, so a non-batch method rejects it.
  EXPECT_DEATH((void)scheme.Gen(cws_out, tr, seed_span, uint8_t{5}, kComponentBeta, 0),
      "Assertion.*failed");
  EXPECT_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, uint8_t{0}, 0),
      "Assertion.*failed");
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_DEATH((void)scheme.Gen(cuda::std::span<int4>(cws, 3), tr, seed_span,
                       uint8_t{5}, kComponentBeta),
      "Assertion.*failed");
  EXPECT_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(cws, 3),
                       uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  // The terminal output correction word must have a zero LSB.
  int4 odd_cws[4] = {};
  odd_cws[3].w = 1;
  EXPECT_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(odd_cws, 4),
                       uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  EXPECT_DEATH((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(odd_cws, 4),
                       uint8_t{0}, ys),
      "Assertion.*failed");
  // A stride past the addressable guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / 3 + 1;
  EXPECT_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, uint8_t{0},
                       overflowing_stride),
      "Assertion.*failed");
}
#endif  // NDEBUG
