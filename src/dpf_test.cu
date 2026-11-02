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
  typename Scheme::Cw corrections[bits + 1];
  const int4 seeds[2] = {{0x12345678, 0x1234abcd, 0x13579bdf, 0x2468ace0},
      {0x7abc1234, 0x01bc3456, 0x4567abcd, 0x76892344}};
  const int4 beta = Group::From(make_int4(0x5abc0123, 0x44556677, 0x12345678, 0x6abc1234)).Into();
  scheme.Gen(corrections, seeds, alpha, beta);
  const size_t count = static_cast<size_t>(static_cast<__uint128_t>(last) - first) + 1;
  const int4 sentinel = make_int4(0x9876, 0x1234, 0x5678, 0xabcd);
  std::vector<int4> outputs[2] = {
      std::vector<int4>(count + 3, sentinel), std::vector<int4>(count + 3, sentinel)};
  for (int party = 0; party < 2; ++party) {
    scheme.EvalRange(party, first, last, seeds[party], corrections,
        cuda::std::span<int4>(outputs[party].data() + 1, count + 2), par_depth);
    EXPECT_EQ(memcmp(&outputs[party][0], &sentinel, sizeof(int4)), 0);
    EXPECT_EQ(memcmp(&outputs[party][count + 1], &sentinel, sizeof(int4)), 0);
    EXPECT_EQ(memcmp(&outputs[party][count + 2], &sentinel, sizeof(int4)), 0);
    for (size_t offset = 0; offset < count; ++offset) {
      const In point = first + static_cast<In>(offset);
      const int4 scalar = scheme.Eval(party, seeds[party], corrections, point);
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
