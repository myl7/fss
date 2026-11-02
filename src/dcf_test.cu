#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prg/aes128_mmo_raw.cuh>
#include "core_api_test_helpers.cuh"
#include "range_gpu_test_helpers.cuh"
#include <atomic>
#include <thread>
#include <climits>

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

// Random nonce for ChaCha (DCF needs mul=4)
constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// Test fixture template for DCF
template <typename Group, typename Prg, fss::DcfPred pred = fss::DcfPred::kLt>
class DcfTestBase : public ::testing::Test {
protected:
  using DcfType = fss::Dcf<kAlphaBits, Group, Prg, uint16_t, pred>;

  int4 s0s[2];
  typename DcfType::Key k0, k1;
  Prg prg;

  DcfTestBase(Prg p) : prg(p) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void TestEvalAll() {
    DcfType dcf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);

    fss_test::EvalAllKey(dcf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey(dcf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    auto expected = Group::From(b_buf);
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto g0 = Group::From(ys0[x]);
      auto g1 = Group::From(ys1[x]);
      auto result = g0 + g1;
      int4 result_buf = result.Into();

      if (x < kAlpha) {
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0)
            << "Failed at x=" << x << " (expected beta)";
      } else {
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
            << "Failed at x=" << x << " (expected zero)";
      }
    }
  }

  void TestEvalLt() {
    DcfType dcf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Test x < alpha should return beta
    std::uniform_int_distribution<uint16_t> dis_lt(0, kAlpha - 1);
    for (int i = 0; i < 50; i++) {
      uint16_t x = dis_lt(gen);

      int4 y0 = fss_test::EvalKey(dcf, false, k0, x);
      int4 y1 = fss_test::EvalKey(dcf, true, k1, x);

      auto g0 = Group::From(y0);
      auto g1 = Group::From(y1);
      auto result = g0 + g1;
      auto expected = Group::From(b_buf);

      int4 result_buf = result.Into();
      int4 expected_buf = expected.Into();
      EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }

    // Test x >= alpha should return 0
    std::uniform_int_distribution<uint16_t> dis_ge(kAlpha, 0xFFFF);
    for (int i = 0; i < 50; i++) {
      uint16_t x = dis_ge(gen);

      int4 y0 = fss_test::EvalKey(dcf, false, k0, x);
      int4 y1 = fss_test::EvalKey(dcf, true, k1, x);

      auto g0 = Group::From(y0);
      auto g1 = Group::From(y1);
      auto result = g0 + g1;

      int4 result_buf = result.Into();
      int4 zero = {0, 0, 0, 0};
      EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }
  }
};

// These tests exercise the 8-wide batched EvalAll() frontier at its in_bits and
// par_depth boundaries with the AES-NI PRG whose Gen8() the batch pipelines through,
// the ChaCha PRG that falls back to the detail::GenBatch() Gen loop, and both
// comparison predicates. in_bits below kBatchPrefix + kBatchTail (6) keeps the
// recursive tree walk, in_bits = 6 is the smallest batched domain and clamps the
// split depth to 0, and larger in_bits run the sequential batched walk
// (par_depth = 0) beside the frontier-width split (par_depth > 0). Bit-exactness is
// gated against the scalar point Eval(), which the frontier rewrite leaves untouched.
template <int in_bits, typename Group, typename Prg, typename In = uint32_t,
    fss::DcfPred pred = fss::DcfPred::kLt, int par_depth = 0>
class DcfFrontierTestBase : public ::testing::Test {
protected:
  using DcfType = fss::Dcf<in_bits, Group, Prg, In, pred>;

  int4 s0s[2];
  typename DcfType::Key k0, k1;
  Prg prg;

  DcfFrontierTestBase() : prg(MakePrg()) {}

  static Prg MakePrg() {
    if constexpr (std::is_same_v<Prg, fss::prg::Aes128MmoRaw<4>>) {
      static constexpr uint8_t keys[4][16] = {
          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
          {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
          {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8},
          {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1},
      };
      return Prg(keys);
    } else {
      return Prg(kNonce0, kNonce1);
    }
  }

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

  // Full-domain share sum: beta on the predicate side of alpha, identity elsewhere.
  void TestEvalAll(In alpha) {
    DcfType dcf{prg};

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    constexpr size_t n = size_t{1} << in_bits;
    std::vector<int4> ys0(n), ys1(n);
    fss_test::EvalAllKey<par_depth>(dcf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey<par_depth>(dcf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    auto expected = Group::From(b_buf);
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
      int4 result_buf = result.Into();

      In xi = static_cast<In>(x);
      bool hit = pred == fss::DcfPred::kLt ? xi < alpha : xi > alpha;
      if (hit) {
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
      } else {
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
      }
    }
  }

  // Bit-exactness of the batched EvalAll() against the scalar point Eval().
  void TestEvalAllMatchesEval(In alpha) {
    DcfType dcf{prg};

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    constexpr size_t n = size_t{1} << in_bits;
    std::vector<int4> ys0(n), ys1(n);
    fss_test::EvalAllKey<par_depth>(dcf, false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    fss_test::EvalAllKey<par_depth>(dcf, true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    for (size_t x = 0; x < n; ++x) {
      int4 e0 = fss_test::EvalKey(dcf, false, k0, static_cast<In>(x));
      int4 e1 = fss_test::EvalKey(dcf, true, k1, static_cast<In>(x));
      EXPECT_EQ(memcmp(&ys0[x], &e0, sizeof(int4)), 0) << "Party 0 mismatch at x=" << x;
      EXPECT_EQ(memcmp(&ys1[x], &e1, sizeof(int4)), 0) << "Party 1 mismatch at x=" << x;
    }
  }
};

using AesRaw4 = fss::prg::Aes128MmoRaw<4>;
using ChaCha4 = fss::prg::ChaCha<4>;

// in_bits 5 keeps the recursive tree walk of the small-domain branch.
class DcfBytesAesRaw5BitsTest : public DcfFrontierTestBase<5, BytesGroup, AesRaw4> {};
class DcfBytesAesRaw6BitsSeqTest
    : public DcfFrontierTestBase<6, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kLt, 0> {};
class DcfBytesAesRaw6BitsClampedSplitTest
    : public DcfFrontierTestBase<6, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kLt, 2> {};
class DcfBytesAesRaw8BitsSeqTest
    : public DcfFrontierTestBase<8, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kLt, 0> {};
class DcfBytesAesRaw8BitsSplitTest
    : public DcfFrontierTestBase<8, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kLt, 2> {};
class DcfBytesAesRaw16BitsDefaultTest : public DcfFrontierTestBase<16, BytesGroup, AesRaw4> {};
class DcfBytesAesRaw20BitsSeqTest
    : public DcfFrontierTestBase<20, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kLt, 0> {};
class DcfUint128AesRaw8BitsSplitTest
    : public DcfFrontierTestBase<8, Uint127Group, AesRaw4, uint32_t, fss::DcfPred::kLt, 2> {};
class DcfUint128AesRaw16BitsDefaultTest : public DcfFrontierTestBase<16, Uint127Group, AesRaw4> {};
class DcfUint128AesRaw20BitsSeqTest
    : public DcfFrontierTestBase<20, Uint127Group, AesRaw4, uint32_t, fss::DcfPred::kLt, 0> {};
class DcfBytesAesRawGt8BitsSeqTest
    : public DcfFrontierTestBase<8, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kGt, 0> {};
class DcfBytesAesRawGt16BitsDefaultTest
    : public DcfFrontierTestBase<16, BytesGroup, AesRaw4, uint32_t, fss::DcfPred::kGt> {};
class DcfUint128AesRawGt8BitsSplitTest
    : public DcfFrontierTestBase<8, Uint127Group, AesRaw4, uint32_t, fss::DcfPred::kGt, 2> {};
class DcfUint128AesRawGt16BitsDefaultTest
    : public DcfFrontierTestBase<16, Uint127Group, AesRaw4, uint32_t, fss::DcfPred::kGt> {};
// ChaCha has no Gen8(), so EvalAll() runs the frontier through the detail::GenBatch() loop.
class DcfBytesChaCha8BitsSeqTest
    : public DcfFrontierTestBase<8, BytesGroup, ChaCha4, uint32_t, fss::DcfPred::kLt, 0> {};
class DcfBytesChaCha16BitsDefaultTest : public DcfFrontierTestBase<16, BytesGroup, ChaCha4> {};
class DcfUint128ChaCha8BitsSplitTest
    : public DcfFrontierTestBase<8, Uint127Group, ChaCha4, uint32_t, fss::DcfPred::kLt, 2> {};
class DcfUint128ChaCha16BitsDefaultTest : public DcfFrontierTestBase<16, Uint127Group, ChaCha4> {};

TEST_F(DcfBytesAesRaw5BitsTest, EvalAll) {
  TestEvalAll(11);
  TestEvalAllMatchesEval(11);
}
TEST_F(DcfBytesAesRaw6BitsSeqTest, EvalAll) {
  TestEvalAll(43);
  TestEvalAllMatchesEval(43);
}
TEST_F(DcfBytesAesRaw6BitsClampedSplitTest, EvalAll) {
  TestEvalAll(43);
  TestEvalAllMatchesEval(43);
}
TEST_F(DcfBytesAesRaw8BitsSeqTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesAesRaw8BitsSplitTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesAesRaw16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesAesRaw20BitsSeqTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128AesRaw8BitsSplitTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128AesRaw16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128AesRaw20BitsSeqTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesAesRawGt8BitsSeqTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesAesRawGt16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128AesRawGt8BitsSplitTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128AesRawGt16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesChaCha8BitsSeqTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfBytesChaCha16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128ChaCha8BitsSplitTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}
TEST_F(DcfUint128ChaCha16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}

// Alpha at the domain edges: 0 and 2 ** in_bits - 1, for both predicates.
TEST_F(DcfBytesAesRaw8BitsSeqTest, EvalAllEdgeAlpha) {
  TestEvalAll(0);
  TestEvalAllMatchesEval(0);
  TestEvalAll((1u << 8) - 1);
  TestEvalAllMatchesEval((1u << 8) - 1);
}
TEST_F(DcfUint128AesRawGt8BitsSplitTest, EvalAllEdgeAlpha) {
  TestEvalAll(0);
  TestEvalAllMatchesEval(0);
  TestEvalAll((1u << 8) - 1);
  TestEvalAllMatchesEval((1u << 8) - 1);
}

// ChaCha PRG tests (mul=4 for DCF)
class DcfBytesChaChaTest : public DcfTestBase<BytesGroup, fss::prg::ChaCha<4>> {
public:
  DcfBytesChaChaTest() : DcfTestBase(fss::prg::ChaCha<4>(kNonce0, kNonce1)) {}
};

class DcfUint128ChaChaTest : public DcfTestBase<Uint127Group, fss::prg::ChaCha<4>> {
public:
  DcfUint128ChaChaTest() : DcfTestBase(fss::prg::ChaCha<4>(kNonce0, kNonce1)) {}
};

class DcfUint64ChaChaTest : public DcfTestBase<Uint64Group, fss::prg::ChaCha<4>> {
public:
  DcfUint64ChaChaTest() : DcfTestBase(fss::prg::ChaCha<4>(kNonce0, kNonce1)) {}
};

// AES PRG tests (mul=4 for DCF)
class DcfBytesAesTest : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128Mmo<4>;
  using DcfType = fss::Dcf<kAlphaBits, Group, Prg, uint16_t>;

  int4 s0s[2];
  DcfType::Key k0, k1;
  cuda::std::array<EVP_CIPHER_CTX *, 4> ctxs;

  void SetUp() override {
    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    unsigned char key2[16] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
    unsigned char key3[16] = {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
    const unsigned char *keys[4] = {key0, key1, key2, key3};
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
TEST_F(DcfBytesChaChaTest, EvalLt) {
  TestEvalLt();
}
TEST_F(DcfUint128ChaChaTest, EvalLt) {
  TestEvalLt();
}
TEST_F(DcfUint64ChaChaTest, EvalLt) {
  TestEvalLt();
}
TEST_F(DcfBytesChaChaTest, EvalAll) {
  TestEvalAll();
}
TEST_F(DcfUint128ChaChaTest, EvalAll) {
  TestEvalAll();
}
TEST_F(DcfUint64ChaChaTest, EvalAll) {
  TestEvalAll();
}

// Tests for AES
TEST_F(DcfBytesAesTest, EvalLt) {
  Prg prg(ctxs);
  DcfType dcf{prg};

  int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
      static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

  std::random_device rd;
  std::mt19937 gen(rd());

  // Test x < alpha
  std::uniform_int_distribution<uint16_t> dis_lt(0, kAlpha - 1);
  for (int i = 0; i < 50; i++) {
    uint16_t x = dis_lt(gen);

    int4 y0 = fss_test::EvalKey(dcf, false, k0, x);
    int4 y1 = fss_test::EvalKey(dcf, true, k1, x);

    auto g0 = Group::From(y0);
    auto g1 = Group::From(y1);
    auto result = g0 + g1;
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
  }

  // Test x >= alpha
  std::uniform_int_distribution<uint16_t> dis_ge(kAlpha, 0xFFFF);
  for (int i = 0; i < 50; i++) {
    uint16_t x = dis_ge(gen);

    int4 y0 = fss_test::EvalKey(dcf, false, k0, x);
    int4 y1 = fss_test::EvalKey(dcf, true, k1, x);

    auto g0 = Group::From(y0);
    auto g1 = Group::From(y1);
    auto result = g0 + g1;

    int4 result_buf = result.Into();
    int4 zero = {0, 0, 0, 0};
    EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
  }
}


namespace {

template <int in_bits, typename Group, fss::DcfPred pred, typename In = uint32_t>
void CheckDcfRange(In first, In last, In alpha, int par_depth) {
  using Scheme = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, In, pred>;
  Scheme scheme(fss::prg::ChaCha<4>(kNonce0, kNonce1));
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const int4 beta = {604, 981, 273, 84};
  struct Corrections {
    int4 before;
    typename Scheme::Cw words[in_bits + 1];
    int4 after;
  } corrections{};
  const int4 sentinel = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};
  corrections.before = corrections.after = sentinel;
  scheme.Gen(corrections.words, seeds, alpha, beta);
  const auto saved_corrections = corrections;
  const size_t count = static_cast<size_t>(last - first) + 1;
  std::vector<int4> shares[2] = {
      std::vector<int4>(count + 4, sentinel), std::vector<int4>(count + 4, sentinel)};
  for (int party = 0; party < 2; ++party) {
    scheme.EvalRange(party, first, last, seeds[party], corrections.words,
        cuda::std::span<int4>(shares[party].data() + 1, count + 2), par_depth);
    if (par_depth > 0) {
      std::vector<int4> sequential(count);
      scheme.EvalRange(party, first, last, seeds[party], corrections.words,
          cuda::std::span<int4>(sequential.data(), sequential.size()), 0);
      EXPECT_EQ(memcmp(shares[party].data() + 1, sequential.data(), count * sizeof(int4)), 0);
    }
    for (size_t offset = 0; offset < count; ++offset) {
      const int4 point = scheme.Eval(party, seeds[party], corrections.words,
          first + static_cast<In>(offset));
      EXPECT_EQ(memcmp(&shares[party][offset + 1], &point, sizeof(int4)), 0)
          << "party=" << party << " offset=" << offset << " depth=" << par_depth;
    }
    for (size_t untouched : {size_t{0}, count + 1, count + 2, count + 3}) {
      EXPECT_EQ(memcmp(&shares[party][untouched], &sentinel, sizeof(int4)), 0);
    }
  }
  EXPECT_EQ(memcmp(&corrections, &saved_corrections, sizeof(corrections)), 0);
  for (size_t offset = 0; offset < count; ++offset) {
    const In x = first + static_cast<In>(offset);
    const bool hit = pred == fss::DcfPred::kLt ? x < alpha : x > alpha;
    const int4 expected = hit ? Group::From(beta).Into() : Group{}.Into();
    const int4 actual = (Group::From(shares[0][offset + 1]) +
        Group::From(shares[1][offset + 1])).Into();
    EXPECT_EQ(memcmp(&actual, &expected, sizeof(int4)), 0)
        << "offset=" << offset << " depth=" << par_depth;
  }
}

template <typename Group, fss::DcfPred pred>
void CheckDcfSmallRanges() {
  for (uint32_t alpha = 0; alpha < 8; ++alpha) {
    for (uint32_t first = 0; first < 8; ++first) {
      for (uint32_t last = first; last < 8; ++last) {
        CheckDcfRange<3, Group, pred>(first, last, alpha, 0);
        CheckDcfRange<3, Group, pred>(first, last, alpha, 2);
      }
    }
  }
  for (int depth : {0, 1, 3, 12, INT_MAX}) {
    CheckDcfRange<8, Group, pred, uint32_t>(0, 255, 107, depth);
    CheckDcfRange<8, Group, pred, uint32_t>(63, 130, 107, depth);
    CheckDcfRange<8, Group, pred, uint32_t>(104, 116, 107, depth);
    CheckDcfRange<8, Group, pred, uint32_t>(255, 255, 255, depth);
  }
}

template <typename Group, fss::DcfPred pred>
void CheckDcfWideRanges() {
  using In = __uint128_t;
  const In domain_last = ~In{0};
  for (int depth : {0, 2, 8}) {
    CheckDcfRange<128, Group, pred>(domain_last - 12, domain_last, domain_last - 5, depth);
    CheckDcfRange<128, Group, pred>((In{1} << 64) - 7, (In{1} << 64) + 5, In{1} << 64, depth);
    CheckDcfRange<128, Group, pred>((In{1} << 100) + 3, (In{1} << 100) + 15,
        (In{1} << 100) + 8, depth);
    CheckDcfRange<128, Group, pred>(domain_last, domain_last, domain_last, depth);
    CheckDcfRange<100, Group, pred>((In{1} << 100) - 13, (In{1} << 100) - 1,
        (In{1} << 100) - 5, depth);
    CheckDcfRange<64, Group, pred, uint64_t>(UINT64_MAX - 12, UINT64_MAX, UINT64_MAX - 5, depth);
  }
}

struct DcfRangeCalls {
  std::atomic<int> active{0};
  std::atomic<int> max_active{0};
  std::atomic<bool> other_thread{false};
  std::atomic<bool> parallel_team{false};
  const std::thread::id caller = std::this_thread::get_id();
};

class DcfRangeObservedPrg {
public:
  explicit DcfRangeObservedPrg(DcfRangeCalls *calls) : calls_(calls), prg_(kNonce0, kNonce1) {}

  __host__ __device__ cuda::std::array<int4, 4> Gen(int4 seed) {
#ifndef __CUDA_ARCH__
    if (calls_ != nullptr) {
      if (std::this_thread::get_id() != calls_->caller) calls_->other_thread = true;
#ifdef _OPENMP
      if (omp_in_parallel()) calls_->parallel_team = true;
#endif
      const int active = calls_->active.fetch_add(1) + 1;
      int maximum = calls_->max_active.load();
      while (active > maximum && !calls_->max_active.compare_exchange_weak(maximum, active)) {}
    }
#endif
    auto result = prg_.Gen(seed);
#ifndef __CUDA_ARCH__
    if (calls_ != nullptr) --calls_->active;
#endif
    return result;
  }

private:
  DcfRangeCalls *calls_;
  fss::prg::ChaCha<4> prg_;
};

}  // namespace

TEST(DcfRangeTest, LtBytes) { CheckDcfSmallRanges<BytesGroup, fss::DcfPred::kLt>(); }
TEST(DcfRangeTest, GtBytes) { CheckDcfSmallRanges<BytesGroup, fss::DcfPred::kGt>(); }
TEST(DcfRangeTest, LtUint127) { CheckDcfSmallRanges<Uint127Group, fss::DcfPred::kLt>(); }
TEST(DcfRangeTest, GtUint127) { CheckDcfSmallRanges<Uint127Group, fss::DcfPred::kGt>(); }
TEST(DcfRangeTest, LtGenericGroup) { CheckDcfSmallRanges<Uint64Group, fss::DcfPred::kLt>(); }
TEST(DcfRangeTest, GtGenericGroup) { CheckDcfSmallRanges<Uint64Group, fss::DcfPred::kGt>(); }
TEST(DcfRangeTest, WideLtBytes) { CheckDcfWideRanges<BytesGroup, fss::DcfPred::kLt>(); }
TEST(DcfRangeTest, WideGtBytes) { CheckDcfWideRanges<BytesGroup, fss::DcfPred::kGt>(); }
TEST(DcfRangeTest, WideLtUint127) { CheckDcfWideRanges<Uint127Group, fss::DcfPred::kLt>(); }
TEST(DcfRangeTest, WideGtUint127) { CheckDcfWideRanges<Uint127Group, fss::DcfPred::kGt>(); }
TEST(DcfRangeTest, WideLtGenericGroup) { CheckDcfWideRanges<Uint64Group, fss::DcfPred::kLt>(); }
TEST(DcfRangeTest, WideGtGenericGroup) { CheckDcfWideRanges<Uint64Group, fss::DcfPred::kGt>(); }

TEST(DcfRangeTest, ZeroDepthUsesCallerThread) {
  using Scheme = fss::Dcf<8, BytesGroup, DcfRangeObservedPrg>;
  Scheme generator(DcfRangeObservedPrg(nullptr));
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  typename Scheme::Cw corrections[9];
  generator.Gen(corrections, seeds, 107, {604, 981, 273, 84});
  for (int party = 0; party < 2; ++party) {
    DcfRangeCalls calls;
    Scheme scheme{DcfRangeObservedPrg(&calls)};
    std::vector<int4> outputs(256);
    scheme.EvalRange(party, 0, 255, seeds[party], corrections,
        cuda::std::span<int4>(outputs.data(), outputs.size()));
    EXPECT_EQ(calls.active.load(), 0);
    EXPECT_EQ(calls.max_active.load(), 1);
    EXPECT_FALSE(calls.other_thread.load());
    EXPECT_FALSE(calls.parallel_team.load());
  }
}

TEST(DcfRangeTest, RejectsInvalidPreconditions) {
#ifdef NDEBUG
  GTEST_SKIP() << "assertion preconditions are disabled";
#else
  using Scheme = fss::Dcf<3, BytesGroup, fss::prg::ChaCha<4>>;
  Scheme scheme(fss::prg::ChaCha<4>(kNonce0, kNonce1));
  const int4 seed = {13, 27, 45, 66};
  typename Scheme::Cw corrections[4]{};
  int4 output[8];
  auto outputs = cuda::std::span<int4>(output, 8);
  EXPECT_DEBUG_DEATH(scheme.EvalRange(false, 3, 2, seed, corrections, outputs), "first <= last");
  EXPECT_DEBUG_DEATH(scheme.EvalRange(false, 0, 8, seed, corrections, outputs), "last <= domain_last");
  EXPECT_DEBUG_DEATH(scheme.EvalRange(false, 0, 7, seed, corrections,
      cuda::std::span<int4>(output, 7)), "ys.size.*>= count");
  EXPECT_DEBUG_DEATH(scheme.EvalRange(false, 0, 7, seed, corrections, outputs, -1), "par_depth >= 0");
  using WideScheme = fss::Dcf<128, BytesGroup, fss::prg::ChaCha<4>, __uint128_t>;
  WideScheme wide(fss::prg::ChaCha<4>(kNonce0, kNonce1));
  typename WideScheme::Cw wide_corrections[129]{};
  EXPECT_DEBUG_DEATH(wide.EvalRange(false, 0, ~__uint128_t{0}, seed,
      wide_corrections, outputs), "distance <");
#endif
}


TEST(DcfGpuRangeTest, LtClosedIntervalsAndFrontiers) {
  fss::Dcf<8, BytesGroup, fss::prg::ChaCha<4>, uint16_t, fss::DcfPred::kLt> scheme{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t, 2, 4>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 0);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 255, 255);
}

TEST(DcfGpuRangeTest, LtWideDomainAndBoundedSubdivision) {
  fss::Dcf<128, BytesGroup, fss::prg::ChaCha<4>, __uint128_t, fss::DcfPred::kLt> wide{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<128, __uint128_t>(wide, ~__uint128_t{0} - 128, ~__uint128_t{0});
  fss_test::CheckGpuRange<128, __uint128_t, 2, 4>(wide,
      (__uint128_t{1} << 100) - 17, (__uint128_t{1} << 100) + 19);
  fss::Dcf<12, BytesGroup, fss::prg::ChaCha<4>, uint16_t, fss::DcfPred::kLt> split{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<12, uint16_t, 0, 1>(split, 0, 4095);
  fss_test::CheckGpuRange<12, uint16_t>(split, 0, 4095);
}

TEST(DcfGpuRangeTest, GtClosedIntervalsAndFrontiers) {
  fss::Dcf<8, BytesGroup, fss::prg::ChaCha<4>, uint16_t, fss::DcfPred::kGt> scheme{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t, 2, 4>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 0);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 255, 255);
}

TEST(DcfGpuRangeTest, GtWideDomainAndBoundedSubdivision) {
  fss::Dcf<128, BytesGroup, fss::prg::ChaCha<4>, __uint128_t, fss::DcfPred::kGt> wide{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<128, __uint128_t>(wide, ~__uint128_t{0} - 128, ~__uint128_t{0});
  fss_test::CheckGpuRange<128, __uint128_t, 2, 4>(wide,
      (__uint128_t{1} << 100) - 17, (__uint128_t{1} << 100) + 19);
  fss::Dcf<12, BytesGroup, fss::prg::ChaCha<4>, uint16_t, fss::DcfPred::kGt> split{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<12, uint16_t, 0, 1>(split, 0, 4095);
  fss_test::CheckGpuRange<12, uint16_t>(split, 0, 4095);
}

TEST(DcfGpuRangeTest, kLtArithmeticGroupNarrowInputAndCappedFrontier) {
  fss::Dcf<8, fss_test::RangeUint127Group, fss::prg::ChaCha<4>, uint8_t, fss::DcfPred::kLt> scheme{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<8, uint8_t, 999, 1024>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint8_t, 2, 4>(scheme, 13, 201);
  fss::Dcf<1, fss_test::RangeUint127Group, fss::prg::ChaCha<4>, uint8_t, fss::DcfPred::kLt> tiny{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<1, uint8_t, 999, 1024>(tiny, 0, 1);
  fss_test::CheckGpuRange<1, uint8_t>(tiny, 1, 1);
}

TEST(DcfGpuRangeTest, kGtArithmeticGroupNarrowInputAndCappedFrontier) {
  fss::Dcf<8, fss_test::RangeUint127Group, fss::prg::ChaCha<4>, uint8_t, fss::DcfPred::kGt> scheme{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<8, uint8_t, 999, 1024>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint8_t, 2, 4>(scheme, 13, 201);
  fss::Dcf<1, fss_test::RangeUint127Group, fss::prg::ChaCha<4>, uint8_t, fss::DcfPred::kGt> tiny{fss::prg::ChaCha<4>{kNonce0, kNonce1}};
  fss_test::CheckGpuRange<1, uint8_t, 999, 1024>(tiny, 0, 1);
  fss_test::CheckGpuRange<1, uint8_t>(tiny, 1, 1);
}
