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


template <typename Group, fss::DcfPred pred, bool has_half, typename Backend = fss::prg::Aes128Soft<4>>
void CheckDcfGenHalf() {
  constexpr int bits = 5;
  using Prg = fss_test::CountedPrg<Backend, 4, has_half>;
  using FullPrg = fss_test::CountedPrg<Backend, 4, false>;
  using Scheme = fss::Dcf<bits, Group, Prg, uint8_t, pred>;
  using FullScheme = fss::Dcf<bits, Group, FullPrg, uint8_t, pred>;
  fss_test::GenHalfAesFixture<4> aes;
  fss_test::PrgCallCounts calls, full_calls;
  auto backend = [&] {
    if constexpr (std::is_same_v<Backend, fss::prg::Aes128Soft<4>>) return aes.Backend();
    else return Backend(aes.keys);
  }();
  Prg prg(backend, calls);
  FullPrg full_prg(backend, full_calls);
  Scheme scheme{prg};
  FullScheme reference{full_prg};
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
      int4 state = fss::util::SetLsb(seeds[b], b);
      Group value;
      for (int i = 0; i < bits; ++i) {
        bool right = (x >> (bits - 1 - i)) & 1;
        int4 left, right_state;
        Group left_value, right_value;
        fss::detail::SchemeAccess::DcfExpandNode<FullScheme>(full_prg, state, value, b,
            full_keys[b].cws[i], left, left_value, right_state, right_value);
        calls = {};
        fss::detail::SchemeAccess::DcfPathStep<Scheme>(prg, state, value, b,
            keys[b].cws[i], right, state, value);
        int4 expected_state = right ? right_state : left;
        int4 expected_value = (right ? right_value : left_value).Into();
        int4 actual_value = value.Into();
        EXPECT_EQ(memcmp(&state, &expected_state, sizeof(int4)), 0);
        EXPECT_EQ(memcmp(&actual_value, &expected_value, GroupCompareSize<Group>()), 0);
        EXPECT_EQ(calls.full, has_half ? 0 : 1);
        EXPECT_EQ(calls.half, has_half ? 1 : 0);
        EXPECT_EQ(calls.right, has_half ? static_cast<int>(right) : 0);
      }
    }
    int4 result = (Group::From(ys[0]) + Group::From(ys[1])).Into();
    bool hit = pred == fss::DcfPred::kLt ? x < 13 : x > 13;
    int4 expected = hit ? Group::From(beta).Into() : Group{}.Into();
    EXPECT_EQ(memcmp(&result, &expected, GroupCompareSize<Group>()), 0);
  }
}

template <typename Group>
void CheckDcfGenHalfPredicates() {
  CheckDcfGenHalf<Group, fss::DcfPred::kLt, true>();
  CheckDcfGenHalf<Group, fss::DcfPred::kLt, false>();
  CheckDcfGenHalf<Group, fss::DcfPred::kGt, true>();
  CheckDcfGenHalf<Group, fss::DcfPred::kGt, false>();
}

TEST(DcfGenHalf, BytesSelectedAndFallback) { CheckDcfGenHalfPredicates<BytesGroup>(); }
TEST(DcfGenHalf, Uint127SelectedAndFallback) { CheckDcfGenHalfPredicates<Uint127Group>(); }
TEST(DcfGenHalf, GenericGroupSelectedAndFallback) { CheckDcfGenHalfPredicates<Uint64Group>(); }


TEST(DcfGenHalf, AesRawSelectedAndFallback) {
  CheckDcfGenHalf<BytesGroup, fss::DcfPred::kLt, true, fss::prg::Aes128MmoRaw<4>>();
  CheckDcfGenHalf<BytesGroup, fss::DcfPred::kLt, false, fss::prg::Aes128MmoRaw<4>>();
  CheckDcfGenHalf<BytesGroup, fss::DcfPred::kGt, true, fss::prg::Aes128MmoRaw<4>>();
  CheckDcfGenHalf<BytesGroup, fss::DcfPred::kGt, false, fss::prg::Aes128MmoRaw<4>>();
}

TEST(DcfGenHalf, GpuAesSoftSelectedRange) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "no CUDA device available";
  fss_test::GenHalfDeviceAesFixture<4> aes;
  ASSERT_EQ(aes.Allocate(), cudaSuccess);
  fss::Dcf<16, Uint64Group, fss::prg::Aes128Soft<4>, uint16_t> lt{aes.Backend()};
  fss_test::CheckGpuRange<16, uint16_t, 8, 64>(lt, 33, 47);
  fss::Dcf<16, Uint64Group, fss::prg::Aes128Soft<4>, uint16_t, fss::DcfPred::kGt> gt{aes.Backend()};
  fss_test::CheckGpuRange<16, uint16_t, 8, 64>(gt, 33, 47);
}

// -----------------------------------------------------------------------------
// Split-corrections API: bit-exactness against the Cw forms.

namespace {

constexpr int4 kSplitSentinel = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};

// Checks the split-corrections API against the Cw forms for one configuration:
// every correction field, every output of Eval, EvalAll and EvalRange, and the
// two-party reconstruction. stride == 0 selects the default-argument call.
template <int in_bits, typename Group, typename Prg, typename In, fss::DcfPred pred>
void CheckDcfSplit(In alpha, int4 beta, size_t stride) {
  using Scheme = fss::Dcf<in_bits, Group, Prg, In, pred>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  Scheme scheme{Prg(kNonce0, kNonce1)};
  const bool default_stride = stride == 0;
  const size_t level_stride = default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(in_bits) * level_stride + 1;
  constexpr size_t kExtra = 3;
  constexpr size_t kDomain = size_t{1} << in_bits;

  typename Scheme::Cw ref[in_bits + 1];
  scheme.Gen(ref, seeds, alpha, beta);

  std::vector<int4> cws(used + kExtra, kSplitSentinel);
  std::vector<int4> vws(used + kExtra, kSplitSentinel);
  const auto cws_out = cuda::std::span<int4>(cws.data(), cws.size());
  const auto vws_out = cuda::std::span<int4>(vws.data(), vws.size());
  In tr = 0;
  if (default_stride) scheme.Gen(cws_out, vws_out, tr, seed_span, alpha, beta);
  else scheme.Gen(cws_out, vws_out, tr, seed_span, alpha, beta, stride);

  In expected_tr = 0;
  for (int l = 0; l < in_bits; ++l) {
    const size_t index = static_cast<size_t>(l) * level_stride;
    EXPECT_EQ(memcmp(&cws[index], &ref[l].s, sizeof(int4)), 0) << "cws layer " << l;
    const int4 v_without_control = fss::util::SetLsb(ref[l].v, false);
    EXPECT_EQ(memcmp(&vws[index], &v_without_control, sizeof(int4)), 0) << "vws layer " << l;
    if (fss::util::GetLsb(ref[l].v)) expected_tr |= static_cast<In>(1) << l;
  }
  EXPECT_EQ(tr, expected_tr) << "packed right-control bits";
  const int4 zero = {0, 0, 0, 0};
  EXPECT_EQ(memcmp(&cws[used - 1], &zero, sizeof(int4)), 0) << "cws terminal";
  EXPECT_EQ(memcmp(&vws[used - 1], &ref[in_bits].v, sizeof(int4)), 0) << "vws terminal";
  // Only the described prefix is written: gap entries and trailing capacity keep the sentinel.
  for (size_t i = 0; i < cws.size(); ++i) {
    if (i % level_stride == 0 && i / level_stride <= static_cast<size_t>(in_bits)) continue;
    EXPECT_EQ(memcmp(&cws[i], &kSplitSentinel, sizeof(int4)), 0) << "cws untouched at " << i;
    EXPECT_EQ(memcmp(&vws[i], &kSplitSentinel, sizeof(int4)), 0) << "vws untouched at " << i;
  }

  const cuda::std::span<const int4> split_cws(cws.data(), used);
  const cuda::std::span<const int4> split_vws(vws.data(), used);
  std::vector<int4> split_all[2];
  for (int party = 0; party < 2; ++party) {
    const bool b = party != 0;
    const int4 s0 = seeds[party];
    for (size_t x = 0; x < kDomain; ++x) {
      const In xi = static_cast<In>(x);
      const int4 aos = scheme.Eval(b, s0, ref, xi);
      const int4 got = default_stride
          ? scheme.Eval(b, s0, split_cws, split_vws, tr, xi)
          : scheme.Eval(b, s0, split_cws, split_vws, tr, xi, stride);
      EXPECT_EQ(memcmp(&aos, &got, sizeof(int4)), 0) << "Eval party=" << party << " x=" << x;
    }

    std::vector<int4> aos_all(kDomain, kSplitSentinel);
    split_all[party].assign(kDomain + 2, kSplitSentinel);
    scheme.EvalAll(b, s0, ref, aos_all.data());
    if (default_stride) scheme.EvalAll(b, s0, split_cws, split_vws, tr, split_all[party].data());
    else scheme.EvalAll(b, s0, split_cws, split_vws, tr, split_all[party].data(), 0, stride);
    EXPECT_EQ(memcmp(aos_all.data(), split_all[party].data(), kDomain * sizeof(int4)), 0)
        << "EvalAll party=" << party;
    EXPECT_EQ(memcmp(&split_all[party][kDomain], &kSplitSentinel, sizeof(int4)), 0)
        << "EvalAll extra capacity party=" << party;

    const In first = In{0};
    const In last = static_cast<In>(kDomain - 1);
    for (int par_depth : {0, 2}) {
      std::vector<int4> aos_range(kDomain + 2, kSplitSentinel);
      std::vector<int4> split_range(kDomain + 2, kSplitSentinel);
      scheme.EvalRange(b, first, last, s0, ref,
          cuda::std::span<int4>(aos_range.data(), kDomain), par_depth);
      const auto ys = cuda::std::span<int4>(split_range.data(), kDomain);
      if (default_stride) scheme.EvalRange(b, first, last, s0, split_cws, split_vws, tr, ys, par_depth);
      else scheme.EvalRange(b, first, last, s0, split_cws, split_vws, tr, ys, par_depth, stride);
      EXPECT_EQ(memcmp(aos_range.data(), split_range.data(), kDomain * sizeof(int4)), 0)
          << "EvalRange party=" << party << " par_depth=" << par_depth;
      for (size_t i = kDomain; i < split_range.size(); ++i) {
        EXPECT_EQ(memcmp(&split_range[i], &kSplitSentinel, sizeof(int4)), 0)
            << "EvalRange extra capacity at " << i;
      }
    }
  }

  for (size_t x = 0; x < kDomain; ++x) {
    const In xi = static_cast<In>(x);
    const bool hit = pred == fss::DcfPred::kLt ? xi < alpha : xi > alpha;
    const int4 expected = hit ? Group::From(beta).Into() : Group{}.Into();
    const int4 actual = (Group::From(split_all[0][x]) + Group::From(split_all[1][x])).Into();
    EXPECT_EQ(memcmp(&actual, &expected, GroupCompareSize<Group>()), 0) << "reconstruction x=" << x;
  }
}

template <int in_bits, typename Group, typename In, fss::DcfPred pred>
void CheckDcfSplitStrides(In alpha) {
  const int4 beta = {604, 981, 273, 84};
  for (size_t stride : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
    CheckDcfSplit<in_bits, Group, fss::prg::ChaCha<4>, In, pred>(alpha, beta, stride);
  }
}

// A Cw[] array is already an int4 array at stride 2, so it feeds the split API
// once the packed right-control bits are materialized (R9). The v view still
// carries the right-control bits in its layer LSBs, which the consumer ignores.
template <int in_bits, typename Group, typename In, fss::DcfPred pred>
void CheckDcfSplitFromCwStorage(In alpha, int4 beta) {
  using Scheme = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, In, pred>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  Scheme scheme{fss::prg::ChaCha<4>(kNonce0, kNonce1)};
  typename Scheme::Cw words[in_bits + 1];
  scheme.Gen(words, seeds, alpha, beta);

  std::vector<int4> storage(2 * (in_bits + 1));
  for (int l = 0; l <= in_bits; ++l) {
    storage[2 * l] = words[l].s;
    storage[2 * l + 1] = words[l].v;
  }
  In tr = 0;
  for (int l = 0; l < in_bits; ++l) {
    if (fss::util::GetLsb(words[l].v)) tr |= static_cast<In>(1) << l;
  }
  const auto cws = cuda::std::span<const int4>(storage.data(), storage.size());
  const auto vws = cuda::std::span<const int4>(storage.data() + 1, storage.size() - 1);
  constexpr size_t kDomain = size_t{1} << in_bits;
  for (int party = 0; party < 2; ++party) {
    for (size_t x = 0; x < kDomain; ++x) {
      const In xi = static_cast<In>(x);
      const int4 expected = scheme.Eval(party, seeds[party], words, xi);
      const int4 actual = scheme.Eval(party, seeds[party], cws, vws, tr, xi, 2);
      EXPECT_EQ(memcmp(&actual, &expected, sizeof(int4)), 0) << "party=" << party << " x=" << x;
    }
    std::vector<int4> expected_all(kDomain, kSplitSentinel);
    std::vector<int4> actual_all(kDomain, kSplitSentinel);
    scheme.EvalAll(party, seeds[party], words, expected_all.data());
    scheme.EvalAll(party, seeds[party], cws, vws, tr, actual_all.data(), 0, 2);
    EXPECT_EQ(memcmp(actual_all.data(), expected_all.data(), kDomain * sizeof(int4)), 0)
        << "EvalAll party=" << party;
    std::vector<int4> expected_range(kDomain, kSplitSentinel);
    std::vector<int4> actual_range(kDomain, kSplitSentinel);
    scheme.EvalRange(party, In{0}, static_cast<In>(kDomain - 1), seeds[party], words,
        cuda::std::span<int4>(expected_range.data(), kDomain));
    scheme.EvalRange(party, In{0}, static_cast<In>(kDomain - 1), seeds[party], cws, vws, tr,
        cuda::std::span<int4>(actual_range.data(), kDomain), 0, 2);
    EXPECT_EQ(memcmp(actual_range.data(), expected_range.data(), kDomain * sizeof(int4)), 0)
        << "EvalRange party=" << party;
  }
}

}  // namespace

// in_bits 1 exercises the single-level tree, 3 keeps the recursive EvalTree()
// walk, and 6 and 8 run the batched EvalAll() frontier.
TEST(DcfSplitCorrectionsTest, Depth1Lt) { CheckDcfSplitStrides<1, BytesGroup, uint8_t, fss::DcfPred::kLt>(1); }
TEST(DcfSplitCorrectionsTest, Depth1Gt) { CheckDcfSplitStrides<1, BytesGroup, uint8_t, fss::DcfPred::kGt>(0); }
TEST(DcfSplitCorrectionsTest, Depth3Uint127Lt) { CheckDcfSplitStrides<3, Uint127Group, uint8_t, fss::DcfPred::kLt>(5); }
TEST(DcfSplitCorrectionsTest, Depth3Uint127Gt) { CheckDcfSplitStrides<3, Uint127Group, uint8_t, fss::DcfPred::kGt>(5); }
TEST(DcfSplitCorrectionsTest, Depth6Lt) { CheckDcfSplitStrides<6, BytesGroup, uint8_t, fss::DcfPred::kLt>(11); }
TEST(DcfSplitCorrectionsTest, Depth8Lt) { CheckDcfSplitStrides<8, BytesGroup, uint8_t, fss::DcfPred::kLt>(kAlpha); }
TEST(DcfSplitCorrectionsTest, Depth8Gt) { CheckDcfSplitStrides<8, BytesGroup, uint8_t, fss::DcfPred::kGt>(kAlpha); }
TEST(DcfSplitCorrectionsTest, Depth8GenericGroup) { CheckDcfSplitStrides<8, Uint64Group, uint8_t, fss::DcfPred::kLt>(kAlpha); }
TEST(DcfSplitCorrectionsTest, CwStorageViewLt) {
  CheckDcfSplitFromCwStorage<8, BytesGroup, uint8_t, fss::DcfPred::kLt>(kAlpha, {604, 981, 273, 84});
}
TEST(DcfSplitCorrectionsTest, CwStorageViewGt) {
  CheckDcfSplitFromCwStorage<3, Uint127Group, uint8_t, fss::DcfPred::kGt>(5, {604, 981, 273, 84});
}

#ifndef NDEBUG
TEST(DcfSplitCorrectionsTest, RejectsInvalidInputs) {
  using Scheme = fss::Dcf<3, BytesGroup, fss::prg::ChaCha<4>, uint8_t>;
  Scheme scheme{fss::prg::ChaCha<4>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  const int4 beta = {604, 981, 273, 84};
  int4 cws[4] = {};
  int4 vws[4] = {};
  int4 ys[8] = {};
  uint8_t tr = 0;
  const auto cws_out = cuda::std::span<int4>(cws, 4);
  const auto vws_out = cuda::std::span<int4>(vws, 4);
  const auto cws_in = cuda::std::span<const int4>(cws, 4);
  const auto vws_in = cuda::std::span<const int4>(vws, 4);
  const int4 s0 = seeds[0];

  // A zero stride is only the batch sentinel, so a non-batch method rejects it.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cws_out, vws_out, tr, seed_span, uint8_t{5}, beta, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, vws_in, uint8_t{0}, uint8_t{0}, 0),
      "Assertion.*failed");
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cuda::std::span<int4>(cws, 3), vws_out, tr, seed_span,
                        uint8_t{5}, beta),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(cws, 3), vws_in,
                        uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cws_in, cuda::std::span<const int4>(vws, 3),
                        uint8_t{0}, ys),
      "Assertion.*failed");
  // The terminal value component must have a zero LSB.
  int4 odd_vws[4] = {};
  odd_vws[3].w = 1;
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in,
                        cuda::std::span<const int4>(odd_vws, 4), uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  // A stride past the addressable guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / 3 + 1;
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, vws_in, uint8_t{0}, uint8_t{0},
                        overflowing_stride),
      "Assertion.*failed");
}
#endif

// The batch geometry validation is host code that runs before any CUDA call, so
// these checks need no device and read no buffer.
TEST(DcfSplitGpuTest, RejectsInvalidBatchGeometry) {
  using Wide = fss::Dcf<32, BytesGroup, fss::prg::ChaCha<4>, uint64_t>;
  Wide wide{fss::prg::ChaCha<4>(kNonce0, kNonce1)};
  int4 seed_storage[1] = {};
  int4 cws_storage[1] = {};
  int4 vws_storage[1] = {};
  int4 ys_storage[1] = {};
  uint64_t tr_storage[1] = {};

  // 2^32 keys of 2^32 outputs make the output product wrap in size_t, so the
  // guard must reject the call before the product is formed. The correction
  // spans are sized nominally here and the key count is above the launch grid
  // limit for every launch shape, so a variant without the output guard would
  // stop at cudaErrorInvalidConfiguration instead of launching into the
  // one-element output buffer.
  constexpr size_t kKeys = size_t{1} << 32;
  constexpr size_t kMinCorrections = 32 * kKeys + kKeys;
  EXPECT_EQ((wide.template EvalAllGpuBatch<24, 256>(false,
      cuda::std::span<const int4>(seed_storage, kKeys),
      cuda::std::span<const int4>(cws_storage, kMinCorrections),
      cuda::std::span<const int4>(vws_storage, kMinCorrections),
      cuda::std::span<const uint64_t>(tr_storage, kKeys),
      cuda::std::span<int4>(ys_storage, 1), 0)), cudaErrorInvalidValue);

  using Small = fss::Dcf<8, BytesGroup, fss::prg::ChaCha<4>, uint16_t>;
  Small small{fss::prg::ChaCha<4>(kNonce0, kNonce1)};
  constexpr size_t kSmallKeys = 2;
  constexpr size_t kSmallRequired = 8 * kSmallKeys + kSmallKeys;
  int4 small_s0s[kSmallKeys] = {};
  int4 small_cws[kSmallRequired] = {};
  int4 small_vws[kSmallRequired] = {};
  int4 small_ys[kSmallKeys * 256] = {};
  uint16_t small_tr[kSmallKeys] = {};
  const auto s0s = cuda::std::span<const int4>(small_s0s, kSmallKeys);
  const auto cws = cuda::std::span<const int4>(small_cws, kSmallRequired);
  const auto vws = cuda::std::span<const int4>(small_vws, kSmallRequired);
  const auto tr = cuda::std::span<const uint16_t>(small_tr, kSmallKeys);
  const auto ys = cuda::std::span<int4>(small_ys, kSmallKeys * 256);

  // A stride below the key count is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s, cws, vws, tr, ys, 0, 1)),
      cudaErrorInvalidValue);
  // A stride past the element guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s, cws, vws, tr, ys, 0,
                        kMaxElements + 1)),
      cudaErrorInvalidValue);
  // A correction span that omits the terminal column is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s,
                        cuda::std::span<const int4>(small_cws, kSmallRequired - 1), vws, tr, ys)),
      cudaErrorInvalidValue);
  // An empty batch succeeds without reading any buffer.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, cuda::std::span<const int4>(),
                        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
                        cuda::std::span<const uint16_t>(), cuda::std::span<int4>())),
      cudaSuccess);
}

// The GPU split launches are compile-verified here and executed only on a
// machine with a free device, comparing against the host Cw-form outputs. The
// wide Group instantiation also covers the split host stubs.
namespace {

template <typename Group>
void CheckSplitGpuLaunches() {
  int devices = 0;
  cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
      (device_status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(device_status, cudaSuccess);

  constexpr int kBits = 8;
  constexpr size_t kDomain = size_t{1} << kBits;
  using Scheme = fss::Dcf<kBits, Group, fss::prg::ChaCha<4>, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<4>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 beta = {7, 19, 23, 42};
  const uint16_t alpha = 107;
  constexpr size_t kStride = 2;

  typename Scheme::Cw ref[kBits + 1];
  scheme.Gen(ref, seeds, alpha, beta);
  std::vector<int4> cws((kBits + 1) * kStride + 2, kSplitSentinel);
  std::vector<int4> vws((kBits + 1) * kStride + 2, kSplitSentinel);
  uint16_t tr = 0;
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()),
      cuda::std::span<int4>(vws.data(), vws.size()), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha, beta, kStride);

  int4 *device_cws = nullptr;
  int4 *device_vws = nullptr;
  uint16_t *device_tr = nullptr;
  int4 *device_ys = nullptr;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  constexpr size_t kKeys = 2;
  ASSERT_EQ(cudaMalloc(&device_cws, cws.size() * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_vws, vws.size() * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tr, kKeys * sizeof(uint16_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, kDomain * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_cws, cws.data(), cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_vws, vws.data(), vws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_tr, &tr, sizeof(uint16_t), cudaMemcpyHostToDevice, stream),
      cudaSuccess);

  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(party);
    const bool b = party != 0;
    const auto span_cws = cuda::std::span<const int4>(device_cws, cws.size());
    const auto span_vws = cuda::std::span<const int4>(device_vws, vws.size());
    std::vector<int4> actual(kDomain, kSplitSentinel);
    ASSERT_EQ((scheme.template EvalAllGpu<8, 16>(b, seeds[party], span_cws, span_vws, tr,
        cuda::std::span<int4>(device_ys, kDomain), stream)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_ys, kDomain * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t x = 0; x < kDomain; ++x) {
      const int4 expected = scheme.Eval(b, seeds[party], ref, static_cast<uint16_t>(x));
      EXPECT_EQ(memcmp(&actual[x], &expected, sizeof(int4)), 0) << "x=" << x;
    }
    // EvalRangeGpu over a partial interval leaves the extra device capacity untouched.
    std::vector<int4> filled(kDomain, kSplitSentinel);
    ASSERT_EQ(cudaMemcpyAsync(device_ys, filled.data(), kDomain * sizeof(int4),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ((scheme.template EvalRangeGpu<8, 16>(b, uint16_t{13}, uint16_t{201}, seeds[party],
        span_cws, span_vws, tr, cuda::std::span<int4>(device_ys, kDomain), stream, kStride)),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(filled.data(), device_ys, kDomain * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (uint16_t x = 13; x <= 201; ++x) {
      const int4 expected = scheme.Eval(b, seeds[party], ref, x);
      EXPECT_EQ(memcmp(&filled[x - 13], &expected, sizeof(int4)), 0) << "range x=" << x;
    }
    for (size_t i = 201 - 13 + 1; i < kDomain; ++i) {
      EXPECT_EQ(memcmp(&filled[i], &kSplitSentinel, sizeof(int4)), 0) << "untouched at " << i;
    }
  }

  // A zero stride and a span that omits the terminal slot are rejected.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, seeds[0],
      cuda::std::span<const int4>(device_cws, cws.size()), cuda::std::span<const int4>(device_vws, vws.size()),
      tr, cuda::std::span<int4>(device_ys, kDomain), stream, 0)), cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{255}, seeds[0],
      cuda::std::span<const int4>(device_cws, kBits), cuda::std::span<const int4>(device_vws, vws.size()),
      tr, cuda::std::span<int4>(device_ys, kDomain), stream)), cudaErrorInvalidValue);

  // Batch launches: the default stride is the dense level-major key count.
  typename Scheme::Cw ref_batch[kKeys][kBits + 1];
  std::vector<int4> batch_cws((kBits + 1) * kKeys, kSplitSentinel);
  std::vector<int4> batch_vws((kBits + 1) * kKeys, kSplitSentinel);
  int4 batch_s0s[kKeys];
  uint16_t batch_tr[kKeys] = {0, 0};
  for (size_t key = 0; key < kKeys; ++key) {
    batch_s0s[key] = seeds[key];
    scheme.Gen(ref_batch[key], seeds, static_cast<uint16_t>(alpha + key), beta);
    uint16_t key_tr = 0;
    // Offsetting by the key index fills the dense level-major layout.
    scheme.Gen(cuda::std::span<int4>(batch_cws.data() + key, batch_cws.size() - key),
        cuda::std::span<int4>(batch_vws.data() + key, batch_vws.size() - key), key_tr,
        cuda::std::span<const int4, 2>(seeds, 2), static_cast<uint16_t>(alpha + key), beta, kKeys);
    batch_tr[key] = key_tr;
  }
  ASSERT_EQ(cudaMemcpyAsync(device_cws, batch_cws.data(), batch_cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_vws, batch_vws.data(), batch_vws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_tr, batch_tr, sizeof(batch_tr), cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  int4 *device_s0s = nullptr;
  ASSERT_EQ(cudaMalloc(&device_s0s, sizeof(batch_s0s)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_s0s, batch_s0s, sizeof(batch_s0s), cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  int4 *device_batch_ys = nullptr;
  ASSERT_EQ(cudaMalloc(&device_batch_ys, kKeys * kDomain * sizeof(int4)), cudaSuccess);
  const auto span_batch_cws = cuda::std::span<const int4>(device_cws, batch_cws.size());
  const auto span_batch_vws = cuda::std::span<const int4>(device_vws, batch_vws.size());
  const auto span_batch_tr = cuda::std::span<const uint16_t>(device_tr, kKeys);
  const auto span_batch_ys = cuda::std::span<int4>(device_batch_ys, kKeys * kDomain);
  for (int use_sentinel = 0; use_sentinel < 2; ++use_sentinel) {
    SCOPED_TRACE(use_sentinel);
    const size_t stride = use_sentinel == 0 ? fss::kDenseStride : kKeys;
    ASSERT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false, cuda::std::span<const int4>(device_s0s, kKeys),
        span_batch_cws, span_batch_vws, span_batch_tr, span_batch_ys, stream, stride)), cudaSuccess);
    std::vector<int4> actual(kKeys * kDomain, kSplitSentinel);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_batch_ys, actual.size() * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t key = 0; key < kKeys; ++key) {
      for (size_t x = 0; x < kDomain; ++x) {
        const int4 expected = scheme.Eval(false, batch_s0s[key], ref_batch[key],
            static_cast<uint16_t>(x));
        EXPECT_EQ(memcmp(&actual[key * kDomain + x], &expected, sizeof(int4)), 0)
            << "key=" << key << " x=" << x;
      }
    }
  }
  // A stride below the key count is rejected.
  EXPECT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false, cuda::std::span<const int4>(device_s0s, kKeys),
      span_batch_cws, span_batch_vws, span_batch_tr, span_batch_ys, stream, 1)), cudaErrorInvalidValue);

  EXPECT_EQ(cudaFree(device_batch_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_s0s), cudaSuccess);
  EXPECT_EQ(cudaFree(device_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_tr), cudaSuccess);
  EXPECT_EQ(cudaFree(device_vws), cudaSuccess);
  EXPECT_EQ(cudaFree(device_cws), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace

TEST(DcfSplitGpuTest, SplitLaunchesMatchCwHost) { CheckSplitGpuLaunches<BytesGroup>(); }
TEST(DcfSplitGpuTest, WideGroupSplitLaunchesCompile) { CheckSplitGpuLaunches<Uint127Group>(); }
