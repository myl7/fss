#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/half_tree_dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <fss/prg/aes128_mmo_raw.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include "core_api_test_helpers.cuh"
#include "range_gpu_test_helpers.cuh"
#include <atomic>
#include <limits>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
using Uint64Group = fss::group::Uint<uint64_t>;

template <typename Group>
constexpr size_t GroupCompareSize() {
  if constexpr (std::is_same_v<Group, Uint64Group>) {
    return 8;
  } else if constexpr (std::is_same_v<Group, Uint127Group>) {
    return 16;
  } else {
    return sizeof(int4);
  }
}

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;
constexpr __uint128_t kBeta = 604;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// Owned-table AES-CCR test fixture
template <typename Group>
class HalfTreeDpfOwnedAesCcrTest : public ::testing::Test {
protected:
  using Prg = fss_test::TestCcrHash;
  using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, fss_test::CcrBackend<Prg>, uint16_t>;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;
  int4 hash_key;

  HalfTreeDpfOwnedAesCcrTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  int4 MakeBBuf() {
    return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  }

  void TestEvalAtAlpha() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
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

  void TestEvalAtNonAlpha() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
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

      EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }
  }

  void TestEvalAll() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
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
};

using HalfTreeDpfBytesOwnedAesCcrTest = HalfTreeDpfOwnedAesCcrTest<BytesGroup>;
using HalfTreeDpfUint128OwnedAesCcrTest = HalfTreeDpfOwnedAesCcrTest<Uint127Group>;
using HalfTreeDpfUint64OwnedAesCcrTest = HalfTreeDpfOwnedAesCcrTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesOwnedAesCcrTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesOwnedAesCcrTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesOwnedAesCcrTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint128OwnedAesCcrTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128OwnedAesCcrTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128OwnedAesCcrTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint64OwnedAesCcrTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64OwnedAesCcrTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64OwnedAesCcrTest, EvalAll) {
  TestEvalAll();
}

// Aes128Soft-based test fixture
template <typename Group>
class HalfTreeDpfAesSoftTest : public ::testing::Test {
protected:
  using Prg = fss::prg::Aes128Soft<1>;
  using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, fss_test::CcrBackend<Prg>, uint16_t>;

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

  int4 MakeBBuf() {
    return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  }

  void TestEvalAtAlpha() {
    uint32_t te0[256];
    uint8_t sbox[256];
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    Prg prg(aes_key, te0, sbox);
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    int4 y0 = fss_test::EvalKey(dpf, false, k0, kAlpha);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, kAlpha);

    auto result = Group::From(y0) + Group::From(y1);
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAtNonAlpha() {
    uint32_t te0[256];
    uint8_t sbox[256];
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    Prg prg(aes_key, te0, sbox);
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 100; i++) {
      uint16_t x = dis(gen);
      if (x == kAlpha) continue;

      int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
      int4 y1 = fss_test::EvalKey(dpf, true, k1, x);

      auto result = Group::From(y0) + Group::From(y1);
      int4 result_buf = result.Into();
      int4 zero = {0, 0, 0, 0};
      EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }
  }

  void TestEvalAll() {
    uint32_t te0[256];
    uint8_t sbox[256];
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    Prg prg(aes_key, te0, sbox);
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
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
};

using HalfTreeDpfBytesAesSoftTest = HalfTreeDpfAesSoftTest<BytesGroup>;
using HalfTreeDpfUint128AesSoftTest = HalfTreeDpfAesSoftTest<Uint127Group>;
using HalfTreeDpfUint64AesSoftTest = HalfTreeDpfAesSoftTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesAesSoftTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesAesSoftTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesAesSoftTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint128AesSoftTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128AesSoftTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128AesSoftTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint64AesSoftTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64AesSoftTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64AesSoftTest, EvalAll) {
  TestEvalAll();
}

// Edge case: in_bits = 1
template <typename Group>
class HalfTreeDpfOneBitTest : public ::testing::Test {
protected:
  using Prg = fss_test::TestCcrHash;
  using DpfType = fss::HalfTreeDpf<1, Group, fss_test::CcrBackend<Prg>, uint8_t>;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;
  int4 hash_key;

  HalfTreeDpfOneBitTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  int4 MakeBBuf() {
    return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  }

  void TestEvalAtAlpha() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    uint8_t alpha = 1;

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    int4 y0 = fss_test::EvalKey(dpf, false, k0, alpha);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, alpha);

    auto result = Group::From(y0) + Group::From(y1);
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAtNonAlpha() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    uint8_t alpha = 1;

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    uint8_t x = 0;
    int4 y0 = fss_test::EvalKey(dpf, false, k0, x);
    int4 y1 = fss_test::EvalKey(dpf, true, k1, x);

    auto result = Group::From(y0) + Group::From(y1);
    int4 result_buf = result.Into();
    int4 zero = {0, 0, 0, 0};
    EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAll() {
    DpfType dpf{fss_test::Ccr(prg), hash_key};
    uint8_t alpha = 1;

    int4 b_buf = MakeBBuf();
    fss_test::GenPair(dpf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    int4 ys0[2], ys1[2];
    fss_test::EvalAllKey(dpf, false, k0, cuda::std::span<int4>(ys0, 2));
    fss_test::EvalAllKey(dpf, true, k1, cuda::std::span<int4>(ys1, 2));

    auto expected = Group::From(b_buf);
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (int x = 0; x < 2; ++x) {
      auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
      int4 result_buf = result.Into();

      if (x == alpha) {
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0) << "Failed at alpha=" << x;
      } else {
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
      }
    }
  }
};

using HalfTreeDpfBytesOneBitTest = HalfTreeDpfOneBitTest<BytesGroup>;
using HalfTreeDpfUint128OneBitTest = HalfTreeDpfOneBitTest<Uint127Group>;
using HalfTreeDpfUint64OneBitTest = HalfTreeDpfOneBitTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesOneBitTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesOneBitTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesOneBitTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint128OneBitTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128OneBitTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128OneBitTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint64OneBitTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64OneBitTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64OneBitTest, EvalAll) {
  TestEvalAll();
}

// These tests exercise the 8-wide batched EvalAll() frontier at its in_bits
// and par_depth boundaries with the PRG whose Gen8() the batch pipelines
// through. in_bits below kBatchPrefix + 1 (4) falls back to the per-input
// point walk, in_bits = 4 is the smallest batched domain and clamps the split
// depth to 0, and larger in_bits run both the sequential batched walk
// (par_depth = 0) and the frontier-width split (par_depth > 0).
template <int in_bits, int par_depth = 0>
class HalfTreeDpfAesRawTestBase : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128MmoRaw<1>;
  using DpfType = fss::HalfTreeDpf<in_bits, Group, fss_test::CcrBackend<Prg>, uint16_t>;

  static constexpr uint8_t kKeys[1][16] = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
  };

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;
  int4 hash_key;

  HalfTreeDpfAesRawTestBase() : prg(kKeys) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }

  int4 MakeBBuf() {
    return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  }

  // Full-domain share sum: beta at alpha, identity elsewhere.
  void TestEvalAll(uint16_t alpha) {
    DpfType dpf{fss_test::Ccr(prg), hash_key};

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
    DpfType dpf{fss_test::Ccr(prg), hash_key};

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

class HalfTreeDpfBytesAesRaw2BitsTest : public HalfTreeDpfAesRawTestBase<2> {};
class HalfTreeDpfBytesAesRaw3BitsTest : public HalfTreeDpfAesRawTestBase<3> {};
class HalfTreeDpfBytesAesRaw4BitsSeqTest : public HalfTreeDpfAesRawTestBase<4, 0> {};
class HalfTreeDpfBytesAesRaw4BitsClampedSplitTest : public HalfTreeDpfAesRawTestBase<4, 2> {};
class HalfTreeDpfBytesAesRaw6BitsSeqTest : public HalfTreeDpfAesRawTestBase<6, 0> {};
class HalfTreeDpfBytesAesRaw6BitsSplitTest : public HalfTreeDpfAesRawTestBase<6, 2> {};
class HalfTreeDpfBytesAesRaw16BitsDefaultTest : public HalfTreeDpfAesRawTestBase<16> {};

TEST_F(HalfTreeDpfBytesAesRaw2BitsTest, EvalAll) {
  constexpr uint16_t kAlpha2 = 2;
  TestEvalAll(kAlpha2);
  TestEvalAllMatchesEval(kAlpha2);
}
TEST_F(HalfTreeDpfBytesAesRaw3BitsTest, EvalAll) {
  constexpr uint16_t kAlpha3 = 5;
  TestEvalAll(kAlpha3);
  TestEvalAllMatchesEval(kAlpha3);
}
TEST_F(HalfTreeDpfBytesAesRaw4BitsSeqTest, EvalAll) {
  constexpr uint16_t kAlpha4 = 11;
  TestEvalAll(kAlpha4);
  TestEvalAllMatchesEval(kAlpha4);
}
TEST_F(HalfTreeDpfBytesAesRaw4BitsClampedSplitTest, EvalAll) {
  constexpr uint16_t kAlpha4 = 11;
  TestEvalAll(kAlpha4);
  TestEvalAllMatchesEval(kAlpha4);
}
TEST_F(HalfTreeDpfBytesAesRaw6BitsSeqTest, EvalAll) {
  constexpr uint16_t kAlpha6 = 43;
  TestEvalAll(kAlpha6);
  TestEvalAllMatchesEval(kAlpha6);
}
TEST_F(HalfTreeDpfBytesAesRaw6BitsSplitTest, EvalAll) {
  constexpr uint16_t kAlpha6 = 43;
  TestEvalAll(kAlpha6);
  TestEvalAllMatchesEval(kAlpha6);
}
TEST_F(HalfTreeDpfBytesAesRaw16BitsDefaultTest, EvalAll) {
  TestEvalAll(kAlpha);
  TestEvalAllMatchesEval(kAlpha);
}

// Alpha at the domain edges: 0 and 2 ** in_bits - 1.
TEST_F(HalfTreeDpfBytesAesRaw6BitsSeqTest, EvalAllEdgeAlpha) {
  TestEvalAll(0);
  TestEvalAllMatchesEval(0);
  TestEvalAll((1U << 6) - 1);
  TestEvalAllMatchesEval((1U << 6) - 1);
}

// Aes128Mmo has no Gen8(), so EvalAll() runs the frontier through the
// detail::GenBatch() loop; this pins the OpenSSL PRG cell of the EvalAll matrix.
class HalfTreeDpfBytesAesTest : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128Mmo<1>;
  using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, fss_test::CcrBackend<Prg>, uint16_t>;

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

  int4 MakeBBuf() {
    return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
  }

  void TestEvalAll() {
    Prg prg(ctxs);
    DpfType dpf{fss_test::Ccr(prg), hash_key};

    int4 b_buf = MakeBBuf();
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

TEST_F(HalfTreeDpfBytesAesTest, EvalAll) {
  TestEvalAll();
}

namespace {
void ExpectBlock(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

TEST(HalfTreeOutput, FinalCwExtraFlagDoesNotChangeEitherParty) {
  using Scheme = fss::HalfTreeDpf<1, BytesGroup, fss_test::TestCcrHash, uint8_t>;
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const int4 seeds[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const int4 beta = {13, 17, 19, 22};
  Scheme::Cw cws[2];
  scheme.Gen(cws, seeds, uint8_t{1}, beta);
  EXPECT_FALSE(cws[1].extra);
  for (uint8_t x = 0; x < 2; ++x) {
    cws[1].extra = false;
    int4 y0 = scheme.Eval(false, seeds[0], cws, x);
    int4 y1 = scheme.Eval(true, seeds[1], cws, x);
    cws[1].extra = true;
    ExpectBlock(scheme.Eval(false, seeds[0], cws, x), y0);
    ExpectBlock(scheme.Eval(true, seeds[1], cws, x), y1);
    ExpectBlock(fss::util::Xor(y0, y1), x == 1 ? beta : int4{});
  }
}
}  // namespace

namespace {
struct HalfTreeRangeHash {
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

template <int bits, typename In, typename Group = BytesGroup>
void CheckHalfTreeRange(In first, In last, In alpha, int depth) {
  std::atomic<size_t> calls{0};
  std::atomic<bool> parallel{false};
  using Scheme = fss::HalfTreeDpf<bits, Group, HalfTreeRangeHash, In>;
  Scheme scheme{HalfTreeRangeHash{fss_test::TestCcrHash{}, &calls, &parallel}, {31, 47, 59, 83}};
  const int4 seeds[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const int4 beta = {13, 17, 19, 22};
  typename Scheme::Cw cws[bits + 1];
  scheme.Gen(cws, seeds, alpha, beta);
  const size_t count = static_cast<size_t>(last - first) + 1;
  const int4 sentinel = {101, 103, 107, 109};
  std::vector<int4> outputs[2] = {
      std::vector<int4>(count + 2, sentinel), std::vector<int4>(count + 2, sentinel)};
  for (int party = 0; party < 2; ++party) {
    calls = 0;
    parallel = false;
    scheme.EvalRange(party, first, last, seeds[party], cws,
        cuda::std::span<int4>(outputs[party].data() + 1, count + 1), depth);
    if (depth == 0) EXPECT_FALSE(parallel.load());
#ifdef _OPENMP
    if (depth > 0 && count > 1 && omp_get_max_threads() > 1) EXPECT_TRUE(parallel.load());
#endif
    if (depth == 0 && count > 2) EXPECT_LT(calls.load(), count * (bits + 1));
    ExpectBlock(outputs[party].front(), sentinel);
    ExpectBlock(outputs[party].back(), sentinel);
    for (size_t j = 0; j < count; ++j) {
      ExpectBlock(outputs[party][j + 1], scheme.Eval(party, seeds[party], cws, static_cast<In>(first + j)));
    }
  }
  for (size_t j = 0; j < count; ++j) {
    const int4 actual = (Group::From(outputs[0][j + 1]) + Group::From(outputs[1][j + 1])).Into();
    const int4 expected = first + j == alpha ? Group::From(beta).Into() : int4{};
    ExpectBlock(actual, expected);
  }
}

TEST(HalfTreeRange, EverySmallClosedInterval) {
  for (uint8_t first = 0; first < 32; ++first) {
    for (uint8_t last = first; last < 32; ++last) {
      for (int depth : {0, 2, 1000}) CheckHalfTreeRange<5>(first, last, uint8_t{17}, depth);
    }
  }
  for (int depth : {0, 3}) {
    CheckHalfTreeRange<1>(uint8_t{0}, uint8_t{1}, uint8_t{1}, depth);
    CheckHalfTreeRange<5, uint8_t, Uint127Group>(uint8_t{3}, uint8_t{24}, uint8_t{17}, depth);
    CheckHalfTreeRange<5, uint8_t, Uint64Group>(uint8_t{3}, uint8_t{24}, uint8_t{17}, depth);
  }
}

TEST(HalfTreeRange, WideCoordinatesAndLastInput) {
  using In = __uint128_t;
  const In high = (In{1} << 100) + 3;
  const In last = ~In{0};
  for (int depth : {0, 3, 1000}) {
    CheckHalfTreeRange<65>((In{1} << 64) - 3, (In{1} << 64) + 5, In{1} << 64, depth);
    CheckHalfTreeRange<128>(high, high + 12, high + 5, depth);
    CheckHalfTreeRange<128>(last - 12, last, last, depth);
    CheckHalfTreeRange<128>(last, last, last, depth);
  }
}

#ifndef NDEBUG
TEST(HalfTreeRange, RejectsInvalidPreconditions) {
  using Scheme = fss::HalfTreeDpf<5, BytesGroup, fss_test::TestCcrHash, uint8_t>;
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  Scheme::Cw cws[6]{};
  int4 seed{};
  int4 outputs[4]{};
  auto ys = cuda::std::span<int4>(outputs, 4);
  EXPECT_DEATH(scheme.EvalRange(false, 0, 0, seed, cws, ys, -1), "par_depth");
  EXPECT_DEATH(scheme.EvalRange(false, 2, 1, seed, cws, ys), "first");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 32, seed, cws, ys), "last");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 4, seed, cws, ys), "ys.size");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 0, seed, nullptr, ys), "cws");
  EXPECT_DEATH(scheme.EvalRange(false, 0, 0, seed, cws,
      cuda::std::span<int4>(static_cast<int4 *>(nullptr), size_t{1})), "ys.data");
  using Wide = fss::HalfTreeDpf<128, BytesGroup, fss_test::TestCcrHash, __uint128_t>;
  Wide wide{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  Wide::Cw wide_cws[129]{};
  EXPECT_DEATH(wide.EvalRange(false, 0, ~__uint128_t{0}, seed, wide_cws, ys), "distance");
}
#endif  // NDEBUG
}  // namespace


TEST(HalfTreeDpfGpuRangeTest, ClosedIntervalsAndFrontiers) {
  fss::HalfTreeDpf<8, BytesGroup, fss_test::TestCcrHash, uint16_t> scheme{fss_test::TestCcrHash{}, int4{11, 22, 33, 44}};
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t, 2, 4>(scheme, 13, 201);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 0, 0);
  fss_test::CheckGpuRange<8, uint16_t>(scheme, 255, 255);
}

TEST(HalfTreeDpfGpuRangeTest, WideDomainAndBoundedSubdivision) {
  fss::HalfTreeDpf<128, BytesGroup, fss_test::TestCcrHash, __uint128_t> wide{fss_test::TestCcrHash{}, int4{11, 22, 33, 44}};
  fss_test::CheckGpuRange<128, __uint128_t>(wide, ~__uint128_t{0} - 128, ~__uint128_t{0});
  fss_test::CheckGpuRange<128, __uint128_t, 2, 4>(wide,
      (__uint128_t{1} << 100) - 17, (__uint128_t{1} << 100) + 19);
  fss::HalfTreeDpf<12, BytesGroup, fss_test::TestCcrHash, uint16_t> split{fss_test::TestCcrHash{}, int4{11, 22, 33, 44}};
  fss_test::CheckGpuRange<12, uint16_t, 0, 1>(split, 0, 4095);
  fss_test::CheckGpuRange<12, uint16_t>(split, 0, 4095);
}

TEST(HalfTreeDpfGpuRangeTest, ArithmeticGroupNarrowInputAndCappedFrontier) {
  fss::HalfTreeDpf<8, fss_test::RangeUint127Group, fss_test::TestCcrHash, uint8_t> scheme{fss_test::TestCcrHash{}, int4{11, 22, 33, 44}};
  fss_test::CheckGpuRange<8, uint8_t, 999, 1024>(scheme, 0, 255);
  fss_test::CheckGpuRange<8, uint8_t, 2, 4>(scheme, 13, 201);
  fss::HalfTreeDpf<1, fss_test::RangeUint127Group, fss_test::TestCcrHash, uint8_t> tiny{fss_test::TestCcrHash{}, int4{11, 22, 33, 44}};
  fss_test::CheckGpuRange<1, uint8_t, 999, 1024>(tiny, 0, 1);
  fss_test::CheckGpuRange<1, uint8_t>(tiny, 1, 1);
}

namespace {

constexpr int4 kSplitSentinel = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};

// Checks the split Cws API against the Cw array forms for one configuration:
// every Cw entry, the packed control bit, every output of Eval, EvalAll
// and EvalRange, and the two-party reconstruction. stride == 0 selects the
// default-argument call.
template <int in_bits, typename Group, typename In>
void CheckHalfTreeSplit(In alpha, int4 beta, size_t stride) {
  using Scheme = fss::HalfTreeDpf<in_bits, Group, fss_test::TestCcrHash, In>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const bool default_stride = stride == 0;
  const size_t level_stride = default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(in_bits) * level_stride + 1;
  constexpr size_t kExtra = 3;
  constexpr size_t kDomain = size_t{1} << in_bits;

  typename Scheme::Cw ref[in_bits + 1];
  scheme.Gen(ref, seeds, alpha, beta);

  std::vector<int4> cws(used + kExtra, kSplitSentinel);
  const auto cws_out = cuda::std::span<int4>(cws.data(), cws.size());
  In tr = 0;
  if (default_stride) scheme.Gen(cws_out, tr, seed_span, alpha, beta);
  else scheme.Gen(cws_out, tr, seed_span, alpha, beta, stride);

  for (int l = 0; l < in_bits; ++l) {
    const size_t index = static_cast<size_t>(l) * level_stride;
    EXPECT_EQ(memcmp(&cws[index], &ref[l].s, sizeof(int4)), 0) << "cws layer " << l;
  }
  // The control component exists only at the last tree level: bit in_bits - 1 of
  // tr carries LCW_1 and every other bit is zero on output.
  EXPECT_EQ(tr, static_cast<In>(ref[in_bits - 1].extra) << (in_bits - 1))
      << "packed control bit";
  EXPECT_EQ(memcmp(&cws[used - 1], &ref[in_bits].s, sizeof(int4)), 0) << "cws terminal";
  // Only the described prefix is written: gap entries and trailing capacity keep the sentinel.
  for (size_t i = 0; i < cws.size(); ++i) {
    if (i % level_stride == 0 && i / level_stride <= static_cast<size_t>(in_bits)) continue;
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
          ? scheme.Eval(b, s0, split_cws, tr, xi)
          : scheme.Eval(b, s0, split_cws, tr, xi, stride);
      EXPECT_EQ(memcmp(&aos, &got, sizeof(int4)), 0) << "Eval party=" << party << " x=" << x;
    }

    std::vector<int4> aos_all(kDomain, kSplitSentinel);
    split_all[party].assign(kDomain + 2, kSplitSentinel);
    scheme.EvalAll(b, s0, ref, aos_all.data());
    if (default_stride) scheme.EvalAll(b, s0, split_cws, tr, split_all[party].data());
    else scheme.EvalAll(b, s0, split_cws, tr, split_all[party].data(), 0, stride);
    EXPECT_EQ(memcmp(aos_all.data(), split_all[party].data(), kDomain * sizeof(int4)), 0)
        << "EvalAll party=" << party;
    EXPECT_EQ(memcmp(&split_all[party][kDomain], &kSplitSentinel, sizeof(int4)), 0)
        << "EvalAll extra capacity party=" << party;

    for (int par_depth : {0, 2}) {
      std::vector<int4> aos_range(kDomain + 2, kSplitSentinel);
      std::vector<int4> split_range(kDomain + 2, kSplitSentinel);
      scheme.EvalRange(b, In{0}, static_cast<In>(kDomain - 1), s0, ref,
          cuda::std::span<int4>(aos_range.data(), kDomain), par_depth);
      const auto ys = cuda::std::span<int4>(split_range.data(), kDomain);
      if (default_stride) {
        scheme.EvalRange(b, In{0}, static_cast<In>(kDomain - 1), s0, split_cws, tr, ys, par_depth);
      } else {
        scheme.EvalRange(b, In{0}, static_cast<In>(kDomain - 1), s0, split_cws, tr, ys,
            par_depth, stride);
      }
      EXPECT_EQ(memcmp(aos_range.data(), split_range.data(), kDomain * sizeof(int4)), 0)
          << "EvalRange party=" << party << " par_depth=" << par_depth;
      for (size_t i = kDomain; i < split_range.size(); ++i) {
        EXPECT_EQ(memcmp(&split_range[i], &kSplitSentinel, sizeof(int4)), 0)
            << "EvalRange extra capacity at " << i;
      }
    }
  }

  for (size_t x = 0; x < kDomain; ++x) {
    const int4 expected = static_cast<In>(x) == alpha ? Group::From(beta).Into() : int4{};
    const int4 actual = (Group::From(split_all[0][x]) + Group::From(split_all[1][x])).Into();
    EXPECT_EQ(memcmp(&actual, &expected, GroupCompareSize<Group>()), 0) << "reconstruction x=" << x;
  }
}

template <int in_bits, typename Group, typename In>
void CheckHalfTreeSplitStrides(In alpha) {
  const int4 beta = {604, 981, 273, 84};
  for (size_t stride : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
    CheckHalfTreeSplit<in_bits, Group, In>(alpha, beta, stride);
  }
}

template <int in_bits, typename Group, typename In>
void CheckHalfTreeSplitFromCwStorage(In alpha, int4 beta) {
  using Scheme = fss::HalfTreeDpf<in_bits, Group, fss_test::TestCcrHash, In>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  typename Scheme::Cw words[in_bits + 1];
  scheme.Gen(words, seeds, alpha, beta);

  std::vector<int4> storage(2 * (in_bits + 1), kSplitSentinel);
  for (int l = 0; l <= in_bits; ++l) storage[2 * l] = words[l].s;
  const In tr = words[in_bits - 1].extra
      ? static_cast<In>(static_cast<In>(1) << (in_bits - 1)) : In{0};
  const auto cws = cuda::std::span<const int4>(storage.data(), storage.size());
  constexpr size_t kDomain = size_t{1} << in_bits;
  for (int party = 0; party < 2; ++party) {
    for (size_t x = 0; x < kDomain; ++x) {
      const In xi = static_cast<In>(x);
      const int4 expected = scheme.Eval(party, seeds[party], words, xi);
      const int4 actual = scheme.Eval(party, seeds[party], cws, tr, xi, 2);
      EXPECT_EQ(memcmp(&actual, &expected, sizeof(int4)), 0) << "party=" << party << " x=" << x;
    }
    std::vector<int4> expected_all(kDomain, kSplitSentinel);
    std::vector<int4> actual_all(kDomain, kSplitSentinel);
    scheme.EvalAll(party, seeds[party], words, expected_all.data());
    scheme.EvalAll(party, seeds[party], cws, tr, actual_all.data(), 0, 2);
    EXPECT_EQ(memcmp(actual_all.data(), expected_all.data(), kDomain * sizeof(int4)), 0)
        << "EvalAll party=" << party;
  }
}

}  // namespace

// in_bits 1 is the single-level tree, 4 is the smallest batched EvalAll()
// domain, 6 keeps the batched frontier with two parallel tasks, and 8 runs the
// full frontier. The Uint127 group covers the arithmetic path.
TEST(HalfTreeDpfCwsTest, Depth1) { CheckHalfTreeSplitStrides<1, BytesGroup, uint8_t>(1); }
TEST(HalfTreeDpfCwsTest, Depth2) { CheckHalfTreeSplitStrides<2, BytesGroup, uint8_t>(3); }
TEST(HalfTreeDpfCwsTest, Depth3) { CheckHalfTreeSplitStrides<3, BytesGroup, uint8_t>(5); }
TEST(HalfTreeDpfCwsTest, Depth4) { CheckHalfTreeSplitStrides<4, BytesGroup, uint8_t>(11); }
TEST(HalfTreeDpfCwsTest, Depth6) { CheckHalfTreeSplitStrides<6, BytesGroup, uint8_t>(11); }
TEST(HalfTreeDpfCwsTest, Depth8) { CheckHalfTreeSplitStrides<8, BytesGroup, uint8_t>(kAlpha); }
TEST(HalfTreeDpfCwsTest, Depth6Uint127) {
  CheckHalfTreeSplitStrides<6, Uint127Group, uint8_t>(11);
}
TEST(HalfTreeDpfCwsTest, Depth8GenericGroup) {
  CheckHalfTreeSplitStrides<8, Uint64Group, uint8_t>(kAlpha);
}
TEST(HalfTreeDpfCwsTest, CwStorageView) {
  CheckHalfTreeSplitFromCwStorage<8, BytesGroup, uint8_t>(kAlpha, {604, 981, 273, 84});
}

// Pins the packed control bit position: HalfTreeDpf uses only bit in_bits - 1 of
// tr. Gen writes no other bit, the other bits are ignored on input, and moving
// the bit to a wrong position changes the evaluation, so the position is caught.
TEST(HalfTreeDpfCwsTest, TrBitPosition) {
  constexpr int in_bits = 4;
  using Scheme = fss::HalfTreeDpf<in_bits, BytesGroup, fss_test::TestCcrHash, uint8_t>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const int4 beta = {604, 981, 273, 84};
  constexpr size_t kDomain = size_t{1} << in_bits;

  // An odd alpha makes the alpha path use LCW_1 at the final level. Pick one whose
  // LCW_1 is set so that moving the bit is observable.
  uint8_t alpha = 1;
  typename Scheme::Cw ref[in_bits + 1];
  for (; alpha < kDomain; alpha += 2) {
    scheme.Gen(ref, seeds, alpha, beta);
    if (ref[in_bits - 1].extra) break;
  }
  ASSERT_TRUE(ref[in_bits - 1].extra) << "no alpha with LCW_1 set";

  std::vector<int4> cws(in_bits + 1 + 2, kSplitSentinel);
  uint8_t tr = 0;
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha, beta);
  const auto split_cws = cuda::std::span<const int4>(cws.data(), in_bits + 1);
  EXPECT_EQ(tr, static_cast<uint8_t>(static_cast<uint8_t>(1) << (in_bits - 1)))
      << "Gen writes only bit in_bits - 1";

  // Bits other than in_bits - 1 are ignored on input, including bits at or above in_bits.
  for (uint8_t noise : {static_cast<uint8_t>(0x0f), static_cast<uint8_t>(0x30)}) {
    const uint8_t noisy = static_cast<uint8_t>(tr | noise);
    for (size_t x = 0; x < kDomain; ++x) {
      for (int party = 0; party < 2; ++party) {
        const int4 expected = scheme.Eval(party, seeds[party], ref, static_cast<uint8_t>(x));
        const int4 actual =
            scheme.Eval(party, seeds[party], split_cws, noisy, static_cast<uint8_t>(x));
        EXPECT_EQ(memcmp(&actual, &expected, sizeof(int4)), 0)
            << "noise=" << static_cast<int>(noise) << " party=" << party << " x=" << x;
      }
    }
  }

  // Moving the bit to bit 0 clears the true position and must change the result.
  const uint8_t moved = static_cast<uint8_t>(tr >> (in_bits - 1));
  size_t differences = 0;
  for (size_t x = 0; x < kDomain; ++x) {
    for (int party = 0; party < 2; ++party) {
      const int4 expected = scheme.Eval(party, seeds[party], ref, static_cast<uint8_t>(x));
      const int4 actual =
          scheme.Eval(party, seeds[party], split_cws, moved, static_cast<uint8_t>(x));
      if (memcmp(&actual, &expected, sizeof(int4)) != 0) ++differences;
    }
  }
  EXPECT_GT(differences, size_t{0}) << "a wrong control bit position was not caught";
}

#ifndef NDEBUG
TEST(HalfTreeDpfCwsTest, RejectsInvalidInputs) {
  using Scheme = fss::HalfTreeDpf<3, BytesGroup, fss_test::TestCcrHash, uint8_t>;
  Scheme scheme{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  const int4 beta = {604, 981, 273, 84};
  int4 cws[4] = {};
  int4 ys[8] = {};
  uint8_t tr = 0;
  const auto cws_out = cuda::std::span<int4>(cws, 4);
  const auto cws_in = cuda::std::span<const int4>(cws, 4);
  const int4 s0 = seeds[0];

  // A zero stride is only the batch sentinel, so a non-batch method rejects it.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cws_out, tr, seed_span, uint8_t{5}, beta, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, uint8_t{0}, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cws_in, uint8_t{0}, ys, 0, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalRange(false, uint8_t{0}, uint8_t{7}, s0, cws_in,
                        uint8_t{0}, cuda::std::span<int4>(ys, 8), 0, 0),
      "Assertion.*failed");
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cuda::std::span<int4>(cws, 3), tr, seed_span,
                        uint8_t{5}, beta),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(cws, 3),
                        uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(cws, 3),
                        uint8_t{0}, ys),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalRange(false, uint8_t{0}, uint8_t{7}, s0,
                        cuda::std::span<const int4>(cws, 3), uint8_t{0},
                        cuda::std::span<int4>(ys, 8)),
      "Assertion.*failed");
  // The terminal output correction word must have a zero LSB.
  int4 odd_cws[4] = {};
  odd_cws[3].w = 1;
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(odd_cws, 4),
                        uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(odd_cws, 4),
                        uint8_t{0}, ys),
      "Assertion.*failed");
  // A stride past the addressable guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / 3 + 1;
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, uint8_t{0},
                        overflowing_stride),
      "Assertion.*failed");
}
#endif  // NDEBUG

// The batch geometry validation is host code that runs before any CUDA call, so
// these checks need no device and read no buffer.
TEST(HalfTreeDpfSplitGpuTest, RejectsInvalidBatchGeometry) {
  using Wide = fss::HalfTreeDpf<32, BytesGroup, fss_test::TestCcrHash, uint64_t>;
  Wide wide{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
  int4 seed_storage[1] = {};
  int4 cws_storage[1] = {};
  int4 ys_storage[1] = {};
  uint64_t tr_storage[1] = {};

  // 2^32 keys of 2^32 outputs make the output product wrap in size_t, so the
  // guard must reject the call before the product is formed. The Cw
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

  using Small = fss::HalfTreeDpf<8, BytesGroup, fss_test::TestCcrHash, uint16_t>;
  Small small{fss_test::TestCcrHash{}, {31, 47, 59, 83}};
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
  // A Cw span that omits the terminal column is rejected.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, s0s,
                        cuda::std::span<const int4>(small_cws, kSmallRequired - 1), tr, ys)),
      cudaErrorInvalidValue);
  // An empty batch succeeds without reading any buffer.
  EXPECT_EQ((small.template EvalAllGpuBatch<16, 256>(false, cuda::std::span<const int4>(),
                        cuda::std::span<const int4>(), cuda::std::span<const uint16_t>(),
                        cuda::std::span<int4>())),
      cudaSuccess);
}

// The GPU split launches are compile-verified here and executed only on a
// machine with a free device, comparing against the host Cw-form outputs.
namespace {

template <typename Group>
void CheckHalfTreeSplitGpuLaunches() {
  int devices = 0;
  cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
      (device_status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(device_status, cudaSuccess);

  constexpr int kBits = 8;
  constexpr size_t kDomain = size_t{1} << kBits;
  using Scheme = fss::HalfTreeDpf<kBits, Group, fss_test::TestCcrHash, uint16_t>;
  Scheme scheme{fss_test::TestCcrHash{}, {11, 22, 33, 44}};
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 beta = {7, 19, 23, 42};
  const uint16_t alpha = 107;
  constexpr size_t kStride = 2;

  typename Scheme::Cw ref[kBits + 1];
  scheme.Gen(ref, seeds, alpha, beta);
  std::vector<int4> cws((kBits + 1) * kStride + 2, kSplitSentinel);
  uint16_t tr = 0;
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha, beta, kStride);

  int4 *device_cws = nullptr;
  int4 *device_ys = nullptr;
  uint16_t *device_tr = nullptr;
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
    const auto span_cws = cuda::std::span<const int4>(device_cws, cws.size());
    std::vector<int4> actual(kDomain, kSplitSentinel);
    ASSERT_EQ((scheme.template EvalAllGpu<8, 16>(b, seeds[party], span_cws, tr,
        cuda::std::span<int4>(device_ys, kDomain), stream, kStride)), cudaSuccess);
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
    ASSERT_EQ((scheme.template EvalRangeGpu<8, 16>(b, uint16_t{13}, uint16_t{201},
        seeds[party], span_cws, tr, cuda::std::span<int4>(device_ys, kDomain), stream, kStride)),
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
      cuda::std::span<const int4>(device_cws, cws.size()), tr,
      cuda::std::span<int4>(device_ys, kDomain), stream, 0)), cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template EvalRangeGpu<8, 16>(false, uint16_t{0}, uint16_t{255}, seeds[0],
      cuda::std::span<const int4>(device_cws, kBits), tr,
      cuda::std::span<int4>(device_ys, kDomain), stream)), cudaErrorInvalidValue);

  // Batch launches: the default stride is the dense level-major key count.
  typename Scheme::Cw ref_batch[kKeys][kBits + 1];
  std::vector<int4> batch_cws((kBits + 1) * kKeys, kSplitSentinel);
  int4 batch_s0s[kKeys];
  uint16_t batch_tr[kKeys] = {0, 0};
  for (size_t key = 0; key < kKeys; ++key) {
    batch_s0s[key] = seeds[key];
    scheme.Gen(ref_batch[key], seeds, static_cast<uint16_t>(alpha + key), beta);
    uint16_t key_tr = 0;
    // Offsetting by the key index fills the dense level-major layout.
    scheme.Gen(cuda::std::span<int4>(batch_cws.data() + key, batch_cws.size() - key), key_tr,
        cuda::std::span<const int4, 2>(seeds, 2), static_cast<uint16_t>(alpha + key), beta, kKeys);
    batch_tr[key] = key_tr;
  }
  ASSERT_EQ(cudaMemcpyAsync(device_cws, batch_cws.data(), batch_cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_tr, batch_tr, sizeof(batch_tr), cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  int4 *device_s0s = nullptr;
  ASSERT_EQ(cudaMalloc(&device_s0s, sizeof(batch_s0s)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_s0s, batch_s0s, sizeof(batch_s0s),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  int4 *device_batch_ys = nullptr;
  ASSERT_EQ(cudaMalloc(&device_batch_ys, kKeys * kDomain * sizeof(int4)), cudaSuccess);
  const auto span_batch_cws = cuda::std::span<const int4>(device_cws, batch_cws.size());
  const auto span_batch_tr = cuda::std::span<const uint16_t>(device_tr, kKeys);
  const auto span_batch_ys = cuda::std::span<int4>(device_batch_ys, kKeys * kDomain);
  for (int use_sentinel = 0; use_sentinel < 2; ++use_sentinel) {
    SCOPED_TRACE(use_sentinel);
    const size_t stride = use_sentinel == 0 ? fss::kDenseStride : kKeys;
    ASSERT_EQ((scheme.template EvalAllGpuBatch<8, 16>(false,
        cuda::std::span<const int4>(device_s0s, kKeys), span_batch_cws, span_batch_tr,
        span_batch_ys, stream, stride)), cudaSuccess);
    std::vector<int4> actual(kKeys * kDomain, kSplitSentinel);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_batch_ys, actual.size() * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t key = 0; key < kKeys; ++key) {
      for (size_t x = 0; x < kDomain; ++x) {
        const int4 expected =
            scheme.Eval(false, batch_s0s[key], ref_batch[key], static_cast<uint16_t>(x));
        EXPECT_EQ(memcmp(&actual[key * kDomain + x], &expected, sizeof(int4)), 0)
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

TEST(HalfTreeDpfSplitGpuTest, SplitLaunchesMatchCwHost) { CheckHalfTreeSplitGpuLaunches<BytesGroup>(); }
TEST(HalfTreeDpfSplitGpuTest, WideGroupSplitLaunchesCompile) {
  CheckHalfTreeSplitGpuLaunches<Uint127Group>();
}
