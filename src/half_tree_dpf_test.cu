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

// ChaCha-based test fixture
template <typename Group>
class HalfTreeDpfChaChaTest : public ::testing::Test {
protected:
  using Prg = fss::prg::ChaCha<1>;
  using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, Prg, uint16_t>;

  int4 s0s[2];
  typename DpfType::Key k0, k1;
  Prg prg;
  int4 hash_key;

  HalfTreeDpfChaChaTest() : prg(kNonce0, kNonce1) {}

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    int4 y0 = dpf.Eval(false, k0, kAlpha);
    int4 y1 = dpf.Eval(true, k1, kAlpha);

    auto g0 = Group::From(y0);
    auto g1 = Group::From(y1);
    auto result = g0 + g1;
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();

    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAtNonAlpha() {
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 100; i++) {
      uint16_t x = dis(gen);
      if (x == kAlpha) continue;

      int4 y0 = dpf.Eval(false, k0, x);
      int4 y1 = dpf.Eval(true, k1, x);

      auto g0 = Group::From(y0);
      auto g1 = Group::From(y1);
      auto result = g0 + g1;

      int4 result_buf = result.Into();
      int4 zero = {0, 0, 0, 0};

      EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0) << "Failed at x=" << x;
    }
  }

  void TestEvalAll() {
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);

    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

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

using HalfTreeDpfBytesChaChaTest = HalfTreeDpfChaChaTest<BytesGroup>;
using HalfTreeDpfUint128ChaChaTest = HalfTreeDpfChaChaTest<Uint127Group>;
using HalfTreeDpfUint64ChaChaTest = HalfTreeDpfChaChaTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesChaChaTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint128ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128ChaChaTest, EvalAll) {
  TestEvalAll();
}

TEST_F(HalfTreeDpfUint64ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64ChaChaTest, EvalAll) {
  TestEvalAll();
}

// Aes128Soft-based test fixture
template <typename Group>
class HalfTreeDpfAesSoftTest : public ::testing::Test {
protected:
  using Prg = fss::prg::Aes128Soft<1>;
  using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, Prg, uint16_t>;

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    int4 y0 = dpf.Eval(false, k0, kAlpha);
    int4 y1 = dpf.Eval(true, k1, kAlpha);

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 100; i++) {
      uint16_t x = dis(gen);
      if (x == kAlpha) continue;

      int4 y0 = dpf.Eval(false, k0, x);
      int4 y1 = dpf.Eval(true, k1, x);

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);

    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

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
  using Prg = fss::prg::ChaCha<1>;
  using DpfType = fss::HalfTreeDpf<1, Group, Prg, uint8_t>;

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
    DpfType dpf{prg, hash_key};
    uint8_t alpha = 1;

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    int4 y0 = dpf.Eval(false, k0, alpha);
    int4 y1 = dpf.Eval(true, k1, alpha);

    auto result = Group::From(y0) + Group::From(y1);
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAtNonAlpha() {
    DpfType dpf{prg, hash_key};
    uint8_t alpha = 1;

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    uint8_t x = 0;
    int4 y0 = dpf.Eval(false, k0, x);
    int4 y1 = dpf.Eval(true, k1, x);

    auto result = Group::From(y0) + Group::From(y1);
    int4 result_buf = result.Into();
    int4 zero = {0, 0, 0, 0};
    EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
  }

  void TestEvalAll() {
    DpfType dpf{prg, hash_key};
    uint8_t alpha = 1;

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    int4 ys0[2], ys1[2];
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0, 2));
    dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1, 2));

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
template <int in_bits, int par_depth = -1>
class HalfTreeDpfAesRawTestBase : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128MmoRaw<1>;
  using DpfType = fss::HalfTreeDpf<in_bits, Group, Prg, uint16_t, par_depth>;

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    constexpr size_t n = 1ULL << in_bits;
    std::vector<int4> ys0(n), ys1(n);
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);

    constexpr size_t n = 1ULL << in_bits;
    std::vector<int4> ys0(n), ys1(n);
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

    for (size_t x = 0; x < n; ++x) {
      int4 e0 = dpf.Eval(false, k0, static_cast<uint16_t>(x));
      int4 e1 = dpf.Eval(true, k1, static_cast<uint16_t>(x));
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
  using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, Prg, uint16_t>;

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
    DpfType dpf{prg, hash_key};

    int4 b_buf = MakeBBuf();
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
    dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));

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
