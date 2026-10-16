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
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, b_buf);

    // Eval at alpha
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

  void TestEvalAll() {
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
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

  void TestEvalAtNonAlpha() {
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
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

TEST_F(DpfBytesAesTest, EvalAtNonAlpha) {
  Prg prg(ctxs);
  DpfType dpf{prg};

  int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
      static_cast<int>((kBeta >> 64) & 0xFFFFFFFF), static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
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

// These tests exercise the 8-wide batched EvalAll() frontier at its in_bits
// and par_depth boundaries with the PRG whose Gen8() the batch pipelines
// through. in_bits below kBatchPrefix + kBatchTail (6) falls back to the
// per-input point walk, in_bits = 6 is the smallest batched domain and clamps
// the split depth to 0, and larger in_bits run both the sequential batched
// walk (par_depth = 0) and the frontier-width split (par_depth > 0).
template <int in_bits, int par_depth = -1>
class DpfAesRawTestBase : public ::testing::Test {
protected:
  using Group = BytesGroup;
  using Prg = fss::prg::Aes128MmoRaw<2>;
  using DpfType = fss::Dpf<in_bits, Group, Prg, uint16_t, par_depth>;

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
    DpfType dpf{prg};

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
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha5, b_buf);

  constexpr size_t n = 1ULL << 5;
  int4 ys0[n], ys1[n];
  dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0, n));
  dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1, n));

  int4 zero = {0, 0, 0, 0};
  for (size_t x = 0; x < n; ++x) {
    int4 e0 = dpf.Eval(false, k0, static_cast<uint8_t>(x));
    int4 e1 = dpf.Eval(true, k1, static_cast<uint8_t>(x));
    EXPECT_EQ(memcmp(&ys0[x], &e0, sizeof(int4)), 0) << "Party 0 mismatch at x=" << x;
    EXPECT_EQ(memcmp(&ys1[x], &e1, sizeof(int4)), 0) << "Party 1 mismatch at x=" << x;

    auto result = BytesGroup::From(ys0[x]) + BytesGroup::From(ys1[x]);
    int4 result_buf = result.Into();
    int4 expected_buf = x == kAlpha5 ? BytesGroup::From(b_buf).Into() : zero;
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, sizeof(int4)), 0) << "Reconstruction mismatch at x=" << x;
  }
}
