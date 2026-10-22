#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/vdpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;
constexpr __uint128_t kBeta = 604;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

template <typename Group>
class VdpfChaChaTest : public ::testing::Test {
protected:
  using Prg = fss::prg::ChaCha<2>;
  using XorHash = fss::hash::Blake3;
  using Hash = fss::hash::Blake3;
  using VdpfType = fss::Vdpf<kAlphaBits, Group, Prg, XorHash, Hash, uint16_t>;

  int4 s0s[2];
  typename VdpfType::Key k0, k1;
  Prg prg;
  int4 hash_iv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};
  XorHash xor_hash;
  Hash hash_;

  VdpfChaChaTest()
      : prg(kNonce0, kNonce1),
        xor_hash(cuda::std::span<const int4, 2>(hash_iv, 2)),
        hash_(cuda::std::span<const int4, 2>(hash_iv, 2)) {}

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

  void GenKey() {
    VdpfType vdpf{prg, xor_hash, hash_};
    ASSERT_EQ(vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(s0s, 2), kAlpha, MakeBBuf()), 0);
  }

  void TestEvalAtAlpha() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    [[maybe_unused]] auto [y0, pi0_tilde] = vdpf.Eval(false, k0, kAlpha);
    [[maybe_unused]] auto [y1, pi1_tilde] = vdpf.Eval(true, k1, kAlpha);

    auto result = Group::From(y0) + Group::From(y1);
    auto expected = Group::From(MakeBBuf());
    int4 r = result.Into();
    int4 e = expected.Into();
    EXPECT_EQ(memcmp(&r, &e, sizeof(int4)), 0);
  }

  void TestEvalAtNonAlpha() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 100; i++) {
      uint16_t x = dis(gen);
      if (x == kAlpha) continue;

      [[maybe_unused]] auto [y0, pi0_tilde] = vdpf.Eval(false, k0, x);
      [[maybe_unused]] auto [y1, pi1_tilde] = vdpf.Eval(true, k1, x);

      auto result = Group::From(y0) + Group::From(y1);
      int4 r = result.Into();
      int4 zero = {0, 0, 0, 0};
      EXPECT_EQ(memcmp(&r, &zero, sizeof(int4)), 0) << "Failed at x=" << x;
    }
  }

  void TestVerifyEval() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    constexpr int L = 50;
    std::vector<cuda::std::array<int4, 4>> pi_tildes_0(L), pi_tildes_1(L);

    for (int i = 0; i < L; i++) {
      uint16_t x = dis(gen);
      [[maybe_unused]] auto [y0, pi_tilde0] = vdpf.Eval(false, k0, x);
      pi_tildes_0[i] = pi_tilde0;
      [[maybe_unused]] auto [y1, pi_tilde1] = vdpf.Eval(true, k1, x);
      pi_tildes_1[i] = pi_tilde1;
    }

    cuda::std::array<int4, 4> pi0, pi1;
    vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes_0), cuda::std::span<const int4, 4>(k0.cs), pi0);
    vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes_1), cuda::std::span<const int4, 4>(k0.cs), pi1);

    EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  }

  void TestEvalAll() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n);
    cuda::std::array<int4, 4> pi0, pi1;

    vdpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), n), pi0);
    vdpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), n), pi1);

    auto expected = Group::From(MakeBBuf());
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
      int4 r = result.Into();

      if (x == kAlpha) {
        EXPECT_EQ(memcmp(&r, &expected_buf, sizeof(int4)), 0) << "Failed at alpha=" << x;
      } else {
        EXPECT_EQ(memcmp(&r, &zero, sizeof(int4)), 0) << "Failed at x=" << x;
      }
    }

    EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  }

  void TestEvalRange() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    using Proof = cuda::std::array<int4, 4>;

    constexpr size_t n = 1ULL << kAlphaBits;
    const size_t alpha = static_cast<size_t>(kAlpha);

    ASSERT_LT(alpha, n);

    // Reference: full-domain evaluation of both keys.
    std::vector<int4> full0(n), full1(n);
    cuda::std::array<int4, 4> full_pi0, full_pi1;

    vdpf.EvalAll(false, k0, cuda::std::span<int4>(full0.data(), n), full_pi0);
    vdpf.EvalAll(true, k1, cuda::std::span<int4>(full1.data(), n), full_pi1);

    ASSERT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(full_pi0), cuda::std::span<const int4, 4>(full_pi1)));

    const int4 expected = Group::From(MakeBBuf()).Into();
    const int4 zero = {0, 0, 0, 0};
    const int4 flag = {1234567, -1234567, 1234567, -1234567};

    // different ranges to test.
    std::vector<std::pair<size_t, size_t>> ranges = {
        {0, n - 1},          // Entire domain
        {0, 3 * n / 4},      // left three quarters
        {n / 4, 3 * n / 4},  // central half
        {1, n - 2},          // Almost full domain, both ends pruned
        {7, 9},              // Tiny range crossing a subtree boundary
        {255, 257},          // Crosses 2^8
        {4095, 4097},        // Crosses 2^12
        {32767, 32769},      // Crosses root midpoint (2^15)

        {123, 12345},    // Unaligned range spanning several subtrees
        {30001, 40003},  // Unaligned, crosses root midpoint
        {65001, 65534},  // Near right edge, substantial left pruning
        {n - 1, n - 1},  // Rightmost leaf only
    };

    for (auto [L, R] : ranges) {
      SCOPED_TRACE(::testing::Message() << "L=" << L << ", R=" << R);

      const size_t count = R - L + 1;

      // We write to an array slightly larger than the range to detect writes outside the allocated buffer.
      std::vector<int4> ys0(count + 2, flag);
      std::vector<int4> ys1(count + 2, flag);

      cuda::std::array<int4, 4> pi0, pi1;

      // EvalRange writing to y[1 : count]
      vdpf.EvalRange(false, L, R, k0, cuda::std::span<int4>(ys0.data() + 1, count), pi0);

      vdpf.EvalRange(true, L, R, k1.s0, k1.cws, k1.cs, k1.ocw, cuda::std::span<int4>(ys1.data() + 1, count), pi1);

      // 1. no writes outside the requested output span.
      EXPECT_EQ(std::memcmp(&ys0.front(), &flag, sizeof(int4)), 0);
      EXPECT_EQ(std::memcmp(&ys0.back(), &flag, sizeof(int4)), 0);
      EXPECT_EQ(std::memcmp(&ys1.front(), &flag, sizeof(int4)), 0);
      EXPECT_EQ(std::memcmp(&ys1.back(), &flag, sizeof(int4)), 0);

      for (size_t offset = 0; offset < count; ++offset) {
        const size_t x = L + offset;

        // Range outputs are stored starting at ys[1] in this test.
        const int4 &y0 = ys0[offset + 1];
        const int4 &y1 = ys1[offset + 1];

        // 2. each share should match EvalAll at the index x in [0, n).
        EXPECT_EQ(std::memcmp(&y0, &full0[x], sizeof(int4)), 0) << "Party 0, x=" << x;

        EXPECT_EQ(std::memcmp(&y1, &full1[x], sizeof(int4)), 0) << "Party 1, x=" << x;

        // 3. shares should reconstruct the point function on the range.
        int4 reconstructed = (Group::From(y0) + Group::From(y1)).Into();

        const int4 &expected_x = (x == alpha) ? expected : zero;

        EXPECT_EQ(std::memcmp(&reconstructed, &expected_x, sizeof(int4)), 0) << "Incorrect reconstruction at x=" << x;
      }

      // 4. the two range proofs should verify.
      EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));

      // 5. Independently construct the proof using pointwise Eval + Prove.
      std::vector<Proof> pi_tildes0, pi_tildes1;
      pi_tildes0.reserve(count);
      pi_tildes1.reserve(count);

      for (size_t x = L; x <= R; ++x) {
        auto result0 = vdpf.Eval(false, k0, static_cast<decltype(kAlpha)>(x));
        auto result1 = vdpf.Eval(true, k1, static_cast<decltype(kAlpha)>(x));

        pi_tildes0.push_back(result0.pi_tilde);
        pi_tildes1.push_back(result1.pi_tilde);

        // Also check independent point evaluations.
        EXPECT_EQ(std::memcmp(&result0.y, &full0[x], sizeof(int4)), 0);
        EXPECT_EQ(std::memcmp(&result1.y, &full1[x], sizeof(int4)), 0);
      }

      Proof reference_pi0, reference_pi1;

      vdpf.Prove(cuda::std::span<const Proof>(pi_tildes0.data(), count), k0.cs, reference_pi0);

      vdpf.Prove(cuda::std::span<const Proof>(pi_tildes1.data(), count), k1.cs, reference_pi1);

      EXPECT_EQ(std::memcmp(pi0.data(), reference_pi0.data(), 4 * sizeof(int4)), 0)
          << "Party 0 proof differs from Eval + Prove";

      EXPECT_EQ(std::memcmp(pi1.data(), reference_pi1.data(), 4 * sizeof(int4)), 0)
          << "Party 1 proof differs from Eval + Prove";

      // 6. Full-domain EvalRange must agree exactly with EvalAll.
      if (L == 0 && R == n - 1) {
        EXPECT_EQ(std::memcmp(pi0.data(), full_pi0.data(), 4 * sizeof(int4)), 0);

        EXPECT_EQ(std::memcmp(pi1.data(), full_pi1.data(), 4 * sizeof(int4)), 0);
      }

      // 7. Tampering with a proof must make verification fail.
      auto tampered = pi1;
      tampered[0].x ^= 1;

      EXPECT_FALSE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(tampered)));
    }
  }

  void TestEvalAllMerkle() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    constexpr size_t n = 1ULL << kAlphaBits;
    std::vector<int4> ys0(n), ys1(n), ys0_chain(n);
    cuda::std::array<int4, 4> pi0, pi1, pi0_chain;

    vdpf.EvalAll(false, k0, cuda::std::span<int4>(ys0_chain.data(), n), pi0_chain);
    vdpf.EvalAllMerkle(false, k0, cuda::std::span<int4>(ys0.data(), n), pi0);
    vdpf.EvalAllMerkle(true, k1, cuda::std::span<int4>(ys1.data(), n), pi1);

    // The shares are the chain-based EvalAll's shares
    EXPECT_EQ(memcmp(ys0.data(), ys0_chain.data(), n * sizeof(int4)), 0);

    auto expected = Group::From(MakeBBuf());
    int4 expected_buf = expected.Into();
    int4 zero = {0, 0, 0, 0};

    for (size_t x = 0; x < n; ++x) {
      auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
      int4 r = result.Into();

      if (x == kAlpha) {
        EXPECT_EQ(memcmp(&r, &expected_buf, sizeof(int4)), 0) << "Failed at alpha=" << x;
      } else {
        EXPECT_EQ(memcmp(&r, &zero, sizeof(int4)), 0) << "Failed at x=" << x;
      }
    }

    EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  }

  void TestProveMerkleOdd() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    constexpr int L = 51;  // odd: the trailing node carries one level unchanged
    std::vector<cuda::std::array<int4, 4>> pi_tildes_0(L), pi_tildes_1(L);

    for (int i = 0; i < L; i++) {
      uint16_t x = dis(gen);
      [[maybe_unused]] auto [y0, pi_tilde0] = vdpf.Eval(false, k0, x);
      pi_tildes_0[i] = pi_tilde0;
      [[maybe_unused]] auto [y1, pi_tilde1] = vdpf.Eval(true, k1, x);
      pi_tildes_1[i] = pi_tilde1;
    }

    cuda::std::array<int4, 4> pi0, pi1;
    vdpf.ProveMerkle(
        cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes_0), cuda::std::span<const int4, 4>(k0.cs), pi0);
    vdpf.ProveMerkle(
        cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes_1), cuda::std::span<const int4, 4>(k0.cs), pi1);

    EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  }
};

using VdpfBytesChaChaTest = VdpfChaChaTest<BytesGroup>;
using VdpfUint128ChaChaTest = VdpfChaChaTest<Uint127Group>;

TEST_F(VdpfBytesChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(VdpfBytesChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(VdpfBytesChaChaTest, VerifyEval) {
  TestVerifyEval();
}
TEST_F(VdpfBytesChaChaTest, EvalAll) {
  TestEvalAll();
}
TEST_F(VdpfBytesChaChaTest, EvalRange) {
  TestEvalRange();
}
TEST_F(VdpfBytesChaChaTest, EvalAllMerkle) {
  TestEvalAllMerkle();
}
TEST_F(VdpfBytesChaChaTest, ProveMerkleOdd) {
  TestProveMerkleOdd();
}

TEST_F(VdpfUint128ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(VdpfUint128ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(VdpfUint128ChaChaTest, VerifyEval) {
  TestVerifyEval();
}
TEST_F(VdpfUint128ChaChaTest, EvalAll) {
  TestEvalAll();
}
TEST_F(VdpfUint128ChaChaTest, EvalRange) {
  TestEvalRange();
}
TEST_F(VdpfUint128ChaChaTest, EvalAllMerkle) {
  TestEvalAllMerkle();
}
TEST_F(VdpfUint128ChaChaTest, ProveMerkleOdd) {
  TestProveMerkleOdd();
}
