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
    int ret;
    do {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> dis;
      s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
      s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
      ret = vdpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha, MakeBBuf());
    } while (ret != 0);
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
TEST_F(VdpfUint128ChaChaTest, EvalAllMerkle) {
  TestEvalAllMerkle();
}
TEST_F(VdpfUint128ChaChaTest, ProveMerkleOdd) {
  TestProveMerkleOdd();
}
