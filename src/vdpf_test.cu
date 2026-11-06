#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <thread>
#include <fss/vdpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include "core_api_test_helpers.cuh"

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
    ASSERT_EQ(fss_test::GenPair(vdpf, k0, k1, cuda::std::span<int4, 2>(s0s, 2), kAlpha, MakeBBuf()), 0);
  }

  void TestEvalAtAlpha() {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    [[maybe_unused]] auto [y0, pi0_tilde] = fss_test::EvalKey(vdpf, false, k0, kAlpha);
    [[maybe_unused]] auto [y1, pi1_tilde] = fss_test::EvalKey(vdpf, true, k1, kAlpha);

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

      [[maybe_unused]] auto [y0, pi0_tilde] = fss_test::EvalKey(vdpf, false, k0, x);
      [[maybe_unused]] auto [y1, pi1_tilde] = fss_test::EvalKey(vdpf, true, k1, x);

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
      [[maybe_unused]] auto [y0, pi_tilde0] = fss_test::EvalKey(vdpf, false, k0, x);
      pi_tildes_0[i] = pi_tilde0;
      [[maybe_unused]] auto [y1, pi_tilde1] = fss_test::EvalKey(vdpf, true, k1, x);
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

    fss_test::EvalAllKey(vdpf, false, k0, cuda::std::span<int4>(ys0.data(), n), pi0);
    fss_test::EvalAllKey(vdpf, true, k1, cuda::std::span<int4>(ys1.data(), n), pi1);

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

    fss_test::EvalAllKey(vdpf, false, k0, cuda::std::span<int4>(full0.data(), n), full_pi0);
    fss_test::EvalAllKey(vdpf, true, k1, cuda::std::span<int4>(full1.data(), n), full_pi1);

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
      fss_test::EvalRangeKey(vdpf, false, L, R, k0, cuda::std::span<int4>(ys0.data() + 1, count), pi0);

      vdpf.EvalRange(true, L, R, k1.s0, k1.cws, k1.cs, cuda::std::span<int4>(ys1.data() + 1, count), pi1);

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
        auto result0 = fss_test::EvalKey(vdpf, false, k0, static_cast<decltype(kAlpha)>(x));
        auto result1 = fss_test::EvalKey(vdpf, true, k1, static_cast<decltype(kAlpha)>(x));

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

    fss_test::EvalAllKey(vdpf, false, k0, cuda::std::span<int4>(ys0_chain.data(), n), pi0_chain);
    fss_test::EvalAllMerkleKey(vdpf, false, k0, cuda::std::span<int4>(ys0.data(), n), pi0);
    fss_test::EvalAllMerkleKey(vdpf, true, k1, cuda::std::span<int4>(ys1.data(), n), pi1);

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
      [[maybe_unused]] auto [y0, pi_tilde0] = fss_test::EvalKey(vdpf, false, k0, x);
      pi_tildes_0[i] = pi_tilde0;
      [[maybe_unused]] auto [y1, pi_tilde1] = fss_test::EvalKey(vdpf, true, k1, x);
      pi_tildes_1[i] = pi_tilde1;
    }

    cuda::std::array<int4, 4> pi0, pi1;
    vdpf.ProveMerkle(
        cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes_0), cuda::std::span<const int4, 4>(k0.cs), pi0);
    vdpf.ProveMerkle(
        cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes_1), cuda::std::span<const int4, 4>(k0.cs), pi1);

    EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  }

  void CheckOrderedBatch(bool merkle) {
    GenKey();
    VdpfType vdpf{prg, xor_hash, hash_};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    // Ordered query sequences: an unsorted adversarial order with duplicates
    // and domain extremes, a single query, then random sparse sequences of
    // odd and even sizes (odd sizes carry a trailing node in the Merkle fold).
    std::vector<std::vector<uint16_t>> queries;
    queries.push_back({kAlpha, 0, 0xFFFF, kAlpha, 1, 0xFFFE, 42, kAlpha, 2});
    queries.push_back({kAlpha});
    for (size_t n : {2, 3, 17, 50, 51}) {
      std::vector<uint16_t> xs;
      xs.reserve(n);
      for (size_t i = 0; i < n; ++i) xs.push_back(dis(gen));
      queries.push_back(std::move(xs));
    }

    for (const auto &xs : queries) {
      SCOPED_TRACE(::testing::Message() << "queries=" << xs.size() << " merkle=" << merkle);
      const size_t n = xs.size();
      std::vector<int4> ys0(n), ys1(n);
      cuda::std::array<int4, 4> pi0, pi1;

      if (merkle) {
        vdpf.BatchEvalMerkle(false, k0.s0, k0.cws, k0.cs, cuda::std::span<const uint16_t>(xs),
            cuda::std::span<int4>(ys0), pi0);
        vdpf.BatchEvalMerkle(true, k1.s0, k1.cws, k1.cs, cuda::std::span<const uint16_t>(xs),
            cuda::std::span<int4>(ys1), pi1);
      } else {
        vdpf.BatchEval(false, k0.s0, k0.cws, k0.cs, cuda::std::span<const uint16_t>(xs),
            cuda::std::span<int4>(ys0), pi0);
        vdpf.BatchEval(true, k1.s0, k1.cws, k1.cs, cuda::std::span<const uint16_t>(xs),
            cuda::std::span<int4>(ys1), pi1);
      }

      // Pointwise reference: Eval per input, then Prove or ProveMerkle.
      std::vector<cuda::std::array<int4, 4>> pi_tildes0(n), pi_tildes1(n);
      std::vector<int4> ref0(n), ref1(n);
      for (size_t i = 0; i < n; ++i) {
        pi_tildes0[i] = vdpf.Eval(false, k0.s0, k0.cws, k0.cs, xs[i], ref0[i]);
        pi_tildes1[i] = vdpf.Eval(true, k1.s0, k1.cws, k1.cs, xs[i], ref1[i]);
      }
      for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(memcmp(&ys0[i], &ref0[i], sizeof(int4)), 0) << "party 0 share at index " << i;
        EXPECT_EQ(memcmp(&ys1[i], &ref1[i], sizeof(int4)), 0) << "party 1 share at index " << i;
      }

      cuda::std::array<int4, 4> reference_pi0, reference_pi1;
      if (merkle) {
        vdpf.ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes0), k0.cs, reference_pi0);
        vdpf.ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes1), k1.cs, reference_pi1);
      } else {
        vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes0), k0.cs, reference_pi0);
        vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes1), k1.cs, reference_pi1);
      }
      EXPECT_EQ(memcmp(pi0.data(), reference_pi0.data(), 4 * sizeof(int4)), 0)
          << "party 0 proof differs from pointwise Eval + fold";
      EXPECT_EQ(memcmp(pi1.data(), reference_pi1.data(), 4 * sizeof(int4)), 0)
          << "party 1 proof differs from pointwise Eval + fold";

      const int4 expected = Group::From(MakeBBuf()).Into();
      const int4 zero = {0, 0, 0, 0};
      for (size_t i = 0; i < n; ++i) {
        int4 reconstructed = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
        const int4 &expected_x = (xs[i] == kAlpha) ? expected : zero;
        EXPECT_EQ(memcmp(&reconstructed, &expected_x, sizeof(int4)), 0) << "x=" << xs[i];
      }

      EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
      auto tampered = pi1;
      tampered[0].x ^= 1;
      EXPECT_FALSE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(tampered)));
    }
  }

  void TestBatchEval() { CheckOrderedBatch(false); }
  void TestBatchEvalMerkle() { CheckOrderedBatch(true); }
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
TEST_F(VdpfBytesChaChaTest, BatchEval) {
  TestBatchEval();
}
TEST_F(VdpfBytesChaChaTest, BatchEvalMerkle) {
  TestBatchEvalMerkle();
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
TEST_F(VdpfUint128ChaChaTest, BatchEval) {
  TestBatchEval();
}
TEST_F(VdpfUint128ChaChaTest, BatchEvalMerkle) {
  TestBatchEvalMerkle();
}

namespace {

struct MerkleRangeUint127Group {
  Uint127Group value;
  __host__ __device__ MerkleRangeUint127Group operator+(MerkleRangeUint127Group rhs) const {
    return {value + rhs.value};
  }
  __host__ __device__ MerkleRangeUint127Group operator-() const { return {-value}; }
  __host__ __device__ static MerkleRangeUint127Group From(int4 input) {
    return {Uint127Group::From(input)};
  }
  __host__ __device__ int4 Into() const { return value.Into(); }
};

template <typename T>
struct MerkleRangeDeviceBuffer {
  T *data = nullptr;
  ~MerkleRangeDeviceBuffer() {
    if (data != nullptr) EXPECT_EQ(cudaFree(data), cudaSuccess);
  }
  cudaError_t Allocate(size_t count) { return cudaMalloc(&data, count * sizeof(T)); }
};

void ExpectMerkleBlock(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

template <int bits, typename In, typename Group, int frontier = 16, int threads = 256,
    typename Prg = fss::prg::ChaCha<2>>
void CheckRangeMerkle(In first, In last, bool gpu, Prg prg = Prg{kNonce0, kNonce1}) {
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<bits, Group, Prg, Hash, Hash, In>;
  const size_t count = static_cast<size_t>(last - first) + 1;
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 beta = {42, 12, 24, 36};
  const int4 guard = {23, 45, 67, 88};
  Hash hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{prg, hash, hash};
  typename Scheme::Key keys[2];
  ASSERT_EQ(fss_test::GenPair(scheme, keys[0], keys[1], seeds, first, beta), 0);
  cuda::std::array<int4, 4> proofs[2];
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party << " count " << count);
    const auto &key = keys[party];
    std::vector<int4> expected(count);
    std::vector<cuda::std::array<int4, 4>> material(count);
    for (size_t j = 0; j < count; ++j)
      material[j] = scheme.Eval(party, key.s0, key.cws, key.cs,
          static_cast<In>(first + j), expected[j]);
    scheme.ProveMerkle(material, key.cs, proofs[party]);
    for (int par_depth : {0, 2}) {
      std::vector<int4> actual(count + 2, guard);
      cuda::std::array<int4, 4> proof;
      scheme.EvalRangeMerkle(party, first, last, key.s0, key.cws, key.cs,
          cuda::std::span<int4>(actual.data() + 1, count + 1), proof, par_depth);
      ExpectMerkleBlock(actual.front(), guard);
      ExpectMerkleBlock(actual.back(), guard);
      for (size_t j = 0; j < count; ++j) ExpectMerkleBlock(actual[j + 1], expected[j]);
      for (int j = 0; j < 4; ++j) ExpectMerkleBlock(proof[j], proofs[party][j]);
    }
    if (!gpu) continue;
    MerkleRangeDeviceBuffer<typename Scheme::Cw> device_cws;
    MerkleRangeDeviceBuffer<int4> device_cs, device_ys, device_pi;
    ASSERT_EQ(device_cws.Allocate(bits + 1), cudaSuccess);
    ASSERT_EQ(device_cs.Allocate(4), cudaSuccess);
    ASSERT_EQ(device_ys.Allocate(count + 2), cudaSuccess);
    ASSERT_EQ(device_pi.Allocate(4), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_cws.data, key.cws, sizeof(key.cws), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_cs.data, key.cs, sizeof(key.cs), cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<int4> actual(count + 2, guard);
    ASSERT_EQ(cudaMemcpy(device_ys.data, actual.data(), actual.size() * sizeof(int4), cudaMemcpyHostToDevice), cudaSuccess);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    const cudaError_t launch = [&] {
      if constexpr (frontier == 16 && threads == 256) {
        return scheme.EvalRangeMerkleGpu(party, first, last, key.s0,
            {device_cws.data, bits + 1}, cuda::std::span<const int4, 4>(device_cs.data, 4),
            {device_ys.data + 1, count + 1}, cuda::std::span<int4, 4>(device_pi.data, 4), stream);
      } else {
        return scheme.template EvalRangeMerkleGpu<frontier, threads>(party, first, last, key.s0,
            {device_cws.data, bits + 1}, cuda::std::span<const int4, 4>(device_cs.data, 4),
            {device_ys.data + 1, count + 1}, cuda::std::span<int4, 4>(device_pi.data, 4), stream);
      }
    }();
    EXPECT_EQ(launch, cudaSuccess);
    const cudaError_t execution = cudaStreamSynchronize(stream);
    EXPECT_EQ(execution, cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    ASSERT_EQ(launch, cudaSuccess);
    ASSERT_EQ(execution, cudaSuccess);
    cuda::std::array<int4, 4> proof;
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data, actual.size() * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(proof.data(), device_pi.data, sizeof(proof), cudaMemcpyDeviceToHost), cudaSuccess);
    ExpectMerkleBlock(actual.front(), guard);
    ExpectMerkleBlock(actual.back(), guard);
    for (size_t j = 0; j < count; ++j) ExpectMerkleBlock(actual[j + 1], expected[j]);
    for (int j = 0; j < 4; ++j) ExpectMerkleBlock(proof[j], proofs[party][j]);
  }
  EXPECT_TRUE(Scheme::Verify(proofs[0], proofs[1]));
}

bool HasMerkleRangeDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST(VdpfRangeMerkle, CpuOddAndOffsetBytes) {
  for (uint16_t count : {1, 4, 5, 7, 13, 257})
    CheckRangeMerkle<16, uint16_t, BytesGroup>(3, 3 + count - 1, false);
}

TEST(VdpfRangeMerkle, CpuOddAndOffsetUint127) {
  for (uint16_t count : {1, 4, 5, 7, 13})
    CheckRangeMerkle<16, uint16_t, MerkleRangeUint127Group>(3, 3 + count - 1, false);
}

TEST(VdpfRangeMerkle, CpuWideDomain) {
  const __uint128_t maximum = ~__uint128_t{0};
  CheckRangeMerkle<128, __uint128_t, BytesGroup>(maximum - 12, maximum, false);
  CheckRangeMerkle<128, __uint128_t, BytesGroup>((__uint128_t{1} << 127) - 3,
      (__uint128_t{1} << 127) + 3, false);
  CheckRangeMerkle<128, __uint128_t, BytesGroup>(maximum, maximum, false);
}

TEST(VdpfRangeMerkle, GpuOddAndOffsetBytes) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  for (uint16_t count : {1, 4, 5, 7, 13, 257, 1025})
    CheckRangeMerkle<16, uint16_t, BytesGroup>(3, 3 + count - 1, true);
}

TEST(VdpfRangeMerkle, GpuOddAndOffsetUint127) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  for (uint16_t count : {1, 4, 5, 7, 13})
    CheckRangeMerkle<16, uint16_t, MerkleRangeUint127Group>(3, 3 + count - 1, true);
}

TEST(VdpfRangeMerkle, GpuWideDomain) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  const __uint128_t maximum = ~__uint128_t{0};
  CheckRangeMerkle<128, __uint128_t, BytesGroup>(maximum - 12, maximum, true);
  CheckRangeMerkle<128, __uint128_t, BytesGroup>((__uint128_t{1} << 127) - 3,
      (__uint128_t{1} << 127) + 3, true);
  CheckRangeMerkle<128, __uint128_t, BytesGroup>(maximum, maximum, true);
}

TEST(VdpfRangeMerkle, GpuFixedFrontierAndChunkSplit) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  CheckRangeMerkle<1, uint8_t, BytesGroup>(0, 1, true);
  CheckRangeMerkle<8, uint8_t, BytesGroup>(0, 255, true);
  CheckRangeMerkle<16, uint16_t, BytesGroup, 2, 4>(3, 1027, true);
  CheckRangeMerkle<128, __uint128_t, BytesGroup, 128, 1024>(0, 1023, true);
  CheckRangeMerkle<16, uint16_t, BytesGroup, 0, 1>(0, 1023, true);
  CheckRangeMerkle<16, uint16_t, BytesGroup, 16, 1024>(0, 1023, true);
}

// Ordered sparse query sequences for the batch entry: duplicates, unsorted
// order, and domain extremes, including an odd trailing fold node.
template <typename In>
std::vector<std::vector<In>> BatchQuerySequences(In alpha, std::mt19937 &gen) {
  std::uniform_int_distribution<int> dis(0, 0xFFFF);
  std::vector<std::vector<In>> queries;
  const In maximum = static_cast<In>((In{1} << 16) - 1);
  queries.push_back({alpha, In{0}, maximum, alpha, In{1}, static_cast<In>(maximum - 1), In{42}, alpha, In{2}});
  queries.push_back({alpha});
  queries.push_back({maximum, static_cast<In>(maximum - 1), static_cast<In>(maximum - 2)});
  for (size_t n : {2, 3, 17, 50, 51, 257}) {
    std::vector<In> xs;
    xs.reserve(n);
    for (size_t i = 0; i < n; ++i) xs.push_back(static_cast<In>(dis(gen)));
    queries.push_back(std::move(xs));
  }
  return queries;
}

template <int bits, typename In, typename Group, int threads = 256,
    typename Prg = fss::prg::ChaCha<2>>
void CheckBatchMerkle(const std::vector<In> &xs, bool gpu, Prg prg = Prg{kNonce0, kNonce1}) {
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<bits, Group, Prg, Hash, Hash, In>;
  const size_t count = xs.size();
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const int4 beta = {42, 12, 24, 36};
  const int4 guard = {23, 45, 67, 88};
  Hash hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{prg, hash, hash};
  typename Scheme::Key keys[2];
  ASSERT_EQ(fss_test::GenPair(scheme, keys[0], keys[1], seeds, In{7}, beta), 0);
  cuda::std::array<int4, 4> proofs[2];
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party << " count " << count);
    const auto &key = keys[party];
    std::vector<int4> expected(count + 2, guard);
    scheme.BatchEvalMerkle(party, key.s0, key.cws, key.cs, cuda::std::span<const In>(xs),
        cuda::std::span<int4>(expected.data() + 1, count + 1), proofs[party]);
    // Pointwise reference for both the shares and the proof.
    std::vector<int4> pointwise(count);
    std::vector<cuda::std::array<int4, 4>> material(count);
    for (size_t j = 0; j < count; ++j)
      material[j] = scheme.Eval(party, key.s0, key.cws, key.cs, xs[j], pointwise[j]);
    cuda::std::array<int4, 4> reference;
    scheme.ProveMerkle(material, key.cs, reference);
    for (int j = 0; j < 4; ++j) ExpectMerkleBlock(proofs[party][j], reference[j]);
    ExpectMerkleBlock(expected.front(), guard);
    ExpectMerkleBlock(expected.back(), guard);
    for (size_t j = 0; j < count; ++j) ExpectMerkleBlock(expected[j + 1], pointwise[j]);
    if (!gpu) continue;
    MerkleRangeDeviceBuffer<typename Scheme::Cw> device_cws;
    MerkleRangeDeviceBuffer<int4> device_cs, device_ys, device_pi;
    MerkleRangeDeviceBuffer<In> device_xs;
    ASSERT_EQ(device_cws.Allocate(bits + 1), cudaSuccess);
    ASSERT_EQ(device_cs.Allocate(4), cudaSuccess);
    ASSERT_EQ(device_xs.Allocate(count), cudaSuccess);
    ASSERT_EQ(device_ys.Allocate(count + 2), cudaSuccess);
    ASSERT_EQ(device_pi.Allocate(4), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_cws.data, key.cws, sizeof(key.cws), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_cs.data, key.cs, sizeof(key.cs), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_xs.data, xs.data(), count * sizeof(In), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_ys.data, expected.data(), (count + 2) * sizeof(int4), cudaMemcpyHostToDevice), cudaSuccess);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    const cudaError_t launch = [&] {
      if constexpr (threads == 256) {
        return scheme.BatchEvalMerkleGpu(party, key.s0,
            {device_cws.data, bits + 1}, cuda::std::span<const int4, 4>(device_cs.data, 4),
            cuda::std::span<const In>(device_xs.data, count),
            {device_ys.data + 1, count + 1}, cuda::std::span<int4, 4>(device_pi.data, 4), stream);
      } else {
        return scheme.template BatchEvalMerkleGpu<threads>(party, key.s0,
            {device_cws.data, bits + 1}, cuda::std::span<const int4, 4>(device_cs.data, 4),
            cuda::std::span<const In>(device_xs.data, count),
            {device_ys.data + 1, count + 1}, cuda::std::span<int4, 4>(device_pi.data, 4), stream);
      }
    }();
    EXPECT_EQ(launch, cudaSuccess);
    const cudaError_t execution = cudaStreamSynchronize(stream);
    EXPECT_EQ(execution, cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    ASSERT_EQ(launch, cudaSuccess);
    ASSERT_EQ(execution, cudaSuccess);
    std::vector<int4> actual(count + 2);
    cuda::std::array<int4, 4> proof;
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data, (count + 2) * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(proof.data(), device_pi.data, sizeof(proof), cudaMemcpyDeviceToHost), cudaSuccess);
    ExpectMerkleBlock(actual.front(), guard);
    ExpectMerkleBlock(actual.back(), guard);
    for (size_t j = 0; j < count; ++j) ExpectMerkleBlock(actual[j + 1], pointwise[j]);
    for (int j = 0; j < 4; ++j) ExpectMerkleBlock(proof[j], proofs[party][j]);
  }
  EXPECT_TRUE(Scheme::Verify(proofs[0], proofs[1]));
}

TEST(VdpfBatchMerkle, CpuMatchesPointwiseEvalAndProveMerkle) {
  std::random_device rd;
  std::mt19937 gen(rd());
  for (const auto &xs : BatchQuerySequences<uint16_t>(7, gen)) {
    CheckBatchMerkle<16, uint16_t, BytesGroup>(xs, false);
    CheckBatchMerkle<16, uint16_t, MerkleRangeUint127Group>(xs, false);
  }
}

TEST(VdpfBatchMerkle, CpuWideDomainSparseQueries) {
  const __uint128_t maximum = ~__uint128_t{0};
  CheckBatchMerkle<128, __uint128_t, BytesGroup>({maximum, maximum - 12, maximum, maximum - 12}, false);
  CheckBatchMerkle<128, __uint128_t, BytesGroup>({maximum}, false);
}

TEST(VdpfBatchMerkle, GpuMatchesHostBatchEvalMerkle) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  std::random_device rd;
  std::mt19937 gen(rd());
  for (const auto &xs : BatchQuerySequences<uint16_t>(7, gen))
    CheckBatchMerkle<16, uint16_t, BytesGroup>(xs, true);
}

TEST(VdpfBatchMerkle, GpuCustomThreadsMatchHost) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  std::random_device rd;
  std::mt19937 gen(rd());
  const auto queries = BatchQuerySequences<uint16_t>(7, gen);
  CheckBatchMerkle<16, uint16_t, BytesGroup, 64>(queries[0], true);
  CheckBatchMerkle<16, uint16_t, BytesGroup, 1>(queries[1], true);
  CheckBatchMerkle<16, uint16_t, MerkleRangeUint127Group, 1024>(queries[3], true);
}

TEST(VdpfBatchMerkle, GpuRejectsInvalidBatchArguments) {
  using Prg = fss::prg::ChaCha<2>;
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<128, BytesGroup, Prg, Hash, Hash, __uint128_t>;
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Hash hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg{kNonce0, kNonce1}, hash, hash};
  typename Scheme::Cw cws[129];
  int4 cs[4], ys[4], pi[4];
  const __uint128_t xs[4] = {0, 1, 2, 3};
  // Empty query sequence.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, cws, cs,
      cuda::std::span<const __uint128_t>(), {ys, 4}, pi), cudaErrorInvalidValue);
  // Output storage shorter than the query sequence.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, cws, cs, {xs, 4},
      cuda::std::span<int4>(ys, 3), pi), cudaErrorInvalidValue);
  // Too few correction words.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{},
      cuda::std::span<const typename Scheme::Cw>(cws, 128), cs, {xs, 4}, {ys, 4},
      pi), cudaErrorInvalidValue);
  // Null buffers.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, cws, cs, {xs, 4},
      cuda::std::span<int4>(static_cast<int4 *>(nullptr), size_t{4}),
      pi), cudaErrorInvalidValue);
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, cws, cs, {xs, 4}, {ys, 4},
      cuda::std::span<int4, 4>(static_cast<int4 *>(nullptr), 4)), cudaErrorInvalidValue);
}

struct XorHashContractPrg {
  __host__ __device__ cuda::std::array<int4, 2> Gen(int4) {
    const int4 child = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};
    return {child, child};
  }
};

struct XorHashContractObserver {
  std::vector<cuda::std::array<int4, 2>> *messages;

  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4> input) {
#ifndef __CUDA_ARCH__
    auto [coordinate, seed] = input;
    messages->push_back({coordinate, seed});
#endif
    return {};
  }
};

TEST(VdpfXorHashContract, WideCoordinateAndClampedSeed) {
  using Scheme = fss::Vdpf<128, BytesGroup, XorHashContractPrg,
      XorHashContractObserver, fss::hash::Blake3, __uint128_t>;
  std::vector<cuda::std::array<int4, 2>> messages;
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Scheme scheme{XorHashContractPrg{}, XorHashContractObserver{&messages},
      fss::hash::Blake3{cuda::std::span<const int4, 2>(iv, 2)}};
  const Scheme::Cw cws[129]{};
  const int4 cs[4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  const int4 initial_seed = {11, 22, 33, 45};
  const int4 expected_seed = {0x12345678, 0x23456789, 0x3456789a, 0x456789aa};
  struct Case {
    __uint128_t x;
    int4 coordinate;
  };
  const Case cases[] = {
      {0, {0, 0, 0, 0}},
      {__uint128_t{1} << 96, {0, 0, 0, 1}},
      {__uint128_t{1} << 127, {0, 0, 0, -2147483647 - 1}},
      {(__uint128_t{0x76543210} << 96) | (__uint128_t{0x6543210f} << 64) |
          (__uint128_t{0x543210fe} << 32) | __uint128_t{0x43210fed},
          {0x43210fed, 0x543210fe, 0x6543210f, 0x76543210}},
  };
  // Constant children isolate the backend payload from tree traversal.
  for (const auto &test_case : cases) {
    for (int party = 0; party < 2; ++party) {
      SCOPED_TRACE(::testing::Message() << "party " << party
          << " coordinate high word " << test_case.coordinate.w);
      messages.clear();
      int4 y;
      scheme.Eval(party != 0, initial_seed, cws,
          cuda::std::span<const int4, 4>(cs, 4), test_case.x, y);
      ASSERT_FALSE(messages.empty());
      for (const auto &message : messages) {
        ExpectMerkleBlock(message[0], test_case.coordinate);
        ExpectMerkleBlock(message[1], expected_seed);
      }
    }
  }
}

void CheckFixedBit96CaseRejects(bool merkle) {
  SCOPED_TRACE(merkle ? "Merkle" : "chain");
  using Input = __uint128_t;
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<97, Uint127Group, fss::prg::ChaCha<2>, Hash, Hash, Input>;
  const int4 iv[2]{};
  Hash hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{fss::prg::ChaCha<2>{0, 0}, hash, hash};
  const Input xs[2] = {0, Input{1} << 96};
  ASSERT_TRUE(xs[0] != xs[1]);
  ASSERT_TRUE(xs[0] < (Input{1} << 97));
  ASSERT_TRUE(xs[1] < (Input{1} << 97));

  // Frozen trial 2 from 010c0d2. All remaining corrections, including the
  // terminal, are zero. Keep the original cs when changing the hash encoding.
  const Scheme::Cw cws[98] = {
      {{1715251663, 1119711520, static_cast<int>(4107019318u), 1095222437}, true}};
  const int4 cs[4] = {
      {static_cast<int>(2323156350u), 1254992691, 377754959, static_cast<int>(2978946471u)},
      {583504993, 39867407, static_cast<int>(2754790639u), static_cast<int>(2786589795u)},
      {static_cast<int>(3353835671u), 972790625, static_cast<int>(2292373154u),
          static_cast<int>(2901350010u)},
      {1253408694, 354916733, static_cast<int>(3490181160u), 2038253979},
  };
  bool nonzero_cs = false;
  for (const int4 &block : cs)
    nonzero_cs |= (block.x | block.y | block.z | block.w) != 0;
  ASSERT_TRUE(nonzero_cs);
  const int4 initial_seed = {2, 0, 0, 0};
  const cuda::std::span<const Scheme::Cw> common_cws(cws, 98);
  const cuda::std::span<const int4, 4> common_cs(cs, 4);
  const cuda::std::span<const Input> common_inputs(xs, 2);
  int4 outputs[2][2];
  cuda::std::array<int4, 4> proofs[2];
  for (int party = 0; party < 2; ++party) {
    if (merkle) {
      scheme.BatchEvalMerkle(party != 0, initial_seed, common_cws, common_cs,
          common_inputs, cuda::std::span<int4>(outputs[party], 2), proofs[party]);
    } else {
      scheme.BatchEval(party != 0, initial_seed, common_cws, common_cs,
          common_inputs, cuda::std::span<int4>(outputs[party], 2), proofs[party]);
    }
  }
  const int4 expected[2] = {
      {static_cast<int>(3665264785u), 94261333, 1337733932, 1969104572},
      {629702511, static_cast<int>(4200705962u), static_cast<int>(2957233363u),
          static_cast<int>(2325862722u)},
  };
  for (int i = 0; i < 2; ++i) {
    SCOPED_TRACE(::testing::Message() << "query index " << i);
    const int4 reconstructed =
        (Uint127Group::From(outputs[0][i]) + Uint127Group::From(outputs[1][i])).Into();
    ExpectMerkleBlock(reconstructed, expected[i]);
    ASSERT_NE(reconstructed.x | reconstructed.y | reconstructed.z | reconstructed.w, 0);
  }
  EXPECT_FALSE(Scheme::Verify(proofs[0], proofs[1]));
}

TEST(VdpfSoundness, FixedBit96CaseRejectsChain) {
  CheckFixedBit96CaseRejects(false);
}

TEST(VdpfSoundness, FixedBit96CaseRejectsMerkle) {
  CheckFixedBit96CaseRejects(true);
}

struct MerkleRangeExecution {
  const std::thread::id caller = std::this_thread::get_id();
  std::atomic<int> active{0};
  std::atomic<bool> invalid_context{false};
  std::atomic<size_t> calls{0};
  void Enter() {
    ++calls;
    if (active.fetch_add(1) != 0 || std::this_thread::get_id() != caller)
      invalid_context = true;
#ifdef _OPENMP
    if (omp_in_parallel()) invalid_context = true;
#endif
  }
};

struct MerkleRangeObservedPrg {
  fss::prg::ChaCha<2> prg;
  MerkleRangeExecution *execution;
  __host__ __device__ cuda::std::array<int4, 2> Gen(int4 seed) {
#ifndef __CUDA_ARCH__
    execution->Enter();
#endif
    auto result = prg.Gen(seed);
#ifndef __CUDA_ARCH__
    --execution->active;
#endif
    return result;
  }
};

struct MerkleRangeObservedHash {
  fss::hash::Blake3 hash;
  MerkleRangeExecution *execution;
  template <typename Input>
  __host__ __device__ auto Hash(Input input) {
#ifndef __CUDA_ARCH__
    execution->Enter();
#endif
    auto result = hash.Hash(input);
#ifndef __CUDA_ARCH__
    --execution->active;
#endif
    return result;
  }
};

TEST(VdpfRangeMerkle, CpuDefaultDepthUsesCallerForAllBackends) {
  using Scheme = fss::Vdpf<8, BytesGroup, MerkleRangeObservedPrg,
      MerkleRangeObservedHash, MerkleRangeObservedHash, uint16_t>;
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  MerkleRangeExecution prg_execution, xor_execution, hash_execution;
  MerkleRangeObservedPrg prg{fss::prg::ChaCha<2>{kNonce0, kNonce1}, &prg_execution};
  MerkleRangeObservedHash xor_hash{fss::hash::Blake3{iv}, &xor_execution};
  MerkleRangeObservedHash hash{fss::hash::Blake3{iv}, &hash_execution};
  Scheme scheme{prg, xor_hash, hash};
  typename Scheme::Key keys[2];
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  ASSERT_EQ(fss_test::GenPair(scheme, keys[0], keys[1], seeds, 42, int4{42, 12, 24, 36}), 0);
  std::vector<int4> ys(65);
  cuda::std::array<int4, 4> pi;
  scheme.EvalRangeMerkle(false, 3, 67, keys[0].s0, keys[0].cws, keys[0].cs, ys, pi);
  for (auto *execution : {&prg_execution, &xor_execution, &hash_execution}) {
    EXPECT_GT(execution->calls.load(), 0);
    EXPECT_EQ(execution->active.load(), 0);
    EXPECT_FALSE(execution->invalid_context.load());
  }
}

TEST(VdpfRangeMerkle, Hash4MatchesScalarFallbackForOddLeaves) {
  using Prg = fss::prg::ChaCha<2>;
  using BatchScheme = fss::Vdpf<8, BytesGroup, Prg, fss::hash::Blake3,
      fss::hash::Blake3, uint16_t>;
  using FallbackScheme = fss::Vdpf<8, BytesGroup, Prg, fss::hash::Blake3,
      MerkleRangeObservedHash, uint16_t>;
  static_assert(HasHash4<fss::hash::Blake3>);
  static_assert(!HasHash4<MerkleRangeObservedHash>);

  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  fss::hash::Blake3 blake{cuda::std::span<const int4, 2>(iv, 2)};
  MerkleRangeExecution execution;
  MerkleRangeObservedHash fallback{blake, &execution};
  BatchScheme batch{Prg{kNonce0, kNonce1}, blake, blake};
  FallbackScheme scalar{Prg{kNonce0, kNonce1}, blake, fallback};

  cuda::std::array<cuda::std::array<int4, 4>, 129> leaves;
  for (size_t i = 0; i < leaves.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      const int base = static_cast<int>(16 * i + 4 * j);
      leaves[i][j] = {base, -base - 1, base ^ 0x55aa55aa,
          static_cast<int>(static_cast<uint32_t>(base) ^ 0xaaaaaaaau)};
    }
  }
  const cuda::std::array<int4, 4> cs = {{{1, 2, 3, 4}, {-1, 0, 1, -1},
      {2147483647, -2147483647 - 1, 0, 1},
      {static_cast<int>(0x80000000u), -1, 1, 0}}};

  for (size_t count : {5, 129}) {
    cuda::std::array<int4, 4> batch_proof, fallback_proof;
    batch.ProveMerkle(
        cuda::std::span<const cuda::std::array<int4, 4>>(leaves.data(), count),
        cuda::std::span<const int4, 4>(cs.data(), 4),
        cuda::std::span<int4, 4>(batch_proof.data(), 4), 0);
    scalar.ProveMerkle(
        cuda::std::span<const cuda::std::array<int4, 4>>(leaves.data(), count),
        cuda::std::span<const int4, 4>(cs.data(), 4),
        cuda::std::span<int4, 4>(fallback_proof.data(), 4), 0);
    EXPECT_GT(execution.calls.load(), 0);
    EXPECT_FALSE(execution.invalid_context.load());
    for (int j = 0; j < 4; ++j) ExpectMerkleBlock(fallback_proof[j], batch_proof[j]);
  }
}

TEST(VdpfRangeMerkle, GpuRejectsInvalidRangeAndStorage) {
  using Prg = fss::prg::ChaCha<2>;
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<128, BytesGroup, Prg, Hash, Hash, __uint128_t>;
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Hash hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg{kNonce0, kNonce1}, hash, hash};
  typename Scheme::Cw cws[129];
  int4 cs[4], ys[4], pi[4];
  auto check = [&](auto first, auto last, size_t capacity) {
    return scheme.EvalRangeMerkleGpu(false, first, last, int4{}, cws, cs, {ys, capacity}, pi);
  };
  EXPECT_EQ(check(__uint128_t{2}, __uint128_t{1}, 4), cudaErrorInvalidValue);
  EXPECT_EQ(check(__uint128_t{0}, ~__uint128_t{0}, 4), cudaErrorInvalidValue);
  EXPECT_EQ(check(__uint128_t{0}, __uint128_t{4}, 4), cudaErrorInvalidValue);
  const size_t scratch_overflow = size_t(-1) / 32;
  EXPECT_EQ(check(__uint128_t{0}, static_cast<__uint128_t>(scratch_overflow - 1),
      scratch_overflow), cudaErrorInvalidValue);
  EXPECT_EQ(scheme.EvalRangeMerkleGpu(false, 0, 0, int4{}, cws, cs,
      cuda::std::span<int4>(static_cast<int4 *>(nullptr), size_t{1}), pi), cudaErrorInvalidValue);
  const size_t grid_overflow = static_cast<size_t>(INT_MAX) * 256 * 2 + 1;
  EXPECT_EQ(check(__uint128_t{0}, static_cast<__uint128_t>(grid_overflow - 1),
      grid_overflow), cudaErrorInvalidConfiguration);
  using Small = fss::Vdpf<4, BytesGroup, Prg, Hash, Hash, uint16_t>;
  Small small{Prg{kNonce0, kNonce1}, hash, hash};
  typename Small::Cw small_cws[5];
  EXPECT_EQ(small.EvalRangeMerkleGpu(false, 15, 16, int4{}, small_cws, cs, ys, pi), cudaErrorInvalidValue);
}

}  // namespace


template <typename Group, bool has_half>
void CheckVdpfGenHalf() {
  constexpr int bits = 5;
  using Backend = fss::prg::Aes128Soft<2>;
  using Prg = fss_test::CountedPrg<Backend, 2, has_half>;
  using FullPrg = fss_test::CountedPrg<Backend, 2, false>;
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<bits, Group, Prg, Hash, Hash, uint8_t>;
  using FullScheme = fss::Vdpf<bits, Group, FullPrg, Hash, Hash, uint8_t>;
  using Proof = cuda::std::array<int4, 4>;
  fss_test::GenHalfAesFixture<2> aes;
  fss_test::PrgCallCounts calls, full_calls;
  int4 iv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  Hash hash(cuda::std::span<const int4, 2>(iv, 2));
  Scheme scheme{Prg(aes.Backend(), calls), hash, hash};
  FullScheme reference{FullPrg(aes.Backend(), full_calls), hash, hash};
  int4 seeds[2] = {{1, 3, 5, 6}, {7, 9, 11, 12}};
  int4 beta = {42, 6, 8, 10};
  typename Scheme::Key keys[2];
  typename FullScheme::Key full_keys[2];
  ASSERT_EQ(fss_test::GenPair(scheme, keys[0], keys[1], seeds, uint8_t{13}, beta), 0);
  ASSERT_EQ(fss_test::GenPair(reference, full_keys[0], full_keys[1], seeds, uint8_t{13}, beta), 0);
  EXPECT_EQ(calls.full, 2 * bits);
  EXPECT_EQ(calls.half, 0);
  Proof points[2][32], full_points[2][32];
  for (uint8_t x = 0; x < 32; ++x) {
    int4 ys[2];
    for (int b = 0; b < 2; ++b) {
      calls = {};
      auto result = fss_test::EvalKey(scheme, b, keys[b], x);
      auto expected = fss_test::EvalKey(reference, b, full_keys[b], x);
      ys[b] = result.y;
      points[b][x] = result.pi_tilde;
      full_points[b][x] = expected.pi_tilde;
      EXPECT_EQ(memcmp(&result.y, &expected.y, sizeof(int4)), 0) << "party=" << b << " x=" << +x;
      EXPECT_EQ(memcmp(result.pi_tilde.data(), expected.pi_tilde.data(), sizeof(Proof)), 0);
      EXPECT_EQ(calls.full, has_half ? 0 : bits);
      EXPECT_EQ(calls.half, has_half ? bits : 0);
      EXPECT_EQ(calls.right, has_half ? __builtin_popcount(x) : 0);
    }
    int4 result = (Group::From(ys[0]) + Group::From(ys[1])).Into();
    int4 expected = x == 13 ? Group::From(beta).Into() : Group{}.Into();
    EXPECT_EQ(memcmp(&result, &expected, sizeof(int4)), 0);
  }
  Proof proofs[2], full_proofs[2];
  for (int b = 0; b < 2; ++b) {
    scheme.Prove(cuda::std::span<const Proof>(points[b], 32), keys[b].cs, proofs[b]);
    reference.Prove(cuda::std::span<const Proof>(full_points[b], 32), full_keys[b].cs, full_proofs[b]);
    EXPECT_EQ(memcmp(proofs[b].data(), full_proofs[b].data(), sizeof(Proof)), 0);
  }
  EXPECT_TRUE(Scheme::Verify(proofs[0], proofs[1]));
}

TEST(VdpfGenHalf, BytesSelectedAndFallbackProofs) {
  CheckVdpfGenHalf<BytesGroup, true>();
  CheckVdpfGenHalf<BytesGroup, false>();
}
TEST(VdpfGenHalf, Uint127SelectedAndFallbackProofs) {
  CheckVdpfGenHalf<Uint127Group, true>();
  CheckVdpfGenHalf<Uint127Group, false>();
}
TEST(VdpfGenHalf, GenericGroupSelectedAndFallbackProofs) {
  using Group = fss::group::Uint<uint64_t>;
  CheckVdpfGenHalf<Group, true>();
  CheckVdpfGenHalf<Group, false>();
}


TEST(VdpfGenHalf, GpuAesSoftSelectedRangeProof) {
  if (!HasMerkleRangeDevice()) GTEST_SKIP() << "no CUDA device available";
  fss_test::GenHalfDeviceAesFixture<2> aes;
  ASSERT_EQ(aes.Allocate(), cudaSuccess);
  CheckRangeMerkle<16, uint16_t, BytesGroup, 8, 64>(3, 17, true, aes.Backend());
}

namespace {

// Checks the split Gen and Eval against the Cw form on the same seeds.
template <int bits, typename In, typename Group>
void CheckVdpfSplitForm(In alpha) {
  using Prg = fss::prg::ChaCha<2>;
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<bits, Group, Prg, Hash, Hash, In>;
  using Cw = typename Scheme::Cw;
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Hash hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg(kNonce0, kNonce1), hash, hash};
  const int4 beta = {7, 19, 23, 42};
  const int4 guard = {23, 45, 67, 88};
  In maximum = static_cast<In>(~In{0});
  if constexpr (bits < sizeof(In) * 8) maximum = static_cast<In>((In{1} << bits) - 1);
  const In xs[4] = {alpha, In{0}, maximum, static_cast<In>(alpha == maximum ? maximum - 1 : alpha + 1)};

  for (size_t stride : {size_t{1}, size_t{7}}) {
    SCOPED_TRACE(::testing::Message() << "stride " << stride);
    const size_t extent = static_cast<size_t>(bits) * stride + 1;
    Cw cws[bits + 1];
    cuda::std::array<int4, 4> cs, split_cs;
    std::vector<int4> split(extent, guard);
    In split_tr = 0;
    int4 seeds[2] = {{0, 22, 33, 44}, {55, 66, 77, 88}};
    int rc = 1;
    for (int attempt = 0; attempt < 64; ++attempt) {
      seeds[0].x = attempt;
      const int cw_rc = scheme.Gen(cws, cuda::std::span<int4, 4>(cs.data(), 4),
          cuda::std::span<const int4, 2>(seeds, 2), alpha, beta);
      const int split_rc = scheme.Gen(cuda::std::span<int4>(split.data(), extent), split_tr,
          cuda::std::span<int4, 4>(split_cs.data(), 4), cuda::std::span<const int4, 2>(seeds, 2), alpha, beta, stride);
      // Both forms share the generation core, so they agree on the retry decision too.
      EXPECT_EQ(split_rc, cw_rc) << "attempt " << attempt;
      rc = cw_rc;
      if (rc == 0) break;
    }
    ASSERT_EQ(rc, 0);

    for (int i = 0; i < 4; ++i) EXPECT_EQ(memcmp(&split_cs[i], &cs[i], sizeof(int4)), 0) << "cs block " << i;
    for (int i = 0; i < bits; ++i) {
      EXPECT_EQ(memcmp(&split[static_cast<size_t>(i) * stride], &cws[i].s, sizeof(int4)), 0) << "layer " << i;
      EXPECT_EQ(((split_tr >> i) & 1) != 0, cws[i].tr) << "layer " << i;
    }
    EXPECT_EQ(memcmp(&split[static_cast<size_t>(bits) * stride], &cws[bits].s, sizeof(int4)), 0) << "terminal";
    if constexpr (bits < sizeof(In) * 8) {
      EXPECT_EQ(static_cast<In>(split_tr >> bits), In{0}) << "unused control bits";
    }
    for (size_t i = 0; i < extent; ++i) {
      const bool written = i % stride == 0 && i / stride <= static_cast<size_t>(bits);
      if (!written) EXPECT_EQ(memcmp(&split[i], &guard, sizeof(int4)), 0) << "unwritten index " << i;
    }

    for (int party = 0; party < 2; ++party) {
      for (In x : xs) {
        SCOPED_TRACE(::testing::Message() << "party " << party << " x " << static_cast<uint64_t>(x));
        int4 y_cw, y_split;
        const auto hash_cw = scheme.Eval(party != 0, seeds[party], cws, cuda::std::span<const int4, 4>(cs.data(), 4),
            x, y_cw);
        const auto hash_split = scheme.Eval(party != 0, seeds[party], cuda::std::span<const int4>(split.data(), extent),
            split_tr, cuda::std::span<const int4, 4>(split_cs.data(), 4), x, y_split, stride);
        EXPECT_EQ(memcmp(&y_split, &y_cw, sizeof(int4)), 0);
        EXPECT_EQ(memcmp(hash_split.data(), hash_cw.data(), sizeof(cuda::std::array<int4, 4>)), 0);
      }
    }
  }
}

}  // namespace

TEST(VdpfSplitForm, OneBitUint8) {
  CheckVdpfSplitForm<1, uint8_t, BytesGroup>(1);
}
TEST(VdpfSplitForm, MiddleBitsUint16) {
  CheckVdpfSplitForm<16, uint16_t, BytesGroup>(kAlpha);
}
TEST(VdpfSplitForm, MiddleBitsUint32) {
  CheckVdpfSplitForm<20, uint32_t, BytesGroup>(0xA5A5A);
}
TEST(VdpfSplitForm, WideUint128) {
  CheckVdpfSplitForm<128, __uint128_t, BytesGroup>(static_cast<__uint128_t>(1) << 127);
}
TEST(VdpfSplitForm, WideUint128Uint127Group) {
  CheckVdpfSplitForm<128, __uint128_t, Uint127Group>((static_cast<__uint128_t>(1) << 127) + 7);
}

TEST(VdpfPreconditionTest, RejectsInvalidHostInputs) {
  using Scheme = fss::Vdpf<8, BytesGroup, fss::prg::ChaCha<2>, fss::hash::Blake3,
      fss::hash::Blake3, uint16_t>;
  const int4 iv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const cuda::std::span<const int4, 2> iv_span(iv, 2);
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1), fss::hash::Blake3(iv_span),
      fss::hash::Blake3(iv_span)};
  constexpr int kInBits = 8;
  std::vector<typename Scheme::Cw> cws(kInBits + 1);
  int4 cs[4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
  const auto cs_span = cuda::std::span<const int4, 4>(cs, 4);
  std::vector<int4> ys(size_t{1} << kInBits);
  cuda::std::array<int4, 4> pi{};
  const int4 s0 = {0, 0, 0, 0};

  EXPECT_THROW((void)scheme.EvalRange(false, 3, 2, s0, cuda::std::span<const typename Scheme::Cw>(cws),
                        cs_span, cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalRange(false, 0, uint16_t{300}, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                        cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalRange(false, 0, 7, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws.data(), 1), cs_span,
                        cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalRange(false, 0, 7, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                        cuda::std::span<int4>(ys.data(), 7), pi),
      std::invalid_argument);

  std::vector<uint16_t> xs(1, 0);
  EXPECT_THROW((void)scheme.BatchEval(false, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws.data(), 1), cs_span,
                        cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.BatchEval(false, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                        cuda::std::span<const uint16_t>(), cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.BatchEval(false, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                        cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.BatchEvalMerkle(false, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws.data(), 1), cs_span,
                        cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.BatchEvalMerkle(false, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                        cuda::std::span<const uint16_t>(), cuda::std::span<int4>(ys), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.BatchEvalMerkle(false, s0,
                        cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                        cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(), pi),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.ProveMerkle(
                        cuda::std::span<const cuda::std::array<int4, 4>>(), cs_span, pi),
      std::invalid_argument);
}

// The terminal LSB is a caller precondition of the range and full-domain
// entries, unlike the buffer checks above: Debug builds abort on the calling
// thread, and Release builds do not check it.
#ifndef NDEBUG
TEST(VdpfPreconditionTest, AbortsOnBadTerminalInDebug) {
  using Scheme = fss::Vdpf<8, BytesGroup, fss::prg::ChaCha<2>, fss::hash::Blake3,
      fss::hash::Blake3, uint16_t>;
  const int4 iv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const cuda::std::span<const int4, 2> iv_span(iv, 2);
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1), fss::hash::Blake3(iv_span),
      fss::hash::Blake3(iv_span)};
  std::vector<typename Scheme::Cw> cws(9);
  int4 cs[4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
  const auto cs_span = cuda::std::span<const int4, 4>(cs, 4);
  std::vector<int4> ys(size_t{1} << 8);
  cuda::std::array<int4, 4> pi{};
  const int4 s0 = {0, 0, 0, 0};
  cws[8].s.w = 1;
  EXPECT_DEATH((void)scheme.EvalRange(false, 0, uint16_t{7}, s0,
                       cuda::std::span<const typename Scheme::Cw>(cws), cs_span,
                       cuda::std::span<int4>(ys), pi),
      "Assertion.*failed");
  EXPECT_DEATH((void)scheme.EvalAll(false, s0, cuda::std::span<const typename Scheme::Cw>(cws),
                       cs_span, cuda::std::span<int4>(ys), pi),
      "Assertion.*failed");
  EXPECT_DEATH((void)scheme.EvalAllMerkle(false, s0,
                       cuda::std::span<const typename Scheme::Cw>(cws),
                       cs_span, cuda::std::span<int4>(ys), pi),
      "Assertion.*failed");
}
#endif  // NDEBUG

TEST(VdpfPreconditionTest, GenKeyStillEvaluatesAndVerifies) {
  using Scheme = fss::Vdpf<1, BytesGroup, fss::prg::ChaCha<2>, fss::hash::Blake3,
      fss::hash::Blake3, uint8_t>;
  const int4 iv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};
  const cuda::std::span<const int4, 2> iv_span(iv, 2);
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1), fss::hash::Blake3(iv_span),
      fss::hash::Blake3(iv_span)};
  Scheme::Cw cws[2];
  int4 cs[4];
  int4 seeds[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  constexpr uint8_t kAlpha = 1;
  const int4 kBeta = {1, 0, 0, 0};
  int rc = 1;
  for (int attempt = 0; attempt < 64; ++attempt) {
    seeds[0].x = attempt;
    rc = scheme.Gen(cws, cuda::std::span<int4, 4>(cs, 4),
        cuda::std::span<const int4, 2>(seeds, 2), kAlpha, kBeta);
    if (rc == 0) break;
  }
  ASSERT_EQ(rc, 0);

  const auto cws_span = cuda::std::span<const Scheme::Cw>(cws, 2);
  const auto cs_span = cuda::std::span<const int4, 4>(cs, 4);
  int4 shares[2][2];
  cuda::std::array<int4, 4> pi_tildes[2][2];
  cuda::std::array<int4, 4> chain[2], merkle[2];
  for (int party = 0; party < 2; ++party) {
    for (int x = 0; x < 2; ++x) {
      pi_tildes[party][x] = scheme.Eval(party != 0, seeds[party], cws_span, cs_span,
          static_cast<uint8_t>(x), shares[party][x]);
    }
    const auto leaves = cuda::std::span<const cuda::std::array<int4, 4>>(pi_tildes[party], 2);
    scheme.Prove(leaves, cs_span, cuda::std::span<int4, 4>(chain[party].data(), 4));
    scheme.ProveMerkle(leaves, cs_span, cuda::std::span<int4, 4>(merkle[party].data(), 4));
  }
  EXPECT_TRUE(Scheme::Verify(cuda::std::span<const int4, 4>(chain[0].data(), 4),
      cuda::std::span<const int4, 4>(chain[1].data(), 4)));
  EXPECT_TRUE(Scheme::Verify(cuda::std::span<const int4, 4>(merkle[0].data(), 4),
      cuda::std::span<const int4, 4>(merkle[1].data(), 4)));
  const int4 expected[2] = {{0, 0, 0, 0}, kBeta};
  for (int x = 0; x < 2; ++x) {
    const int4 reconstructed =
        (BytesGroup::From(shares[0][x]) + BytesGroup::From(shares[1][x])).Into();
    EXPECT_EQ(memcmp(&reconstructed, &expected[x], sizeof(int4)), 0);
  }
}
