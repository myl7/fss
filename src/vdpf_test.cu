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
