#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include "core_api_test_helpers.cuh"
#include <fss/dmpf.cuh>
#include <fss/prp/chacha_feistel.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

namespace {

// Split at the largest complete left subtree. This recursive reference does
// not depend on a chunk size or on the production level-by-level reducer.
template <typename Hash>
cuda::std::array<int4, 2> ReferenceMerkleRoot(
    Hash &hash, cuda::std::span<const cuda::std::array<int4, 2>> leaves) {
  if (leaves.size() == 1) return leaves[0];
  size_t left_size = 1;
  while (left_size * 2 < leaves.size()) left_size *= 2;
  auto left = ReferenceMerkleRoot(hash,
      cuda::std::span<const cuda::std::array<int4, 2>>(leaves.data(), left_size));
  auto right = ReferenceMerkleRoot(hash,
      cuda::std::span<const cuda::std::array<int4, 2>>(leaves.data() + left_size, leaves.size() - left_size));
  int4 msg[4] = {left[0], left[1], right[0], right[1]};
  return hash.Hash(cuda::std::span<const int4, 4>(msg));
}

}  // namespace

template <typename Group>
class VdmpfChaChaTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using XorHash = fss::hash::Blake3;
  using Hash = fss::hash::Blake3;
  using Prp = fss::prp::Aes128Feistel;
  using VdmpfType = fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, XorHash, Hash, Prp, uint16_t>;

  static constexpr int m = VdmpfType::kBucketCount;

  uint16_t alphas[kT] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
      1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000};
  int4 betas[kT];

  Prg prg;
  int4 hash_iv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};
  XorHash xor_hash;
  Hash hash_;
  Prp prp_;

  VdmpfChaChaTest()
      : prg(kNonce0, kNonce1),
        xor_hash(cuda::std::span<const int4, 2>(hash_iv, 2)),
        hash_(cuda::std::span<const int4, 2>(hash_iv, 2)) {}

  void SetUp() override {
    for (int i = 0; i < kT; ++i) {
      betas[i] = {(i + 1) * 11, 0, 0, 0};
    }
  }

  int4 RandomSeed(std::mt19937 &gen) {
    std::uniform_int_distribution<int> dis;
    return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void GenKeys(typename VdmpfType::Key &k0, typename VdmpfType::Key &k1) {
    VdmpfType vdmpf{prg, xor_hash, hash_, prp_};
    std::random_device rd;
    std::mt19937 gen(rd());

    int4 sigma = RandomSeed(gen);
    cuda::std::array<int4, size_t{m} * 2> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = RandomSeed(gen);
      s0s[size_t{m} + i] = RandomSeed(gen);
    }
    ASSERT_EQ(fss_test::GenMulti(vdmpf, k0, k1, sigma, cuda::std::span<const int4>(s0s),
        cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT, 1000), 0);
  }

  void TestEvalAtAlpha() {
    typename VdmpfType::Key k0, k1;
    GenKeys(k0, k1);
    VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

    std::vector<uint16_t> xs(alphas, alphas + kT);
    std::vector<int4> ys0(kT), ys1(kT);
    cuda::std::array<int4, 4> pi0, pi1;

    fss_test::BatchEvalKey(vdmpf, false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0), pi0);
    fss_test::BatchEvalKey(vdmpf, true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1), pi1);

    for (int i = 0; i < kT; ++i) {
      auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
      int4 r = result.Into();
      int4 e = betas[i];
      e.w &= ~1;
      EXPECT_EQ(memcmp(&r, &e, sizeof(int4)), 0) << "Failed at alpha=" << alphas[i];
    }
  }

  void TestEvalAtNonAlpha() {
    typename VdmpfType::Key k0, k1;
    GenKeys(k0, k1);
    VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    constexpr int kNumTrials = 100;
    std::vector<uint16_t> xs;
    xs.reserve(kNumTrials);
    for (int i = 0; i < kNumTrials; ++i) {
      uint16_t x = dis(gen);
      bool is_alpha = false;
      for (int j = 0; j < kT; ++j) {
        if (x == alphas[j]) {
          is_alpha = true;
          break;
        }
      }
      if (is_alpha) {
        --i;
        continue;
      }
      xs.push_back(x);
    }

    std::vector<int4> ys0(xs.size()), ys1(xs.size());
    cuda::std::array<int4, 4> pi0, pi1;

    fss_test::BatchEvalKey(vdmpf, false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0), pi0);
    fss_test::BatchEvalKey(vdmpf, true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1), pi1);

    int4 zero = {0, 0, 0, 0};
    for (size_t i = 0; i < xs.size(); ++i) {
      auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
      int4 r = result.Into();
      EXPECT_EQ(memcmp(&r, &zero, sizeof(int4)), 0) << "Failed at x=" << xs[i];
    }
  }

  void TestVerifyBatchEval() {
    typename VdmpfType::Key k0, k1;
    GenKeys(k0, k1);
    VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    // Mix alpha and non-alpha points.
    std::vector<uint16_t> xs(alphas, alphas + kT);
    for (int i = 0; i < 50; ++i) {
      xs.push_back(dis(gen));
    }

    std::vector<int4> ys0(xs.size()), ys1(xs.size());
    cuda::std::array<int4, 4> pi0, pi1;

    fss_test::BatchEvalKey(vdmpf, false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0), pi0);
    fss_test::BatchEvalKey(vdmpf, true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1), pi1);

    EXPECT_TRUE(VdmpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
  }

  void TestBatchEvalDuplicates() {
    typename VdmpfType::Key k0, k1;
    GenKeys(k0, k1);
    VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

    // Duplicated inputs: every alpha and 50 random points appear multiple times,
    // so several omegas locate the same within-bucket index and the 3 * m located
    // records crowd the m buckets.
    std::mt19937 gen(20261017);
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);
    std::vector<uint16_t> xs;
    for (int rep = 0; rep < 3; ++rep) {
      for (int i = 0; i < kT; ++i) xs.push_back(alphas[i]);
    }
    for (int i = 0; i < 100; ++i) xs.push_back(dis(gen));
    for (int i = 0; i < 50; ++i) xs.push_back(xs[3 * kT + i]);

    std::vector<int4> ys0(xs.size()), ys1(xs.size());
    cuda::std::array<int4, 4> pi0, pi1;

    fss_test::BatchEvalKey(vdmpf, false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0), pi0);
    fss_test::BatchEvalKey(vdmpf, true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1), pi1);

    EXPECT_TRUE(VdmpfType::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));

    // Scalar reference: every duplicated position of an x yields the same value,
    // beta_j when x is an alpha, 0 otherwise.
    for (size_t i = 0; i < xs.size(); ++i) {
      int4 e = {0, 0, 0, 0};
      for (int j = 0; j < kT; ++j) {
        if (xs[i] == alphas[j]) {
          e = betas[j];
          break;
        }
      }
      e.w &= ~1;
      auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
      int4 r = result.Into();
      EXPECT_EQ(memcmp(&r, &e, sizeof(int4)), 0) << "Failed at x=" << xs[i] << " pos=" << i;
    }
  }

  void TestCanonicalMerkleReference() {
    typename VdmpfType::Key k0, k1;
    GenKeys(k0, k1);
    VdmpfType scheme{prg, xor_hash, hash_, prp_};
    using Inner = fss::Vdpf<kBucketBits, Group, Prg, XorHash, Hash, __uint128_t>;
    Inner inner{prg, xor_hash, hash_};
    fss::cuckoo_hash::detail::PrpHash<Prp, uint16_t> locate{prp_};
    using Node = cuda::std::array<int4, 2>;
    for (int count : {0, 1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257, 513}) {
      SCOPED_TRACE(::testing::Message() << "queries=" << count);
      // Repetition puts a long, ragged sequence in the same candidate buckets.
      std::vector<uint16_t> xs(count, alphas[0]);
      std::vector<int4> actual_ys[2];
      cuda::std::array<int4, 4> actual_pi[2];
      for (int party = 0; party < 2; ++party) {
        SCOPED_TRACE(::testing::Message() << "party=" << party);
        const auto &key = party == 0 ? k0 : k1;
        actual_ys[party].resize(count);
        fss_test::BatchEvalMerkleKey(scheme, party != 0, key,
            cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(actual_ys[party]), actual_pi[party]);

        std::vector<std::vector<Node>> leaves(m);
        std::vector<int4> reference_ys(count, int4{0, 0, 0, 0});
        for (int omega = 0; omega < count; ++omega) {
          for (int candidate = 0; candidate < 3; ++candidate) {
            auto [bucket, index] = locate.Locate(
                key.sigma, xs[omega], candidate, VdmpfType::kDomainSize, key.b_size_rt);
            if (bucket >= key.m_rt) continue;
            int4 share;
            const auto &bucket_key = key.bks[bucket];
            auto corrected = inner.Eval(party != 0, bucket_key.s0, bucket_key.cws,
                bucket_key.cs, index, share);
            reference_ys[omega] = (Group::From(reference_ys[omega]) + Group::From(share)).Into();
            leaves[bucket].push_back(hash_.Hash(cuda::std::span<const int4, 4>(corrected)));
          }
        }
        std::vector<Node> bucket_nodes(m);
        for (int bucket = 0; bucket < m; ++bucket) {
          Node anchor = hash_.Hash(cuda::std::span<const int4, 4>(key.bks[bucket].cs));
          bucket_nodes[bucket] = anchor;
          if (!leaves[bucket].empty()) {
            auto root = ReferenceMerkleRoot(hash_, cuda::std::span<const Node>(leaves[bucket]));
            int4 msg[4] = {anchor[0], anchor[1], root[0], root[1]};
            bucket_nodes[bucket] = hash_.Hash(cuda::std::span<const int4, 4>(msg));
          }
        }
        auto root = ReferenceMerkleRoot(hash_, cuda::std::span<const Node>(bucket_nodes));
        int4 reference_pi[4] = {root[0], root[1], int4{0, 0, 0, 0}, int4{0, 0, 0, 0}};
        for (int i = 0; i < count; ++i) {
          EXPECT_EQ(memcmp(&actual_ys[party][i], &reference_ys[i], sizeof(int4)), 0) << "query=" << i;
        }
        for (int word = 0; word < 4; ++word) {
          EXPECT_EQ(memcmp(&actual_pi[party][word], &reference_pi[word], sizeof(int4)), 0)
              << "proof word=" << word;
        }
      }
      EXPECT_TRUE(VdmpfType::Verify(actual_pi[0], actual_pi[1]));
      for (int i = 0; i < count; ++i) {
        int4 reconstructed = (Group::From(actual_ys[0][i]) + Group::From(actual_ys[1][i])).Into();
        int4 expected = betas[0];
        expected.w &= ~1;
        EXPECT_EQ(memcmp(&reconstructed, &expected, sizeof(int4)), 0);
      }
    }
  }
};

using VdmpfBytesChaChaTest = VdmpfChaChaTest<BytesGroup>;
using VdmpfUint128ChaChaTest = VdmpfChaChaTest<Uint127Group>;

TEST_F(VdmpfBytesChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(VdmpfBytesChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(VdmpfBytesChaChaTest, VerifyBatchEval) {
  TestVerifyBatchEval();
}

TEST_F(VdmpfBytesChaChaTest, CanonicalMerkleMatchesRecursiveReference) {
  TestCanonicalMerkleReference();
}
TEST_F(VdmpfBytesChaChaTest, BatchEvalDuplicates) {
  TestBatchEvalDuplicates();
}

TEST_F(VdmpfUint128ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(VdmpfUint128ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(VdmpfUint128ChaChaTest, VerifyBatchEval) {
  TestVerifyBatchEval();
}
TEST_F(VdmpfUint128ChaChaTest, BatchEvalDuplicates) {
  TestBatchEvalDuplicates();
}

TEST(VdmpfCuckooHashTest, Compact) {
  fss::prp::Aes128Feistel prp;
  constexpr int t = 30;
  uint16_t as[t];
  for (int i = 0; i < t; ++i) as[i] = static_cast<uint16_t>(i * 100 + 10);
  int m = fss::cuckoo_hash::ChBucket(t, 80);
  std::vector<std::pair<int, int>> table(m, {-1, -1});

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis;
  int4 sigma;
  int ret;
  constexpr __uint128_t n = __uint128_t{1} << 16;
  int B = static_cast<int>((n * 3 + m - 1) / m);
  do {
    sigma = {dis(gen), dis(gen), dis(gen), dis(gen)};
    std::fill(table.begin(), table.end(), std::pair<int, int>{-1, -1});
    fss::cuckoo_hash::detail::Compact<fss::prp::Aes128Feistel, uint16_t> compact{prp};
    ret = compact.Run(std::span<const uint16_t>(as, t), m, sigma, n, B, 1000, std::span<std::pair<int, int>>(table));
  } while (ret != 0);

  // Each alpha should appear exactly once in the table.
  int count = 0;
  for (int i = 0; i < m; ++i) {
    if (table[i].first != -1) count++;
  }
  EXPECT_EQ(count, t);
}

namespace {
namespace multi_point_domain {
using Group = fss::group::Bytes;
using Prg = fss::prg::ChaCha<2>;
using Prp = fss::prp::ChaChaFeistel;
using ProofHash = fss::hash::Blake3;
ProofHash MakeProofHash() {
  const int4 iv[2] = {{2, 3, 5, 7}, {11, 13, 17, 19}};
  return ProofHash{iv};
}
template <int bits, int bucket_bits>
concept DmpfDomain = requires {
  typename fss::Dmpf<bits, 30, bucket_bits, Group, Prg, Prp, __uint128_t>;
};

template <int bits, int bucket_bits>
concept VdmpfDomain = requires {
  typename fss::Vdmpf<bits, 30, bucket_bits, Group, Prg, ProofHash, ProofHash, Prp, __uint128_t>;
};

static_assert(DmpfDomain<126, 127> && VdmpfDomain<126, 127>);
static_assert(!DmpfDomain<127, 127> && !VdmpfDomain<127, 127>);
static_assert(!DmpfDomain<0, 127> && !VdmpfDomain<0, 127>);
static_assert(!DmpfDomain<126, 30> && !VdmpfDomain<126, 30>);
void ExpectBlock(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

TEST(MultiPointEvaluation, WideDomainSharesAndProofs) {
  using Dmpf = fss::Dmpf<126, 30, 123, Group, Prg, Prp, __uint128_t>;
  using Vdmpf = fss::Vdmpf<126, 30, 123, Group, Prg, ProofHash, ProofHash, Prp, __uint128_t>;
  Dmpf dmpf{Prg{23, 47}, Prp{}};
  Vdmpf vdmpf{Prg{23, 47}, MakeProofHash(), MakeProofHash(), Prp{}};
  Dmpf::Key dk0, dk1;
  Vdmpf::Key vk0, vk1;
  int4 seeds[size_t{Dmpf::kBucketCount} * 2];
  for (int i = 0; i < Dmpf::kBucketCount; ++i) {
    seeds[i] = {i + 1, i + 2, i + 3, 2 * i};
    seeds[size_t{Dmpf::kBucketCount} + i] = {i + 5, i + 7, i + 11, 2 * i + 2};
  }
  __uint128_t alphas[30];
  int4 betas[30];
  for (int i = 0; i < 30; ++i) {
    alphas[i] = (__uint128_t{1} << 125) + i + 1;
    betas[i] = {i + 1, i + 2, i + 3, 2 * i + 4};
  }
  int4 sigma = {3, 5, 7, 11};
  ASSERT_EQ(fss_test::GenMulti(dmpf, dk0, dk1, sigma, seeds, alphas, betas, 30), 0);
  EXPECT_GT(dk0.b_size_rt, __uint128_t{1} << 64);
  int4 y0[30], y1[30];
  fss_test::BatchEvalKey(dmpf, false, dk0, alphas, y0);
  fss_test::BatchEvalKey(dmpf, true, dk1, alphas, y1);
  for (int i = 0; i < 30; ++i) ExpectBlock(fss::util::Xor(y0[i], y1[i]), betas[i]);
  ASSERT_EQ(fss_test::GenMulti(vdmpf, vk0, vk1, sigma, seeds, alphas, betas, 30, 1000), 0);
  cuda::std::array<int4, 4> pi0, pi1;
  fss_test::BatchEvalKey(vdmpf, false, vk0, alphas, y0, pi0);
  fss_test::BatchEvalKey(vdmpf, true, vk1, alphas, y1, pi1);
  EXPECT_TRUE(Vdmpf::Verify(pi0, pi1));
  for (int i = 0; i < 30; ++i) ExpectBlock(fss::util::Xor(y0[i], y1[i]), betas[i]);
  fss_test::BatchEvalMerkleKey(vdmpf, false, vk0, alphas, y0, pi0);
  fss_test::BatchEvalMerkleKey(vdmpf, true, vk1, alphas, y1, pi1);
  EXPECT_TRUE(Vdmpf::Verify(pi0, pi1));
  for (int i = 0; i < 30; ++i) ExpectBlock(fss::util::Xor(y0[i], y1[i]), betas[i]);
}
struct FailingPrg {
  __host__ __device__ cuda::std::array<int4, 2> Gen(int4) { return {}; }
};

struct ZeroVerificationHash {
  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4>) {
    return {};
  }
};

TEST(VdmpfGeneration, InnerVdpfFailurePreservesInputs) {
  using Scheme = fss::Vdmpf<8, 30, 4, Group, FailingPrg, ZeroVerificationHash, ProofHash, Prp, uint16_t>;
  Scheme scheme{FailingPrg{}, ZeroVerificationHash{}, MakeProofHash(), Prp{}};
  Scheme::Cw corrections[size_t{Scheme::kBucketCount} * 5];
  int4 checks[size_t{Scheme::kBucketCount} * 4];
  auto cws = cuda::std::span<Scheme::Cw>(corrections);
  auto cs = cuda::std::span<int4>(checks);
  int4 seeds[size_t{Scheme::kBucketCount} * 2];
  for (int i = 0; i < Scheme::kBucketCount; ++i) {
    seeds[i] = {11, 0, 0, 0};
    seeds[size_t{Scheme::kBucketCount} + i] = {0, 17, 0, 0};
  }
  uint16_t alphas[30];
  int4 betas[30];
  for (int i = 0; i < 30; ++i) {
    alphas[i] = i * 7;
    betas[i] = {2, 0, 0, 0};
  }
  int4 sigma = {3, 5, 7, 11};
  int m_rt = 0;
  __uint128_t b_size_rt = 0;
  EXPECT_EQ(scheme.Gen(cws, cs, m_rt, b_size_rt, sigma, seeds, alphas, betas, 30, 1000), 2);
  for (int i = 0; i < Scheme::kBucketCount; ++i) {
    ExpectBlock(seeds[i], int4{11, 0, 0, 0});
    ExpectBlock(seeds[size_t{Scheme::kBucketCount} + i], int4{0, 17, 0, 0});
  }
  ExpectBlock(sigma, int4{3, 5, 7, 11});
}
}  // namespace multi_point_domain
}  // namespace
