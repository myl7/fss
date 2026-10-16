#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <fss/dmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include <fss/prp/aes128_feistel_raw.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

namespace {

int4 FixedSigma(int tweak = 0) {
  int4 sigma = {0x0f0e0d0c, 0x0b0a0908, 0x07060504, static_cast<int>(0x03020100u)};
  sigma.x ^= tweak;
  return sigma;
}

// Aes128FeistelRaw must reproduce Aes128Feistel bit-for-bit on every input.
void TestBitIdentityExhaustive(__uint128_t domain, int4 sigma) {
  fss::prp::Aes128Feistel ref;
  fss::prp::Aes128FeistelRaw raw;
  for (__uint128_t x = 0; x < domain; ++x) {
    ASSERT_EQ(raw.Permu(sigma, x, domain), ref.Permu(sigma, x, domain))
        << "mismatch at x=" << static_cast<uint64_t>(x) << " domain=" << static_cast<uint64_t>(domain)
        << " tweak=" << (sigma.x & 0xff);
  }
}

}  // namespace

TEST(Aes128FeistelRawPrpTest, BitIdentityExhaustiveSmallDomains) {
  for (int tweak = 0; tweak < 4; ++tweak) {
    TestBitIdentityExhaustive(2, FixedSigma(tweak));
    TestBitIdentityExhaustive(3, FixedSigma(tweak));
    TestBitIdentityExhaustive(4, FixedSigma(tweak));
    TestBitIdentityExhaustive(5, FixedSigma(tweak));
    TestBitIdentityExhaustive(1000, FixedSigma(tweak));
    TestBitIdentityExhaustive(1000003, FixedSigma(tweak));
    TestBitIdentityExhaustive(__uint128_t{1} << 16, FixedSigma(tweak));
  }
}

TEST(Aes128FeistelRawPrpTest, BitIdentitySampled128BitDomain) {
  fss::prp::Aes128Feistel ref;
  fss::prp::Aes128FeistelRaw raw;
  const __uint128_t domain = (__uint128_t{1} << 100) + 12345;
  constexpr int kSamples = 1 << 20;
  for (int tweak : {0, 1}) {
    int4 sigma = FixedSigma(tweak);
    for (int i = 0; i < kSamples; ++i) {
      // Stride 7919 keeps the sampled inputs spread over the 101-bit domain.
      __uint128_t x = (static_cast<__uint128_t>(i) * 7919 + 13) % domain;
      ASSERT_EQ(raw.Permu(sigma, x, domain), ref.Permu(sigma, x, domain)) << "mismatch at i=" << i << " tweak=" << tweak;
    }
  }
}

TEST(Aes128FeistelRawPrpTest, DeterminismAndSeedVariation) {
  fss::prp::Aes128Feistel ref;
  fss::prp::Aes128FeistelRaw raw;
  const __uint128_t domain = 1000003;
  int4 sigma0 = FixedSigma();
  int4 sigma1 = FixedSigma(1);

  int differing = 0;
  constexpr int kSamples = 1000;
  for (int i = 0; i < kSamples; ++i) {
    __uint128_t x = static_cast<__uint128_t>(i * 7919 + 13) % domain;
    __uint128_t y0a = raw.Permu(sigma0, x, domain);
    __uint128_t y0b = raw.Permu(sigma0, x, domain);
    EXPECT_EQ(y0a, y0b) << "nondeterministic at x=" << i;
    EXPECT_EQ(y0a, ref.Permu(sigma0, x, domain)) << "reference mismatch at x=" << i;
    __uint128_t y1 = raw.Permu(sigma1, x, domain);
    if (y0a != y1) ++differing;
  }
  // Distinct keys must give essentially distinct permutations.
  EXPECT_GT(differing, kSamples * 9 / 10);
}

TEST(Aes128FeistelRawPrpTest, SeedSwitchingAndPreloadedInstance) {
  fss::prp::Aes128Feistel ref;
  fss::prp::Aes128FeistelRaw raw;
  fss::prp::Aes128FeistelRaw preloaded{FixedSigma(0)};
  const __uint128_t domain = (__uint128_t{1} << 50) + 7;
  int4 sigma0 = FixedSigma(0);
  int4 sigma1 = FixedSigma(1);

  // Alternating seeds exercise the cache re-expansion path; every call must
  // still follow the seed it was given, and the preloaded instance (whose
  // schedules were expanded at construction) must agree.
  for (int i = 0; i < 1000; ++i) {
    __uint128_t x = static_cast<__uint128_t>(i * 104729 + 5) % domain;
    int4 sigma = (i & 1) ? sigma1 : sigma0;
    __uint128_t y = raw.Permu(sigma, x, domain);
    EXPECT_EQ(y, ref.Permu(sigma, x, domain)) << "mismatch at i=" << i;
    if ((i & 1) == 0) {
      EXPECT_EQ(preloaded.Permu(sigma0, x, domain), y) << "preloaded mismatch at i=" << i;
    }
  }
}

TEST(Aes128FeistelRawPrpTest, BucketBalance) {
  // Cuckoo hashing relies on the PRP spreading inputs over buckets.
  fss::prp::Aes128FeistelRaw raw;
  const __uint128_t domain = 1 << 16;
  constexpr int kBuckets = 64;
  int4 sigma = FixedSigma();

  int counts[kBuckets] = {};
  for (__uint128_t x = 0; x < domain; ++x) {
    __uint128_t y = raw.Permu(sigma, x, domain);
    ++counts[static_cast<int>(y) % kBuckets];
  }
  int mean = static_cast<int>(domain) / kBuckets;
  for (int c : counts) {
    // Loose bound, deterministic for the fixed seed.
    EXPECT_LT(c, mean * 3 / 2);
    EXPECT_GT(c, mean / 2);
  }
}

template <typename Group>
class DmpfAes128FeistelRawTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using Prp = fss::prp::Aes128FeistelRaw;
  using DmpfType = fss::Dmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Prp, uint16_t>;

  static constexpr int m = DmpfType::m;

  uint16_t alphas[kT] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
      1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000};
  int4 betas[kT];

  Prg prg;
  Prp prp_;

  DmpfAes128FeistelRawTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    for (int i = 0; i < kT; ++i) {
      betas[i] = {(i + 1) * 11, 0, 0, 0};
    }
  }

  int4 RandomSeed(std::mt19937 &gen) {
    std::uniform_int_distribution<int> dis;
    return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void GenKeys(typename DmpfType::Key &k0, typename DmpfType::Key &k1) {
    DmpfType dmpf{prg, prp_};
    std::random_device rd;
    std::mt19937 gen(rd());

    int4 sigma = RandomSeed(gen);
    cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = {RandomSeed(gen), RandomSeed(gen)};
    }
    ASSERT_EQ(dmpf.Gen(k0, k1, sigma, cuda::std::span<cuda::std::array<int4, 2>, m>(s0s),
        cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT), 0);
  }

  void TestEvalAtAlpha() {
    typename DmpfType::Key k0, k1;
    GenKeys(k0, k1);
    DmpfType dmpf{prg, prp_};

    std::vector<uint16_t> xs(alphas, alphas + kT);
    std::vector<int4> ys0(kT), ys1(kT);

    dmpf.BatchEval(false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0));
    dmpf.BatchEval(true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1));

    for (int i = 0; i < kT; ++i) {
      auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
      int4 r = result.Into();
      int4 e = betas[i];
      e.w &= ~1;
      EXPECT_EQ(memcmp(&r, &e, sizeof(int4)), 0) << "Failed at alpha=" << alphas[i];
    }
  }

  void TestEvalAtNonAlpha() {
    typename DmpfType::Key k0, k1;
    GenKeys(k0, k1);
    DmpfType dmpf{prg, prp_};

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

    dmpf.BatchEval(false, k0, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys0));
    dmpf.BatchEval(true, k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(ys1));

    int4 zero = {0, 0, 0, 0};
    for (size_t i = 0; i < xs.size(); ++i) {
      auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
      int4 r = result.Into();
      EXPECT_EQ(memcmp(&r, &zero, sizeof(int4)), 0) << "Failed at x=" << xs[i];
    }
  }

  void TestEvalAll() {
    typename DmpfType::Key k0, k1;
    GenKeys(k0, k1);

    DmpfType dmpf{prg, prp_};

    constexpr size_t domain_size = 1ULL << kInBits;

    std::vector<int4> expected_results(domain_size, int4{0, 0, 0, 0});

    for (size_t i = 0; i < kT; ++i) {
      expected_results[alphas[i]] = betas[i];
    }

    std::vector<int4> ys0(domain_size);
    std::vector<int4> ys1(domain_size);

    dmpf.EvalAll(false, k0, ys0);
    dmpf.EvalAll(true, k1, ys1);

    for (size_t i = 0; i < domain_size; ++i) {
      auto result = (Group::From(ys0[i]) + Group::From(ys1[i]));
      int4 r = result.Into();
      int4 e = expected_results[i];
      e.w &= ~1;
      EXPECT_EQ(memcmp(&r, &e, sizeof(int4)), 0) << "Failed at i=" << i;
    }
  }
};

using DmpfBytesAes128FeistelRawTest = DmpfAes128FeistelRawTest<BytesGroup>;
using DmpfUint128Aes128FeistelRawTest = DmpfAes128FeistelRawTest<Uint127Group>;

TEST_F(DmpfBytesAes128FeistelRawTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DmpfBytesAes128FeistelRawTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DmpfBytesAes128FeistelRawTest, EvalAll) {
  TestEvalAll();
}

TEST_F(DmpfUint128Aes128FeistelRawTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DmpfUint128Aes128FeistelRawTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DmpfUint128Aes128FeistelRawTest, EvalAll) {
  TestEvalAll();
}

TEST(CuckooHashAes128FeistelRawTest, Compact) {
  fss::prp::Aes128FeistelRaw prp;
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
    fss::cuckoo_hash::Compact<fss::prp::Aes128FeistelRaw, uint16_t> compact{prp};
    ret = compact.Run(std::span<const uint16_t>(as, t), m, sigma, n, B, 1000, std::span<std::pair<int, int>>(table));
  } while (ret != 0);

  // Verify: every input placed, no bucket overflow, per-party positions reproducible.
  std::vector<int> placed(t, 0);
  for (int i = 0; i < m; ++i) {
    if (table[i].first == -1) continue;
    int j = table[i].first;
    ASSERT_GE(j, 0);
    ASSERT_LT(j, t);
    ASSERT_EQ(placed[j], 0) << "input placed twice";
    placed[j] = 1;

    int k = table[i].second;
    fss::cuckoo_hash::PrpHash<fss::prp::Aes128FeistelRaw, uint16_t> prp_hash{prp};
    auto [bucket, index] = prp_hash.Locate(sigma, as[j], k, n, B);
    EXPECT_EQ(bucket, i);
  }
  for (int j = 0; j < t; ++j) {
    EXPECT_EQ(placed[j], 1) << "input " << j << " not placed";
  }
}
