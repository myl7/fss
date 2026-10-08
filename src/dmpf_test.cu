#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/dmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/aes128_feistel.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;

static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

template <typename Group>
class DmpfChaChaTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using Prp = fss::prp::Aes128Feistel;
  using DmpfType = fss::Dmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Prp, uint16_t>;

  static constexpr int m = DmpfType::m;

  uint16_t alphas[kT] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
      1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000};
  int4 betas[kT];

  Prg prg;
  Prp prp_;

  DmpfChaChaTest() : prg(gChaChaDeviceNonces) {}

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

    int ret;
    do {
      int4 sigma = RandomSeed(gen);
      cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
      for (int i = 0; i < m; ++i) {
        s0s[i] = {RandomSeed(gen), RandomSeed(gen)};
      }
      ret = dmpf.Gen(k0, k1, sigma, cuda::std::span<const cuda::std::array<int4, 2>, m>(s0s),
          std::span<const uint16_t>(alphas, kT), std::span<const int4>(betas, kT), kT);
    } while (ret != 0);
  }

  void TestEvalAtAlpha() {
    typename DmpfType::Key k0, k1;
    GenKeys(k0, k1);
    DmpfType dmpf{prg, prp_};

    std::vector<uint16_t> xs(alphas, alphas + kT);
    std::vector<int4> ys0(kT), ys1(kT);

    dmpf.BatchEval(false, k0, std::span<const uint16_t>(xs), std::span<int4>(ys0));
    dmpf.BatchEval(true, k1, std::span<const uint16_t>(xs), std::span<int4>(ys1));

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

    dmpf.BatchEval(false, k0, std::span<const uint16_t>(xs), std::span<int4>(ys0));
    dmpf.BatchEval(true, k1, std::span<const uint16_t>(xs), std::span<int4>(ys1));

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

using DmpfBytesChaChaTest = DmpfChaChaTest<BytesGroup>;
using DmpfUint128ChaChaTest = DmpfChaChaTest<Uint127Group>;

TEST_F(DmpfBytesChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DmpfBytesChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DmpfBytesChaChaTest, EvalAll) {
  TestEvalAll();
}

TEST_F(DmpfUint128ChaChaTest, EvalAtAlpha) {
  TestEvalAtAlpha();
}
TEST_F(DmpfUint128ChaChaTest, EvalAtNonAlpha) {
  TestEvalAtNonAlpha();
}
TEST_F(DmpfUint128ChaChaTest, EvalAll) {
  TestEvalAll();
}

TEST(CuckooHashTest, Compact) {
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
    fss::cuckoo_hash::Compact<fss::prp::Aes128Feistel, uint16_t> compact{prp};
    ret = compact.Run(std::span<const uint16_t>(as, t), m, sigma, n, B, 1000, std::span<std::pair<int, int>>(table));
  } while (ret != 0);

  // Each alpha should appear exactly once in the table.
  int count = 0;
  for (int i = 0; i < m; ++i) {
    if (table[i].first != -1) count++;
  }
  EXPECT_EQ(count, t);
}
