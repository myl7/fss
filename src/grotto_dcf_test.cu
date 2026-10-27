#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <memory>
#include <fss/grotto_dcf.cuh>
#include <fss/prg/chacha.cuh>

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

class GrottoDcfChaChaTest : public ::testing::Test {
protected:
  using GrottoDcfType = fss::GrottoDcf<kAlphaBits, fss::prg::ChaCha<2>, uint16_t>;

  int4 s0s[2];
  GrottoDcfType::Key k0, k1;
  fss::prg::ChaCha<2> prg;

  GrottoDcfChaChaTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;

    s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};

    GrottoDcfType dcf{prg};
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha);
  }
};

// Test Preprocess + Eval for all x in domain
TEST_F(GrottoDcfChaChaTest, PreprocessEvalAll) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  auto tree0 = dcf.Preprocess(false, k0);
  auto tree1 = dcf.Preprocess(true, k1);

  for (size_t x = 0; x < N; ++x) {
    bool share0 = GrottoDcfType::Eval(tree0, static_cast<uint16_t>(x));
    bool share1 = GrottoDcfType::Eval(tree1, static_cast<uint16_t>(x));
    bool result = share0 ^ share1;
    bool expected = (kAlpha <= x);
    EXPECT_EQ(result, expected) << "Failed at x=" << x;
  }
}

// Test EvalAll
TEST_F(GrottoDcfChaChaTest, EvalAll) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);

  dcf.EvalAll(false, k0, cuda::std::span<bool>(ys0.get(), N));
  dcf.EvalAll(true, k1, cuda::std::span<bool>(ys1.get(), N));

  for (size_t x = 0; x < N; ++x) {
    bool result = ys0[x] ^ ys1[x];
    bool expected = (kAlpha <= x);
    EXPECT_EQ(result, expected) << "Failed at x=" << x;
  }
}

// Test Preprocess + Eval matches EvalAll
TEST_F(GrottoDcfChaChaTest, PreprocessEvalMatchesEvalAll) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  // Preprocess path
  auto tree0 = dcf.Preprocess(false, k0);

  // EvalAll path
  auto ys0 = std::make_unique<bool[]>(N);
  dcf.EvalAll(false, k0, cuda::std::span<bool>(ys0.get(), N));

  for (size_t x = 0; x < N; ++x) {
    bool from_preprocess = GrottoDcfType::Eval(tree0, static_cast<uint16_t>(x));
    EXPECT_EQ(from_preprocess, ys0[x]) << "Mismatch at x=" << x;
  }
}

// Test edge cases: alpha=0 and alpha=N-1
TEST_F(GrottoDcfChaChaTest, EdgeAlphaZero) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  GrottoDcfType::Key edge_k0, edge_k1;
  dcf.Gen(edge_k0, edge_k1, cuda::std::span<const int4, 2>(s0s, 2),
      static_cast<uint16_t>(0));

  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);
  dcf.EvalAll(false, edge_k0, cuda::std::span<bool>(ys0.get(), N));
  dcf.EvalAll(true, edge_k1, cuda::std::span<bool>(ys1.get(), N));

  for (size_t x = 0; x < N; ++x) {
    bool result = ys0[x] ^ ys1[x];
    // alpha=0, so 1[0 <= x] = true for all x
    EXPECT_TRUE(result) << "Failed at x=" << x;
  }
}

TEST_F(GrottoDcfChaChaTest, EdgeAlphaMax) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  uint16_t alpha_max = static_cast<uint16_t>(N - 1);
  GrottoDcfType::Key edge_k0, edge_k1;
  dcf.Gen(edge_k0, edge_k1, cuda::std::span<const int4, 2>(s0s, 2), alpha_max);

  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);
  dcf.EvalAll(false, edge_k0, cuda::std::span<bool>(ys0.get(), N));
  dcf.EvalAll(true, edge_k1, cuda::std::span<bool>(ys1.get(), N));

  for (size_t x = 0; x < N; ++x) {
    bool result = ys0[x] ^ ys1[x];
    // alpha=N-1, so 1[N-1 <= x] = true only at x=N-1
    bool expected = (x == N - 1);
    EXPECT_EQ(result, expected) << "Failed at x=" << x;
  }
}

// Single-path Eval against every (alpha, x) pair of small domains.
template <int InBits>
void CheckSinglePathSweep() {
  using Prg = fss::prg::ChaCha<2>;
  using Gdcf = fss::GrottoDcf<InBits, Prg, uint8_t>;
  constexpr uint32_t n = 1u << InBits;
  Gdcf dcf{Prg(kNonce0, kNonce1)};
  int4 s0s[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  for (uint32_t alpha = 0; alpha < n; ++alpha) {
    typename Gdcf::Key k0, k1;
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), static_cast<uint8_t>(alpha));
    for (uint32_t x = 0; x < n; ++x) {
      bool y0 = dcf.Eval(false, k0, static_cast<uint8_t>(x));
      bool y1 = dcf.Eval(true, k1, static_cast<uint8_t>(x));
      ASSERT_EQ(y0 != y1, alpha <= x) << "in_bits=" << InBits << " alpha=" << alpha << " x=" << x;
    }
  }
}

TEST(GrottoDcfSinglePath, Sweep1) {
  CheckSinglePathSweep<1>();
}
TEST(GrottoDcfSinglePath, Sweep2) {
  CheckSinglePathSweep<2>();
}
TEST(GrottoDcfSinglePath, Sweep5) {
  CheckSinglePathSweep<5>();
}
TEST(GrottoDcfSinglePath, Sweep8) {
  CheckSinglePathSweep<8>();
}

// Single-path Eval on wide domains at the comparison boundaries: alpha itself, its neighbors,
// alpha with the last or the top bit flipped, and a random point.
template <int InBits, typename In>
void CheckSinglePathWide() {
  using Prg = fss::prg::ChaCha<2>;
  using Gdcf = fss::GrottoDcf<InBits, Prg, In>;
  const __uint128_t mask = InBits == 128 ? ~static_cast<__uint128_t>(0) : (static_cast<__uint128_t>(1) << InBits) - 1;
  std::mt19937_64 rng(5);
  auto random_in = [&] { return static_cast<In>(((static_cast<__uint128_t>(rng()) << 64) | rng()) & mask); };
  Gdcf dcf{Prg(kNonce0, kNonce1)};
  for (int trial = 0; trial < 64; ++trial) {
    int4 s0s[2] = {{static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng())},
        {static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng())}};
    In alpha = random_in();
    typename Gdcf::Key k0, k1;
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha);
    const In top_bit = static_cast<In>(static_cast<In>(1) << (InBits - 1));
    const In xs[] = {alpha, static_cast<In>((alpha - 1) & mask), static_cast<In>((alpha + 1) & mask),
        static_cast<In>(alpha ^ 1), static_cast<In>(alpha ^ top_bit), random_in()};
    for (In x : xs) {
      bool y0 = dcf.Eval(false, k0, x);
      bool y1 = dcf.Eval(true, k1, x);
      ASSERT_EQ(y0 != y1, alpha <= x) << "in_bits=" << InBits << " trial=" << trial;
    }
  }
}

TEST(GrottoDcfSinglePath, Wide20) {
  CheckSinglePathWide<20, uint32_t>();
}
TEST(GrottoDcfSinglePath, Wide32) {
  CheckSinglePathWide<32, uint32_t>();
}
TEST(GrottoDcfSinglePath, Wide64) {
  CheckSinglePathWide<64, uint64_t>();
}
TEST(GrottoDcfSinglePath, Wide128) {
  CheckSinglePathWide<128, __uint128_t>();
}
