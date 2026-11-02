#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <memory>
#include <fss/grotto_dcf.cuh>
#include <fss/prg/chacha.cuh>
#include "core_api_test_helpers.cuh"

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
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), kAlpha);
  }
};

// Test Preprocess + Eval for all x in domain
TEST_F(GrottoDcfChaChaTest, PreprocessEvalAll) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  auto tree0 = fss_test::PreprocessKey(dcf, false, k0);
  auto tree1 = fss_test::PreprocessKey(dcf, true, k1);

  for (size_t x = 0; x < N; ++x) {
    bool share0 = GrottoDcfType::Eval(tree0.View(), static_cast<uint16_t>(x));
    bool share1 = GrottoDcfType::Eval(tree1.View(), static_cast<uint16_t>(x));
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

  fss_test::EvalAllKey(dcf, false, k0, cuda::std::span<bool>(ys0.get(), N));
  fss_test::EvalAllKey(dcf, true, k1, cuda::std::span<bool>(ys1.get(), N));

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

  for (bool party : {false, true}) {
    const auto &key = party ? k1 : k0;
    auto tree = fss_test::PreprocessKey(dcf, party, key);
    auto raw_ys = std::make_unique<bool[]>(N);
    auto tree_ys = std::make_unique<bool[]>(N);
    fss_test::EvalAllKey(dcf, party, key, cuda::std::span<bool>(raw_ys.get(), N));
    GrottoDcfType::EvalAll(tree.View(), tree_ys.get());

    for (size_t x = 0; x < N; ++x) {
      bool from_preprocess = GrottoDcfType::Eval(tree.View(), static_cast<uint16_t>(x));
      EXPECT_EQ(from_preprocess, raw_ys[x]) << "party " << party << ", input " << x;
      EXPECT_EQ(tree_ys[x], raw_ys[x]) << "party " << party << ", input " << x;
    }
  }
}

// Test edge cases: alpha=0 and alpha=N-1
TEST_F(GrottoDcfChaChaTest, EdgeAlphaZero) {
  GrottoDcfType dcf{prg};
  constexpr size_t N = 1ULL << kAlphaBits;

  GrottoDcfType::Key edge_k0, edge_k1;
  fss_test::GenPair(dcf, edge_k0, edge_k1, cuda::std::span<const int4, 2>(s0s, 2),
      static_cast<uint16_t>(0));

  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);
  fss_test::EvalAllKey(dcf, false, edge_k0, cuda::std::span<bool>(ys0.get(), N));
  fss_test::EvalAllKey(dcf, true, edge_k1, cuda::std::span<bool>(ys1.get(), N));

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
  fss_test::GenPair(dcf, edge_k0, edge_k1, cuda::std::span<const int4, 2>(s0s, 2), alpha_max);

  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);
  fss_test::EvalAllKey(dcf, false, edge_k0, cuda::std::span<bool>(ys0.get(), N));
  fss_test::EvalAllKey(dcf, true, edge_k1, cuda::std::span<bool>(ys1.get(), N));

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
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), static_cast<uint8_t>(alpha));
    for (uint32_t x = 0; x < n; ++x) {
      bool y0 = fss_test::EvalKey(dcf, false, k0, static_cast<uint8_t>(x));
      bool y1 = fss_test::EvalKey(dcf, true, k1, static_cast<uint8_t>(x));
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
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha);
    const In top_bit = static_cast<In>(static_cast<In>(1) << (InBits - 1));
    const In xs[] = {alpha, static_cast<In>((alpha - 1) & mask), static_cast<In>((alpha + 1) & mask),
        static_cast<In>(alpha ^ 1), static_cast<In>(alpha ^ top_bit), random_in()};
    for (In x : xs) {
      bool y0 = fss_test::EvalKey(dcf, false, k0, x);
      bool y1 = fss_test::EvalKey(dcf, true, k1, x);
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


template <bool has_half>
void CheckGrottoGenHalf() {
  constexpr int bits = 5;
  using Backend = fss::prg::Aes128Soft<2>;
  using Prg = fss_test::CountedPrg<Backend, 2, has_half>;
  using Scheme = fss::GrottoDcf<bits, Prg, uint8_t>;
  fss_test::GenHalfAesFixture<2> aes;
  fss_test::PrgCallCounts calls;
  Scheme scheme{Prg(aes.Backend(), calls)};
  int4 seeds[2] = {{1, 3, 5, 6}, {7, 9, 11, 12}};
  typename Scheme::Key keys[2];
  fss_test::GenPair(scheme, keys[0], keys[1], seeds, uint8_t{13});
  EXPECT_EQ(calls.full, 2 * bits);
  EXPECT_EQ(calls.half, 0);
  for (uint8_t x = 0; x < 32; ++x) {
    bool ys[2];
    for (int b = 0; b < 2; ++b) {
      calls = {};
      ys[b] = fss_test::EvalKey(scheme, b, keys[b], x);
      EXPECT_EQ(calls.full, has_half ? 0 : bits);
      EXPECT_EQ(calls.half, has_half ? bits : 0);
      EXPECT_EQ(calls.right, has_half ? __builtin_popcount(x) : 0);
    }
    EXPECT_EQ(ys[0] != ys[1], x >= 13);
  }
}

TEST(GrottoDcfGenHalf, SelectedAndFallback) {
  CheckGrottoGenHalf<true>();
  CheckGrottoGenHalf<false>();
}

namespace {

constexpr int4 kSplitSentinel = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};

// Checks the split correction-word API against the Cw forms for one configuration:
// every correction word entry, every Eval share, the preprocessed parity tree, the
// owned tree, every EvalAll share, and the two-party reconstruction. stride == 0
// selects the default-argument call.
template <int in_bits, typename In>
void CheckGrottoSplit(In alpha, size_t stride) {
  using Scheme = fss::GrottoDcf<in_bits, fss::prg::ChaCha<2>, In>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const bool default_stride = stride == 0;
  const size_t level_stride = default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(in_bits) * level_stride + 1;
  constexpr size_t kExtra = 3;
  constexpr size_t kDomain = size_t{1} << in_bits;
  constexpr size_t kTreeSize = 2 * kDomain - 1;

  typename Scheme::Cw ref[in_bits + 1];
  scheme.Gen(ref, seeds, alpha);
  std::vector<int4> cws(used + kExtra, kSplitSentinel);
  const auto cws_out = cuda::std::span<int4>(cws.data(), cws.size());
  In tr = 0;
  if (default_stride) scheme.Gen(cws_out, tr, seed_span, alpha);
  else scheme.Gen(cws_out, tr, seed_span, alpha, stride);

  In expected_tr = 0;
  for (int l = 0; l < in_bits; ++l) {
    const size_t index = static_cast<size_t>(l) * level_stride;
    EXPECT_EQ(memcmp(&cws[index], &ref[l].s, sizeof(int4)), 0) << "cws layer " << l;
    if (ref[l].tr) expected_tr |= static_cast<In>(static_cast<In>(1) << l);
  }
  EXPECT_EQ(tr, expected_tr) << "packed right-control bits";
  EXPECT_EQ(memcmp(&cws[used - 1], &ref[in_bits].s, sizeof(int4)), 0) << "cws terminal";
  // Only the described prefix is written: gap entries and trailing capacity keep the sentinel.
  for (size_t i = 0; i < cws.size(); ++i) {
    if (i % level_stride == 0 && i / level_stride <= static_cast<size_t>(in_bits)) continue;
    EXPECT_EQ(memcmp(&cws[i], &kSplitSentinel, sizeof(int4)), 0) << "cws untouched at " << i;
  }

  const cuda::std::span<const int4> split_cws(cws.data(), used);
  std::unique_ptr<bool[]> split_all[2];
  for (int party = 0; party < 2; ++party) {
    const bool b = party != 0;
    const int4 s0 = seeds[party];
    for (size_t x = 0; x < kDomain; ++x) {
      const In xi = static_cast<In>(x);
      const bool expected = scheme.Eval(b, s0, ref, xi);
      const bool actual = default_stride
          ? scheme.Eval(b, s0, split_cws, tr, xi)
          : scheme.Eval(b, s0, split_cws, tr, xi, stride);
      EXPECT_EQ(actual, expected) << "Eval party=" << party << " x=" << x;
    }

    // The split preprocessing produces the same tree as the Cw preprocessing.
    auto expected_tree = std::make_unique<bool[]>(kTreeSize);
    auto actual_tree = std::make_unique<bool[]>(kTreeSize);
    scheme.Preprocess(b, s0, ref, expected_tree.get());
    if (default_stride) scheme.Preprocess(b, s0, split_cws, tr, actual_tree.get());
    else scheme.Preprocess(b, s0, split_cws, tr, actual_tree.get(), 0, stride);
    for (size_t j = 0; j < kTreeSize; ++j) {
      EXPECT_EQ(actual_tree[j], expected_tree[j]) << "Preprocess party=" << party << " j=" << j;
    }
    // The owned tree exposes the same entries.
    auto owned = default_stride ? scheme.PreprocessOwned(b, s0, split_cws, tr)
                                : scheme.PreprocessOwned(b, s0, split_cws, tr, 0, stride);
    const auto view = owned.View();
    for (size_t j = 0; j < kTreeSize; ++j) {
      EXPECT_EQ(view.p[j], expected_tree[j]) << "PreprocessOwned party=" << party << " j=" << j;
    }

    auto expected_all = std::make_unique<bool[]>(kDomain);
    split_all[party] = std::make_unique<bool[]>(kDomain);
    scheme.EvalAll(b, s0, ref, expected_all.get());
    if (default_stride) scheme.EvalAll(b, s0, split_cws, tr, split_all[party].get());
    else scheme.EvalAll(b, s0, split_cws, tr, split_all[party].get(), 0, stride);
    for (size_t x = 0; x < kDomain; ++x) {
      EXPECT_EQ(split_all[party][x], expected_all[x]) << "EvalAll party=" << party << " x=" << x;
    }
  }

  for (size_t x = 0; x < kDomain; ++x) {
    EXPECT_EQ(split_all[0][x] != split_all[1][x], static_cast<In>(x) >= alpha)
        << "reconstruction x=" << x;
  }
}

template <int in_bits, typename In>
void CheckGrottoSplitStrides(In alpha) {
  for (size_t stride : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
    CheckGrottoSplit<in_bits, In>(alpha, stride);
  }
}

template <int in_bits, typename In>
void CheckGrottoSplitFromCwStorage(In alpha) {
  using Scheme = fss::GrottoDcf<in_bits, fss::prg::ChaCha<2>, In>;
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  typename Scheme::Cw words[in_bits + 1];
  scheme.Gen(words, seeds, alpha);

  std::vector<int4> storage(2 * (in_bits + 1), kSplitSentinel);
  for (int l = 0; l <= in_bits; ++l) storage[2 * l] = words[l].s;
  In tr = 0;
  for (int l = 0; l < in_bits; ++l) {
    if (words[l].tr) tr |= static_cast<In>(static_cast<In>(1) << l);
  }
  const auto cws = cuda::std::span<const int4>(storage.data(), storage.size());
  constexpr size_t kDomain = size_t{1} << in_bits;
  for (int party = 0; party < 2; ++party) {
    for (size_t x = 0; x < kDomain; ++x) {
      const In xi = static_cast<In>(x);
      const bool expected = scheme.Eval(party, seeds[party], words, xi);
      const bool actual = scheme.Eval(party, seeds[party], cws, tr, xi, 2);
      EXPECT_EQ(actual, expected) << "party=" << party << " x=" << x;
    }
    auto expected_all = std::make_unique<bool[]>(kDomain);
    auto actual_all = std::make_unique<bool[]>(kDomain);
    scheme.EvalAll(party, seeds[party], words, expected_all.get());
    scheme.EvalAll(party, seeds[party], cws, tr, actual_all.get(), 0, 2);
    for (size_t x = 0; x < kDomain; ++x) {
      EXPECT_EQ(actual_all[x], expected_all[x]) << "EvalAll party=" << party << " x=" << x;
    }
  }
}

}  // namespace

// in_bits 1 is the single-level tree, 2 and 5 cover the recursive expansion, and
// 8 fills the domain. The split key is the Dpf key, so the control bits follow
// the Dpf layout.
TEST(GrottoDcfCwsTest, Depth1) { CheckGrottoSplitStrides<1, uint8_t>(1); }
TEST(GrottoDcfCwsTest, Depth2) { CheckGrottoSplitStrides<2, uint8_t>(3); }
TEST(GrottoDcfCwsTest, Depth5) { CheckGrottoSplitStrides<5, uint8_t>(17); }
TEST(GrottoDcfCwsTest, Depth8) { CheckGrottoSplitStrides<8, uint16_t>(107); }
TEST(GrottoDcfCwsTest, Depth8WideInput) {
  CheckGrottoSplitStrides<8, uint64_t>(107);
}
TEST(GrottoDcfCwsTest, CwStorageView) {
  CheckGrottoSplitFromCwStorage<5, uint8_t>(17);
  CheckGrottoSplitFromCwStorage<8, uint16_t>(107);
}

#ifndef NDEBUG
TEST(GrottoDcfCwsTest, RejectsInvalidInputs) {
  using Scheme = fss::GrottoDcf<3, fss::prg::ChaCha<2>, uint8_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  int4 cws[4] = {};
  bool ys[8] = {};
  uint8_t tr = 0;
  const auto cws_out = cuda::std::span<int4>(cws, 4);
  const auto cws_in = cuda::std::span<const int4>(cws, 4);
  const int4 s0 = seeds[0];

  // A zero stride is only the batch sentinel, so a non-batch method rejects it.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cws_out, tr, seed_span, uint8_t{5}, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, uint8_t{0}, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Preprocess(false, s0, cws_in, uint8_t{0}, ys, 0, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.PreprocessOwned(false, s0, cws_in, uint8_t{0}, 0, 0),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cws_in, uint8_t{0}, ys, 0, 0),
      "Assertion.*failed");
  // A span that omits the terminal slot at cws[in_bits*stride] is rejected.
  EXPECT_DEBUG_DEATH((void)scheme.Gen(cuda::std::span<int4>(cws, 3), tr, seed_span,
                        uint8_t{5}),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(cws, 3),
                        uint8_t{0}, uint8_t{0}),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.Preprocess(false, s0, cuda::std::span<const int4>(cws, 3),
                        uint8_t{0}, ys),
      "Assertion.*failed");
  EXPECT_DEBUG_DEATH((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(cws, 3),
                        uint8_t{0}, ys),
      "Assertion.*failed");
  // A stride past the addressable guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / 3 + 1;
  EXPECT_DEBUG_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, uint8_t{0},
                        overflowing_stride),
      "Assertion.*failed");
}
#endif  // NDEBUG

// The GPU split launches are compile-verified here and executed only on a
// machine with a free device, comparing against the host Cw-form outputs.
namespace {

void CheckGrottoSplitGpuLaunches() {
  int devices = 0;
  cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
      (device_status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(device_status, cudaSuccess);

  constexpr int kBits = 8;
  constexpr size_t kDomain = size_t{1} << kBits;
  constexpr size_t kTreeSize = 2 * kDomain - 1;
  using Scheme = fss::GrottoDcf<kBits, fss::prg::ChaCha<2>, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<2>(kNonce0, kNonce1)};
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;
  constexpr size_t kStride = 2;

  typename Scheme::Cw ref[kBits + 1];
  scheme.Gen(ref, seeds, alpha);
  std::vector<int4> cws((kBits + 1) * kStride + 2, kSplitSentinel);
  uint16_t tr = 0;
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha, kStride);

  int4 *device_cws = nullptr;
  bool *device_ys = nullptr;
  bool *device_tree = nullptr;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_cws, cws.size() * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, kDomain * sizeof(bool)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tree, kTreeSize * sizeof(bool)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_cws, cws.data(), cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  const auto span_cws = cuda::std::span<const int4>(device_cws, cws.size());

  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(party);
    const bool b = party != 0;
    auto actual = std::make_unique<bool[]>(kDomain);
    ASSERT_EQ((scheme.template EvalAllGpu<8, 16>(b, seeds[party], span_cws, tr,
        cuda::std::span<bool>(device_ys, kDomain), stream, kStride)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual.get(), device_ys, kDomain * sizeof(bool),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t x = 0; x < kDomain; ++x) {
      const bool expected = scheme.Eval(b, seeds[party], ref, static_cast<uint16_t>(x));
      EXPECT_EQ(actual[x], expected) << "x=" << x;
    }

    auto expected_tree = std::make_unique<bool[]>(kTreeSize);
    scheme.Preprocess(b, seeds[party], ref, expected_tree.get());
    auto actual_tree = std::make_unique<bool[]>(kTreeSize);
    ASSERT_EQ((scheme.template PreprocessGpu<8, 16>(b, seeds[party], span_cws, tr,
        cuda::std::span<bool>(device_tree, kTreeSize), stream, kStride)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_tree.get(), device_tree, kTreeSize * sizeof(bool),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t j = 0; j < kTreeSize; ++j) {
      EXPECT_EQ(actual_tree[j], expected_tree[j]) << "tree j=" << j;
    }
  }

  // A zero stride and a span that omits the terminal are rejected.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, seeds[0], span_cws, tr,
      cuda::std::span<bool>(device_ys, kDomain), stream, 0)), cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template PreprocessGpu<8, 16>(false, seeds[0],
      cuda::std::span<const int4>(device_cws, kBits), tr,
      cuda::std::span<bool>(device_tree, kTreeSize), stream)), cudaErrorInvalidValue);

  EXPECT_EQ(cudaFree(device_tree), cudaSuccess);
  EXPECT_EQ(cudaFree(device_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_cws), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace

TEST(GrottoDcfSplitGpuTest, SplitLaunchesMatchCwHost) { CheckGrottoSplitGpuLaunches(); }
