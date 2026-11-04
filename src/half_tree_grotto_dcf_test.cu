#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/prg/chacha.cuh>
#include "core_api_test_helpers.cuh"

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// Eval against every (alpha, x) pair of small domains, and PackedHalfTreeDpf::Eval on the same
// key against 1[x == alpha]. kDepth 1 to 3 covers a last tree bit at the root.
template <int InBits, typename In = uint16_t>
void CheckSweep() {
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<InBits, fss_test::CcrBackend<Prg>, In>;
  using DpfType = fss::PackedHalfTreeDpf<InBits, 1, fss_test::CcrBackend<Prg>, In>;
  constexpr uint32_t n = 1u << InBits;
  const int4 hash_key = {0x13572468, 0x2468ace0, 0x0badf00d, 0x7f7f7f7f};
  Scheme dcf{Prg(kNonce0, kNonce1), hash_key};
  DpfType dpf{fss_test::Ccr(Prg(kNonce0, kNonce1)), hash_key};
  int4 s0s[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  for (uint32_t alpha = 0; alpha < n; ++alpha) {
    typename Scheme::Key k0, k1;
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), static_cast<In>(alpha));
    for (uint32_t x = 0; x < n; ++x) {
      In xi = static_cast<In>(x);
      bool y0 = fss_test::EvalKey(dcf, false, k0, xi);
      bool y1 = fss_test::EvalKey(dcf, true, k1, xi);
      ASSERT_EQ(y0 != y1, alpha <= x) << "in_bits=" << InBits << " alpha=" << alpha << " x=" << x;
      bool eq0 = DpfType::Extract(fss_test::EvalKey(dpf, false, k0, xi), xi) != 0;
      bool eq1 = DpfType::Extract(fss_test::EvalKey(dpf, true, k1, xi), xi) != 0;
      ASSERT_EQ(eq0 != eq1, x == alpha) << "in_bits=" << InBits << " alpha=" << alpha << " x=" << x;
    }
  }
}

TEST(HalfTreeGrottoDcf, SweepDepth1) {
  CheckSweep<8, uint8_t>();
}
TEST(HalfTreeGrottoDcf, SweepDepth2) {
  CheckSweep<9>();
}
TEST(HalfTreeGrottoDcf, SweepDepth3) {
  CheckSweep<10>();
}

// Eval on wide domains at the comparison boundaries: alpha itself, its neighbors, the other
// block of alpha's lane at the last tree level, the opposite top half, and a random point.
template <int InBits, typename In>
void CheckWide() {
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<InBits, fss_test::CcrBackend<Prg>, In>;
  const __uint128_t mask = InBits == 128 ? ~static_cast<__uint128_t>(0) : (static_cast<__uint128_t>(1) << InBits) - 1;
  std::mt19937_64 rng(3);
  auto random_in = [&] { return static_cast<In>(((static_cast<__uint128_t>(rng()) << 64) | rng()) & mask); };
  Scheme dcf{Prg(kNonce0, kNonce1), {0x02468ace, 0x13579bdf, 0x55aa55aa, 0x0f0f0f0f}};
  for (int trial = 0; trial < 64; ++trial) {
    int4 s0s[2] = {{static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng())},
        {static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng())}};
    In alpha = random_in();
    typename Scheme::Key k0, k1;
    fss_test::GenPair(dcf, k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha);
    const In last_tree_bit = static_cast<In>(static_cast<In>(1) << (InBits - Scheme::kDepth));
    const In top_bit = static_cast<In>(static_cast<In>(1) << (InBits - 1));
    const In xs[] = {alpha, static_cast<In>((alpha - 1) & mask), static_cast<In>((alpha + 1) & mask),
        static_cast<In>(alpha ^ last_tree_bit), static_cast<In>(alpha ^ top_bit), random_in()};
    for (In x : xs) {
      bool y0 = fss_test::EvalKey(dcf, false, k0, x);
      bool y1 = fss_test::EvalKey(dcf, true, k1, x);
      ASSERT_EQ(y0 != y1, alpha <= x) << "in_bits=" << InBits << " trial=" << trial;
    }
  }
}

TEST(HalfTreeGrottoDcf, Wide20) {
  CheckWide<20, uint32_t>();
}
TEST(HalfTreeGrottoDcf, Wide128) {
  CheckWide<128, __uint128_t>();
}

TEST(HalfTreeGrottoDcf, PackedPreprocessingAndFullDomainShares) {
  using Hash = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<8, Hash, uint8_t>;
  Scheme scheme{Hash{kNonce0, kNonce1}, {1, 2, 3, 4}};
  const int4 seeds[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  for (int alpha : {0, 127, 128, 255}) {
    Scheme::Cw cws[Scheme::kDepth + 1];
    scheme.Gen(cws, seeds, static_cast<uint8_t>(alpha));
    bool shares[2][256];
    for (int party = 0; party < 2; ++party) {
      auto tree = scheme.PreprocessOwned(party != 0, seeds[party], cws);
      bool raw[256];
      scheme.EvalAll(party != 0, seeds[party], cws, raw);
      Scheme::EvalAll(tree.View(), shares[party]);
      for (int x = 0; x < 256; ++x) {
        EXPECT_EQ(shares[party][x], raw[x]);
        EXPECT_EQ(shares[party][x], Scheme::Eval(tree.View(), static_cast<uint8_t>(x)));
      }
    }
    for (int x = 0; x < 256; ++x) {
      EXPECT_EQ(shares[0][x] ^ shares[1][x], alpha <= x);
    }
  }
}

namespace {

constexpr int4 kSplitSentinel = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab};

// Checks the split Cws API against the Cw array forms for one configuration:
// every Cw entry, every Eval share, the packed leaves, the parity tree,
// the owned tree, every EvalAll share, and the two-party reconstruction.
// stride == 0 selects the default-argument call. The construction carries no
// control bits, so there is no packed-control parameter.
template <int in_bits, typename In>
void CheckHalfTreeGrottoSplit(In alpha, size_t stride) {
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<in_bits, fss_test::CcrBackend<Prg>, In>;
  constexpr int kDepth = Scheme::kDepth;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  constexpr size_t kTreeSize = 2 * kBlocks - 1;
  constexpr size_t kDomain = size_t{1} << in_bits;
  const int4 hash_key = {0x13572468, 0x2468ace0, 0x0badf00d, 0x7f7f7f7f};
  Scheme scheme{Prg(kNonce0, kNonce1), hash_key};
  const int4 seeds[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  const bool default_stride = stride == 0;
  const size_t level_stride = default_stride ? 1 : stride;
  const size_t used = static_cast<size_t>(kDepth) * level_stride + 1;
  constexpr size_t kExtra = 3;

  typename Scheme::Cw ref[kDepth + 1];
  scheme.Gen(ref, seeds, alpha);
  std::vector<int4> cws(used + kExtra, kSplitSentinel);
  const auto cws_out = cuda::std::span<int4>(cws.data(), cws.size());
  if (default_stride) scheme.Gen(cws_out, seed_span, alpha);
  else scheme.Gen(cws_out, seed_span, alpha, stride);

  for (int l = 0; l <= kDepth; ++l) {
    const size_t index = static_cast<size_t>(l) * level_stride;
    EXPECT_EQ(memcmp(&cws[index], &ref[l].s, sizeof(int4)), 0) << "cws layer " << l;
  }
  // Only the described prefix is written: gap entries and trailing capacity keep the sentinel.
  for (size_t i = 0; i < cws.size(); ++i) {
    if (i % level_stride == 0 && i / level_stride <= static_cast<size_t>(kDepth)) continue;
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
          ? scheme.Eval(b, s0, split_cws, xi)
          : scheme.Eval(b, s0, split_cws, xi, stride);
      EXPECT_EQ(actual, expected) << "Eval party=" << party << " x=" << x;
    }

    // The split preprocessing produces the same packed leaves and tree as the Cw
    // preprocessing, and the owned tree exposes the same entries.
    std::vector<int4> expected_blocks(kBlocks, kSplitSentinel);
    auto expected_tree = std::make_unique<bool[]>(kTreeSize);
    scheme.Preprocess(b, s0, ref, expected_blocks.data(), expected_tree.get());
    std::vector<int4> actual_blocks(kBlocks, kSplitSentinel);
    auto actual_tree = std::make_unique<bool[]>(kTreeSize);
    if (default_stride) {
      scheme.Preprocess(b, s0, split_cws, actual_blocks.data(), actual_tree.get());
    } else {
      scheme.Preprocess(b, s0, split_cws, actual_blocks.data(), actual_tree.get(), 0, stride);
    }
    EXPECT_EQ(memcmp(actual_blocks.data(), expected_blocks.data(), kBlocks * sizeof(int4)), 0)
        << "Preprocess blocks party=" << party;
    for (size_t j = 0; j < kTreeSize; ++j) {
      EXPECT_EQ(actual_tree[j], expected_tree[j]) << "Preprocess tree party=" << party
                                                  << " j=" << j;
    }
    auto owned = default_stride ? scheme.PreprocessOwned(b, s0, split_cws)
                                : scheme.PreprocessOwned(b, s0, split_cws, 0, stride);
    const auto view = owned.View();
    EXPECT_EQ(memcmp(view.blocks, expected_blocks.data(), kBlocks * sizeof(int4)), 0)
        << "PreprocessOwned blocks party=" << party;
    for (size_t j = 0; j < kTreeSize; ++j) {
      EXPECT_EQ(view.p[j], expected_tree[j]) << "PreprocessOwned tree party=" << party
                                             << " j=" << j;
    }

    auto expected_all = std::make_unique<bool[]>(kDomain);
    split_all[party] = std::make_unique<bool[]>(kDomain);
    scheme.EvalAll(b, s0, ref, expected_all.get());
    if (default_stride) scheme.EvalAll(b, s0, split_cws, split_all[party].get());
    else scheme.EvalAll(b, s0, split_cws, split_all[party].get(), 0, stride);
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
void CheckHalfTreeGrottoSplitStrides(In alpha) {
  for (size_t stride : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
    CheckHalfTreeGrottoSplit<in_bits, In>(alpha, stride);
  }
}

}  // namespace

// in_bits 8 is the shallowest packed tree (kDepth 1), 11 is the smallest
// frontier path (kDepth 4), and 12 and 16 widen the domain.
TEST(HalfTreeGrottoDcfCwsTest, Depth1) {
  CheckHalfTreeGrottoSplitStrides<8, uint8_t>(107);
}
TEST(HalfTreeGrottoDcfCwsTest, Depth4) {
  CheckHalfTreeGrottoSplitStrides<11, uint16_t>(1000);
}
TEST(HalfTreeGrottoDcfCwsTest, Depth5) {
  CheckHalfTreeGrottoSplitStrides<12, uint16_t>(1234);
}
TEST(HalfTreeGrottoDcfCwsTest, Depth9WideInput) {
  CheckHalfTreeGrottoSplit<16, uint64_t>(12345, 0);
  CheckHalfTreeGrottoSplit<16, uint64_t>(12345, 3);
}

// The Preprocess, PreprocessOwned, and split EvalAll entries are host
// boundaries and keep throwing; the split Gen and split Eval are low-level
// entries shared with the device and abort in Debug.
TEST(HalfTreeGrottoDcfCwsTest, RejectsInvalidInputs) {
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<8, fss_test::CcrBackend<Prg>, uint8_t>;
  constexpr int kDepth = Scheme::kDepth;
  Scheme scheme{Prg(kNonce0, kNonce1), {0x13572468, 0x2468ace0, 0x0badf00d, 0x7f7f7f7f}};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  int4 cws[kDepth + 1] = {};
  int4 blocks[4] = {};
  bool p[8] = {};
  const int4 s0 = seeds[0];
  const auto cws_in = cuda::std::span<const int4>(cws, kDepth + 1);

  // A zero stride is only the batch sentinel, so a non-batch method rejects it.
  EXPECT_THROW((void)scheme.Preprocess(false, s0, cws_in, blocks, p, 0, 0),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.PreprocessOwned(false, s0, cws_in, 0, 0), std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalAll(false, s0, cws_in, p, 0, 0), std::invalid_argument);
  // A span that omits the packed-output correction word at cws[kDepth*stride] is rejected.
  EXPECT_THROW((void)scheme.Preprocess(false, s0, cuda::std::span<const int4>(cws, kDepth),
                        blocks, p),
      std::invalid_argument);
  EXPECT_THROW((void)scheme.EvalAll(false, s0, cuda::std::span<const int4>(cws, kDepth), p),
      std::invalid_argument);
}

#ifndef NDEBUG
TEST(HalfTreeGrottoDcfCwsTest, AbortsOnInvalidLowLevelInputsInDebug) {
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<8, fss_test::CcrBackend<Prg>, uint8_t>;
  constexpr int kDepth = Scheme::kDepth;
  Scheme scheme{Prg(kNonce0, kNonce1), {0x13572468, 0x2468ace0, 0x0badf00d, 0x7f7f7f7f}};
  const int4 seeds[2] = {{13, 27, 45, 66}, {19, 31, 53, 78}};
  const cuda::std::span<const int4, 2> seed_span(seeds, 2);
  int4 cws[kDepth + 1] = {};
  const auto cws_out = cuda::std::span<int4>(cws, kDepth + 1);
  const auto cws_in = cuda::std::span<const int4>(cws, kDepth + 1);
  const int4 s0 = seeds[0];

  EXPECT_DEATH((void)scheme.Gen(cws_out, seed_span, uint8_t{5}, 0), "Assertion.*failed");
  EXPECT_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, 0), "Assertion.*failed");
  EXPECT_DEATH((void)scheme.Gen(cuda::std::span<int4>(cws, kDepth), seed_span, uint8_t{5}),
      "Assertion.*failed");
  EXPECT_DEATH((void)scheme.Eval(false, s0, cuda::std::span<const int4>(cws, kDepth),
                       uint8_t{0}),
      "Assertion.*failed");
  // A stride past the addressable guard is rejected.
  constexpr size_t kMaxElements = std::numeric_limits<size_t>::max() / sizeof(int4);
  const size_t overflowing_stride = (kMaxElements - 1) / kDepth + 1;
  EXPECT_DEATH((void)scheme.Eval(false, s0, cws_in, uint8_t{0}, overflowing_stride),
      "Assertion.*failed");
}
#endif  // NDEBUG

// The GPU split launches are compile-verified here and executed only on a
// machine with a free device, comparing against the host Cw-form outputs.
namespace {

void CheckHalfTreeGrottoSplitGpuLaunches() {
  int devices = 0;
  cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
      (device_status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(device_status, cudaSuccess);

  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<8, fss_test::CcrBackend<Prg>, uint16_t>;
  constexpr int kDepth = Scheme::kDepth;
  constexpr size_t kBlocks = size_t{1} << kDepth;
  constexpr size_t kTreeSize = 2 * kBlocks - 1;
  constexpr size_t kDomain = size_t{1} << 8;
  Scheme scheme{Prg(kNonce0, kNonce1), {0x13572468, 0x2468ace0, 0x0badf00d, 0x7f7f7f7f}};
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint16_t alpha = 107;
  constexpr size_t kStride = 2;

  typename Scheme::Cw ref[kDepth + 1];
  scheme.Gen(ref, seeds, alpha);
  std::vector<int4> cws((kDepth + 1) * kStride + 2, kSplitSentinel);
  scheme.Gen(cuda::std::span<int4>(cws.data(), cws.size()),
      cuda::std::span<const int4, 2>(seeds, 2), alpha, kStride);

  int4 *device_cws = nullptr;
  int4 *device_blocks = nullptr;
  bool *device_ys = nullptr;
  bool *device_tree = nullptr;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_cws, cws.size() * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_blocks, kBlocks * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_ys, kDomain * sizeof(bool)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tree, kTreeSize * sizeof(bool)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_cws, cws.data(), cws.size() * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  const auto span_cws = cuda::std::span<const int4>(device_cws, cws.size());

  auto first = std::make_unique<bool[]>(kDomain);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(party);
    const bool b = party != 0;
    auto actual = std::make_unique<bool[]>(kDomain);
    ASSERT_EQ((scheme.template EvalAllGpu<8, 16>(b, seeds[party], span_cws,
        cuda::std::span<bool>(device_ys, kDomain), stream, kStride)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual.get(), device_ys, kDomain * sizeof(bool),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    // EvalAllGpu() and EvalAll() take the same leaf-prefix route, so their shares
    // are comparable. Scalar Eval() takes the other route and its shares are not.
    auto expected_all = std::make_unique<bool[]>(kDomain);
    scheme.EvalAll(b, seeds[party], ref, expected_all.get());
    for (size_t x = 0; x < kDomain; ++x) {
      EXPECT_EQ(actual[x], expected_all[x]) << "x=" << x;
      if (party == 0) {
        first[x] = actual[x];
      } else {
        EXPECT_EQ(static_cast<bool>(first[x] ^ actual[x]), x >= alpha) << "x=" << x;
      }
    }

    std::vector<int4> expected_blocks(kBlocks, kSplitSentinel);
    auto expected_tree = std::make_unique<bool[]>(kTreeSize);
    scheme.Preprocess(b, seeds[party], ref, expected_blocks.data(), expected_tree.get());
    std::vector<int4> actual_blocks(kBlocks, kSplitSentinel);
    auto actual_tree = std::make_unique<bool[]>(kTreeSize);
    ASSERT_EQ((scheme.template PreprocessGpu<8, 16>(b, seeds[party], span_cws,
        cuda::std::span<int4>(device_blocks, kBlocks),
        cuda::std::span<bool>(device_tree, kTreeSize), stream, kStride)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_blocks.data(), device_blocks, kBlocks * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_tree.get(), device_tree, kTreeSize * sizeof(bool),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(memcmp(actual_blocks.data(), expected_blocks.data(), kBlocks * sizeof(int4)), 0);
    for (size_t j = 0; j < kTreeSize; ++j) {
      EXPECT_EQ(actual_tree[j], expected_tree[j]) << "tree j=" << j;
    }
  }

  // A zero stride and a span that omits the terminal are rejected.
  EXPECT_EQ((scheme.template EvalAllGpu<8, 16>(false, seeds[0], span_cws,
      cuda::std::span<bool>(device_ys, kDomain), stream, 0)), cudaErrorInvalidValue);
  EXPECT_EQ((scheme.template PreprocessGpu<8, 16>(false, seeds[0],
      cuda::std::span<const int4>(device_cws, kDepth),
      cuda::std::span<int4>(device_blocks, kBlocks),
      cuda::std::span<bool>(device_tree, kTreeSize), stream)), cudaErrorInvalidValue);

  EXPECT_EQ(cudaFree(device_tree), cudaSuccess);
  EXPECT_EQ(cudaFree(device_ys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_blocks), cudaSuccess);
  EXPECT_EQ(cudaFree(device_cws), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace

TEST(HalfTreeGrottoDcfSplitGpuTest, SplitLaunchesMatchCwHost) {
  CheckHalfTreeGrottoSplitGpuLaunches();
}
