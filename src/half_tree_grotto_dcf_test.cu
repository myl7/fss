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
