#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/prg/chacha.cuh>

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

// Eval against every (alpha, x) pair of small domains, and PackedHalfTreeDpf::Eval on the same
// key against 1[x == alpha]. kDepth 1 to 3 covers a last tree bit at the root.
template <int InBits, typename In = uint16_t>
void CheckSweep() {
  using Prg = fss::prg::ChaCha<1>;
  using Scheme = fss::HalfTreeGrottoDcf<InBits, Prg, In>;
  using DpfType = fss::PackedHalfTreeDpf<InBits, 1, Prg, In>;
  constexpr uint32_t n = 1u << InBits;
  const int4 hash_key = {0x13572468, 0x2468ace0, 0x0badf00d, 0x7f7f7f7f};
  Scheme dcf{Prg(kNonce0, kNonce1), hash_key};
  DpfType dpf{Prg(kNonce0, kNonce1), hash_key};
  int4 s0s[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  for (uint32_t alpha = 0; alpha < n; ++alpha) {
    typename Scheme::Key k0, k1;
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), static_cast<In>(alpha));
    for (uint32_t x = 0; x < n; ++x) {
      In xi = static_cast<In>(x);
      bool y0 = dcf.Eval(false, k0, xi);
      bool y1 = dcf.Eval(true, k1, xi);
      ASSERT_EQ(y0 != y1, alpha <= x) << "in_bits=" << InBits << " alpha=" << alpha << " x=" << x;
      bool eq0 = DpfType::Extract(dpf.Eval(false, k0, xi), xi) != 0;
      bool eq1 = DpfType::Extract(dpf.Eval(true, k1, xi), xi) != 0;
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
  using Prg = fss::prg::ChaCha<1>;
  using Scheme = fss::HalfTreeGrottoDcf<InBits, Prg, In>;
  const __uint128_t mask = InBits == 128 ? ~static_cast<__uint128_t>(0) : (static_cast<__uint128_t>(1) << InBits) - 1;
  std::mt19937_64 rng(3);
  auto random_in = [&] { return static_cast<In>(((static_cast<__uint128_t>(rng()) << 64) | rng()) & mask); };
  Scheme dcf{Prg(kNonce0, kNonce1), {0x02468ace, 0x13579bdf, 0x55aa55aa, 0x0f0f0f0f}};
  for (int trial = 0; trial < 64; ++trial) {
    int4 s0s[2] = {{static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng())},
        {static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng()), static_cast<int>(rng())}};
    In alpha = random_in();
    typename Scheme::Key k0, k1;
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha);
    const In last_tree_bit = static_cast<In>(static_cast<In>(1) << (InBits - Scheme::kDepth));
    const In top_bit = static_cast<In>(static_cast<In>(1) << (InBits - 1));
    const In xs[] = {alpha, static_cast<In>((alpha - 1) & mask), static_cast<In>((alpha + 1) & mask),
        static_cast<In>(alpha ^ last_tree_bit), static_cast<In>(alpha ^ top_bit), random_in()};
    for (In x : xs) {
      bool y0 = dcf.Eval(false, k0, x);
      bool y1 = dcf.Eval(true, k1, x);
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
