// Point-evaluation CPU fast path tests.
//
// The file serves two build modes:
// - Default (CMake): plain GTest cases, compiled like the other *test.cu.
// - POINT_EVAL_CPU_STANDALONE: the same checks run from a small main(), so
//   the fast-path assertions can be verified without a CUDA toolchain or a
//   GTest install, e.g. with clang++ -std=c++20 -O2 -maes -msse4.1 plus
//   minimal cuda_runtime.h / cuda::std stubs. Pass --bench to also print an
//   indicative per-evaluation timing at in_bits = 20.

#ifdef POINT_EVAL_CPU_STANDALONE
#include <chrono>
#include <cstdio>
#else
#include <gtest/gtest.h>
#endif
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <fss/dpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/point_eval_cpu.cuh>
#include <fss/prg/aes128_mmo_raw.cuh>
#include <fss/prg/chacha.cuh>

namespace {

using BytesGroup = fss::group::Bytes;
using Uint8Group = fss::group::Uint<uint8_t>;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;

using Aes2 = fss::prg::Aes128MmoRaw<2>;
using Aes1 = fss::prg::Aes128MmoRaw<1>;
using ChaCha2 = fss::prg::ChaCha<2>;
using ChaCha1 = fss::prg::ChaCha<1>;

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);
constexpr uint8_t kAesKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};
constexpr __uint128_t kBeta = 604;

template <typename Prg>
Prg MakePrg() {
  if constexpr (std::is_same_v<Prg, fss::prg::ChaCha<1>> || std::is_same_v<Prg, fss::prg::ChaCha<2>>) {
    return Prg(kNonce0, kNonce1);
  } else {
    return Prg(kAesKeys);
  }
}

// Deterministic per-case seeding so a reported mismatch reproduces exactly.
std::mt19937 MakeGen(uint64_t tag) {
  return std::mt19937(static_cast<uint32_t>(tag * 0x9E3779B97F4A7C15ULL));
}

std::mt19937_64 MakeGen64(uint64_t tag) {
  return std::mt19937_64(tag * 0x2545F4914F6CDD1DULL);
}

int4 MakeSeed(std::mt19937 &gen) {
  std::uniform_int_distribution<int> dis;
  return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
}

int4 MakeBBuf() {
  return {static_cast<int>(kBeta & 0xFFFFFFFF), 0, 0, 0};
}

template <int in_bits, typename In>
In MaxDomain() {
  if constexpr (in_bits <= 64) {
    return static_cast<In>((uint64_t{1} << in_bits) - 1);
  } else {
    return static_cast<In>((__uint128_t{1} << in_bits) - 1);
  }
}

template <typename In>
void PrintVal(std::FILE *f, In val) {
  if constexpr (std::is_same_v<In, __uint128_t>) {
    std::fprintf(f, "%016llx%016llx", static_cast<unsigned long long>(static_cast<uint64_t>(val >> 64)),
        static_cast<unsigned long long>(static_cast<uint64_t>(val)));
  } else {
    std::fprintf(f, "%llx", static_cast<unsigned long long>(val));
  }
}

template <typename In>
In RandomIn(std::mt19937_64 &gen) {
  if constexpr (std::is_same_v<In, __uint128_t>) {
    return static_cast<In>(static_cast<__uint128_t>(gen()) | static_cast<__uint128_t>(gen()) << 64);
  } else {
    return static_cast<In>(gen());
  }
}

// Boundary points around alpha plus randomized x, masked to the domain.
template <int in_bits, typename In>
std::vector<In> MakePoints(In alpha, std::mt19937_64 &gen) {
  const In kMax = MaxDomain<in_bits, In>();
  std::vector<In> xs = {0, 1};
  if (alpha > 0) xs.push_back(alpha - 1);
  xs.push_back(alpha);
  if (alpha < kMax) xs.push_back(alpha + 1);
  xs.push_back(kMax);
  for (int i = 0; i < 32; ++i) xs.push_back(RandomIn<In>(gen) & kMax);
  return xs;
}

template <typename In>
void ReportMismatch(const char *scheme, int in_bits, const char *what, In x) {
  std::fprintf(stderr, "%s in_bits=%d %s mismatch at x=", scheme, in_bits, what);
  PrintVal(stderr, x);
  std::fprintf(stderr, "\n");
}

// Bit-exact fast-path vs generic Dpf::Eval for both parties, plus the
// share-sum property (beta at alpha, identity elsewhere).
template <int in_bits, typename Group, typename Prg, typename In>
bool RunDpfEquivalence(In alpha, uint64_t tag) {
  std::mt19937 gen = MakeGen(tag);
  std::mt19937_64 gen64 = MakeGen64(tag);
  int4 s0s[2] = {MakeSeed(gen), MakeSeed(gen)};
  Prg prg = MakePrg<Prg>();
  fss::Dpf<in_bits, Group, Prg, In> dpf{prg};
  typename fss::Dpf<in_bits, Group, Prg, In>::Key k0, k1;
  int4 b_buf = MakeBBuf();
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);
  const fss::cpu::DpfPointKey<in_bits> pk0 = fss::cpu::DpfPackPoint<in_bits, Group, Prg, In>(k0);
  const fss::cpu::DpfPointKey<in_bits> pk1 = fss::cpu::DpfPackPoint<in_bits, Group, Prg, In>(k1);

  const int4 beta_g = Group::From(b_buf).Into();
  const int4 zero = {0, 0, 0, 0};
  bool ok = true;
  for (In x : MakePoints<in_bits, In>(alpha, gen64)) {
    const int4 e0 = dpf.Eval(false, k0, x);
    const int4 e1 = dpf.Eval(true, k1, x);
    const int4 f0 = fss::cpu::DpfEvalPoint<in_bits, Group, Prg, In>(false, pk0, x, prg);
    const int4 f1 = fss::cpu::DpfEvalPoint<in_bits, Group, Prg, In>(true, pk1, x, prg);
    if (memcmp(&e0, &f0, sizeof(int4)) != 0 || memcmp(&e1, &f1, sizeof(int4)) != 0) {
      ReportMismatch("dpf", in_bits, "equivalence", x);
      ok = false;
    }
    const int4 sum = (Group::From(e0) + Group::From(e1)).Into();
    const int4 expected = x == alpha ? beta_g : zero;
    if (memcmp(&sum, &expected, sizeof(int4)) != 0) {
      ReportMismatch("dpf", in_bits, "share sum", x);
      ok = false;
    }
  }
  return ok;
}

// Bit-exact fast-path vs generic HalfTreeDpf::Eval for both parties, plus
// the share-sum property.
template <int in_bits, typename Group, typename Prg, typename In>
bool RunHalfTreeEquivalence(In alpha, uint64_t tag) {
  std::mt19937 gen = MakeGen(tag);
  std::mt19937_64 gen64 = MakeGen64(tag);
  std::uniform_int_distribution<int> dis;
  int4 s0s[2] = {MakeSeed(gen), MakeSeed(gen)};
  int4 hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
  Prg prg = MakePrg<Prg>();
  fss::HalfTreeDpf<in_bits, Group, Prg, In> dpf{prg, hash_key};
  typename fss::HalfTreeDpf<in_bits, Group, Prg, In>::Key k0, k1;
  int4 b_buf = MakeBBuf();
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, b_buf);
  const fss::cpu::HalfTreeDpfPointKey<in_bits> pk0 = fss::cpu::HalfTreeDpfPackPoint<in_bits, Group, Prg, In>(k0);
  const fss::cpu::HalfTreeDpfPointKey<in_bits> pk1 = fss::cpu::HalfTreeDpfPackPoint<in_bits, Group, Prg, In>(k1);

  const int4 beta_g = Group::From(b_buf).Into();
  const int4 zero = {0, 0, 0, 0};
  bool ok = true;
  for (In x : MakePoints<in_bits, In>(alpha, gen64)) {
    const int4 e0 = dpf.Eval(false, k0, x);
    const int4 e1 = dpf.Eval(true, k1, x);
    const int4 f0 = fss::cpu::HalfTreeDpfEvalPoint<in_bits, Group, Prg, In>(false, pk0, x, dpf.prg, dpf.hash_key);
    const int4 f1 = fss::cpu::HalfTreeDpfEvalPoint<in_bits, Group, Prg, In>(true, pk1, x, dpf.prg, dpf.hash_key);
    if (memcmp(&e0, &f0, sizeof(int4)) != 0 || memcmp(&e1, &f1, sizeof(int4)) != 0) {
      ReportMismatch("half-tree dpf", in_bits, "equivalence", x);
      ok = false;
    }
    const int4 sum = (Group::From(e0) + Group::From(e1)).Into();
    const int4 expected = x == alpha ? beta_g : zero;
    if (memcmp(&sum, &expected, sizeof(int4)) != 0) {
      ReportMismatch("half-tree dpf", in_bits, "share sum", x);
      ok = false;
    }
  }
  return ok;
}

// in_bits boundaries {1, 2, 7, 8, 16, 20, 32} with the narrowest fitting In,
// alpha boundary values at in_bits 8 and 1.
template <typename Group, typename Prg>
bool RunDpfGrid() {
  bool ok = true;
  ok &= RunDpfEquivalence<1, Group, Prg, uint8_t>(0, 101);
  ok &= RunDpfEquivalence<1, Group, Prg, uint8_t>(1, 102);
  ok &= RunDpfEquivalence<2, Group, Prg, uint8_t>(2, 103);
  ok &= RunDpfEquivalence<7, Group, Prg, uint8_t>(107, 104);
  ok &= RunDpfEquivalence<8, Group, Prg, uint8_t>(107, 105);
  ok &= RunDpfEquivalence<8, Group, Prg, uint8_t>(0, 106);
  ok &= RunDpfEquivalence<8, Group, Prg, uint8_t>(255, 107);
  ok &= RunDpfEquivalence<16, Group, Prg, uint16_t>(107, 108);
  ok &= RunDpfEquivalence<20, Group, Prg, uint>(1000067, 109);
  ok &= RunDpfEquivalence<32, Group, Prg, uint>(0xdeadbeef, 110);
  return ok;
}

// One __uint128_t In case wider than 64 domain bits.
template <typename Group, typename Prg>
bool RunDpfU128In() {
  constexpr int kBits = 100;
  return RunDpfEquivalence<kBits, Group, Prg, __uint128_t>((__uint128_t{1} << 99) + 12345, 111);
}

template <typename Group, typename Prg>
bool RunHalfTreeGrid() {
  bool ok = true;
  // in_bits == 1 exercises the last-level-only path with both alpha bits.
  ok &= RunHalfTreeEquivalence<1, Group, Prg, uint8_t>(0, 201);
  ok &= RunHalfTreeEquivalence<1, Group, Prg, uint8_t>(1, 202);
  ok &= RunHalfTreeEquivalence<2, Group, Prg, uint8_t>(2, 203);
  ok &= RunHalfTreeEquivalence<7, Group, Prg, uint8_t>(107, 204);
  ok &= RunHalfTreeEquivalence<8, Group, Prg, uint8_t>(107, 205);
  // Last-level LCW selection at alpha = 0 (a_n = 0) and alpha = max (a_n = 1).
  ok &= RunHalfTreeEquivalence<8, Group, Prg, uint8_t>(0, 206);
  ok &= RunHalfTreeEquivalence<8, Group, Prg, uint8_t>(255, 207);
  ok &= RunHalfTreeEquivalence<16, Group, Prg, uint16_t>(107, 208);
  ok &= RunHalfTreeEquivalence<20, Group, Prg, uint>(1000067, 209);
  ok &= RunHalfTreeEquivalence<32, Group, Prg, uint>(0xdeadbeef, 210);
  return ok;
}

template <typename Group, typename Prg>
bool RunHalfTreeU128In() {
  constexpr int kBits = 100;
  return RunHalfTreeEquivalence<kBits, Group, Prg, __uint128_t>((__uint128_t{1} << 99) + 12345, 211);
}

}  // namespace

#ifdef POINT_EVAL_CPU_STANDALONE

namespace {

// Times one point evaluation per iteration; the accumulator keeps every
// iteration observable so the loop cannot be deleted.
template <int in_bits, typename Group, typename Prg, typename In>
void TimeDpfPoint(int iters) {
  using DpfType = fss::Dpf<in_bits, Group, Prg, In>;
  Prg prg = MakePrg<Prg>();
  DpfType dpf{prg};
  int4 s0s[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  typename DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), 42, MakeBBuf());
  const fss::cpu::DpfPointKey<in_bits> pk = fss::cpu::DpfPackPoint<in_bits, Group, Prg, In>(k0);
  In x = 100;

  int4 acc = {0, 0, 0, 0};
  for (int i = 0; i < 2000; ++i) {
    int4 y = fss::cpu::DpfEvalPoint<in_bits, Group, Prg, In>(false, pk, x, prg);
    acc.x ^= y.x;
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    int4 y = fss::cpu::DpfEvalPoint<in_bits, Group, Prg, In>(false, pk, x, prg);
    acc.x ^= y.x;
    acc.y ^= y.y;
    acc.z ^= y.z;
    acc.w ^= y.w;
  }
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < 2000; ++i) {
    int4 y = dpf.Eval(false, k0, x);
    acc.x ^= y.x;
  }
  const auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    int4 y = dpf.Eval(false, k0, x);
    acc.x ^= y.x;
    acc.y ^= y.y;
    acc.z ^= y.z;
    acc.w ^= y.w;
  }
  const auto t3 = std::chrono::steady_clock::now();
  const double fast = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
  const double generic = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;
  std::printf("Dpf<%d> point eval: fast %.1f ns, generic %.1f ns (sink %08x)\n", in_bits, fast, generic, acc.x);
}

template <int in_bits, typename Group, typename Prg, typename In>
void TimeHalfTreeDpfPoint(int iters) {
  using DpfType = fss::HalfTreeDpf<in_bits, Group, Prg, In>;
  Prg prg = MakePrg<Prg>();
  int4 hash_key = {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, 0x2468ace0};
  DpfType dpf{prg, hash_key};
  int4 s0s[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  typename DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), 42, MakeBBuf());
  const fss::cpu::HalfTreeDpfPointKey<in_bits> pk = fss::cpu::HalfTreeDpfPackPoint<in_bits, Group, Prg, In>(k0);
  In x = 100;

  int4 acc = {0, 0, 0, 0};
  for (int i = 0; i < 2000; ++i) {
    int4 y = fss::cpu::HalfTreeDpfEvalPoint<in_bits, Group, Prg, In>(false, pk, x, dpf.prg, dpf.hash_key);
    acc.x ^= y.x;
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    int4 y = fss::cpu::HalfTreeDpfEvalPoint<in_bits, Group, Prg, In>(false, pk, x, dpf.prg, dpf.hash_key);
    acc.x ^= y.x;
    acc.y ^= y.y;
    acc.z ^= y.z;
    acc.w ^= y.w;
  }
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < 2000; ++i) {
    int4 y = dpf.Eval(false, k0, x);
    acc.x ^= y.x;
  }
  const auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    int4 y = dpf.Eval(false, k0, x);
    acc.x ^= y.x;
    acc.y ^= y.y;
    acc.z ^= y.z;
    acc.w ^= y.w;
  }
  const auto t3 = std::chrono::steady_clock::now();
  const double fast = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
  const double generic = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;
  std::printf("HalfTreeDpf<%d> point eval: fast %.1f ns, generic %.1f ns (sink %08x)\n", in_bits, fast, generic, acc.x);
}

struct Case {
  const char *name;
  bool (*fn)();
};

}  // namespace

int main(int argc, char **argv) {
  const Case cases[] = {
      {"DpfPointEvalCpuTest.BytesAesRawEquivalence", [] { return RunDpfGrid<BytesGroup, Aes2>(); }},
      {"DpfPointEvalCpuTest.Uint128ChaChaEquivalence", [] { return RunDpfGrid<Uint127Group, ChaCha2>(); }},
      {"DpfPointEvalCpuTest.Uint8AesRawEquivalence", [] { return RunDpfGrid<Uint8Group, Aes2>(); }},
      {"DpfPointEvalCpuTest.Uint128InAesRawEquivalence", [] { return RunDpfU128In<BytesGroup, Aes2>(); }},
      {"DpfPointEvalCpuTest.Uint128InChaChaEquivalence", [] { return RunDpfU128In<Uint127Group, ChaCha2>(); }},
      {"HalfTreePointEvalCpuTest.BytesAesRawEquivalence", [] { return RunHalfTreeGrid<BytesGroup, Aes1>(); }},
      {"HalfTreePointEvalCpuTest.Uint128ChaChaEquivalence", [] { return RunHalfTreeGrid<Uint127Group, ChaCha1>(); }},
      {"HalfTreePointEvalCpuTest.Uint8AesRawEquivalence", [] { return RunHalfTreeGrid<Uint8Group, Aes1>(); }},
      {"HalfTreePointEvalCpuTest.Uint128InAesRawEquivalence", [] { return RunHalfTreeU128In<BytesGroup, Aes1>(); }},
  };
  int failed = 0;
  for (const Case &c : cases) {
    const bool ok = c.fn();
    std::printf("%-52s %s\n", c.name, ok ? "PASS" : "FAIL");
    if (!ok) ++failed;
  }
  std::printf("%zu cases, %d failed\n", sizeof(cases) / sizeof(cases[0]), failed);

  if (argc > 1 && std::strcmp(argv[1], "--bench") == 0 && failed == 0) {
    constexpr int kIters = 500000;
    TimeDpfPoint<20, fss::group::Uint<uint64_t>, Aes2, uint>(kIters);
    TimeDpfPoint<20, BytesGroup, Aes2, uint>(kIters);
    TimeHalfTreeDpfPoint<20, fss::group::Uint<uint64_t>, Aes1, uint>(kIters);
    TimeHalfTreeDpfPoint<20, BytesGroup, Aes1, uint>(kIters);
  }
  return failed == 0 ? 0 : 1;
}

#else

TEST(DpfPointEvalCpuTest, BytesAesRawEquivalence) {
  EXPECT_TRUE((RunDpfGrid<BytesGroup, Aes2>()));
}

TEST(DpfPointEvalCpuTest, Uint128ChaChaEquivalence) {
  EXPECT_TRUE((RunDpfGrid<Uint127Group, ChaCha2>()));
}

TEST(DpfPointEvalCpuTest, Uint8AesRawEquivalence) {
  EXPECT_TRUE((RunDpfGrid<Uint8Group, Aes2>()));
}

TEST(DpfPointEvalCpuTest, Uint128InAesRawEquivalence) {
  EXPECT_TRUE((RunDpfU128In<BytesGroup, Aes2>()));
}

TEST(DpfPointEvalCpuTest, Uint128InChaChaEquivalence) {
  EXPECT_TRUE((RunDpfU128In<Uint127Group, ChaCha2>()));
}

TEST(HalfTreePointEvalCpuTest, BytesAesRawEquivalence) {
  EXPECT_TRUE((RunHalfTreeGrid<BytesGroup, Aes1>()));
}

TEST(HalfTreePointEvalCpuTest, Uint128ChaChaEquivalence) {
  EXPECT_TRUE((RunHalfTreeGrid<Uint127Group, ChaCha1>()));
}

TEST(HalfTreePointEvalCpuTest, Uint8AesRawEquivalence) {
  EXPECT_TRUE((RunHalfTreeGrid<Uint8Group, Aes1>()));
}

TEST(HalfTreePointEvalCpuTest, Uint128InAesRawEquivalence) {
  EXPECT_TRUE((RunHalfTreeU128In<BytesGroup, Aes1>()));
}

#endif  // POINT_EVAL_CPU_STANDALONE
