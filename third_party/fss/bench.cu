#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <fss/prg/chacha.cuh>
#include <vector>
#include <fss/eval_all_gpu.cuh>
#include <array>
#include <memory>
#include <span>
#include <fss/point_eval_cpu.cuh>
#include <fss/point_eval_gpu.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include <fss/prp/chacha_feistel.cuh>
#ifdef FSS_BENCH_AES_NI
#include <fss/prg/aes128_mmo_raw.cuh>
#endif

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
#ifndef FSS_BENCH_NUM_KEYS
#define FSS_BENCH_NUM_KEYS (1 << 20)
#endif
#ifndef FSS_BENCH_THREADS_PER_BLOCK
#define FSS_BENCH_THREADS_PER_BLOCK 256
#endif
constexpr int kInBits = FSS_BENCH_DOMAIN_BITS;
constexpr int kN = FSS_BENCH_NUM_KEYS;
constexpr int kThreadsPerBlock = FSS_BENCH_THREADS_PER_BLOCK;
static_assert(kInBits >= 2 && kInBits <= 30);
static_assert(kN > 0 && kThreadsPerBlock > 0 && kThreadsPerBlock <= 1024);
constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

using BytesGroup = fss::group::Bytes;

// Wrap Uint<__uint128_t, 2^127> in a plain struct to avoid embedding a 128-bit
// integer literal in __global__ function stubs (nvcc stub gen bug).
struct UintGroup {
  using Impl = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
  Impl impl;

  __host__ __device__ UintGroup operator+(UintGroup rhs) const {
    return {impl + rhs.impl};
  }
  __host__ __device__ UintGroup operator-() const {
    return {-impl};
  }
  __host__ __device__ static UintGroup From(int4 buf) {
    UintGroup g;
    g.impl = Impl::From(buf);
    return g;
  }
  __host__ __device__ int4 Into() const {
    return impl.Into();
  }
};
static_assert(Groupable<UintGroup>);

#define CUDA_CHECK(x) \
  do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "cuda error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

static bool HasGpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

static const int4 kSeeds[2] = {
    {0x11111111, 0x22222222, 0x33333333, 0x44444440},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
};
static constexpr uint32_t kAlpha = 42;
static const int4 kBeta = {7, 0, 0, 0};

// ============================================================
// CPU AES context (mul=2 for DPF, mul=4 for DCF)
// ============================================================

static constexpr unsigned char kAesKeys[4][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
    {0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {0x16, 0x25, 0x34, 0x43, 0x52, 0x61, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

#ifdef FSS_BENCH_AES_NI
template <int mul>
using CpuPrg = fss::prg::Aes128MmoRaw<mul>;
template <int mul>
struct AesCtx {
  CpuPrg<mul> prg{kAesKeys};
};
#else
template <int mul>
using CpuPrg = fss::prg::Aes128Mmo<mul>;
template <int mul>
struct AesCtx {
  cuda::std::array<EVP_CIPHER_CTX *, mul> ctxs;
  fss::prg::Aes128Mmo<mul> prg;

  AesCtx() : ctxs(MakeCtxs()), prg(ctxs) {}
  ~AesCtx() {
    fss::prg::Aes128Mmo<mul>::FreeCtxs(ctxs);
  }

private:
  static cuda::std::array<EVP_CIPHER_CTX *, mul> MakeCtxs() {
    const unsigned char *keys[mul];
    for (int i = 0; i < mul; i++) keys[i] = kAesKeys[i];
    return fss::prg::Aes128Mmo<mul>::CreateCtxs(keys);
  }
};

#endif

// Check both parties at the threshold and its neighbors before timing.
template <typename Group, bool comparison, typename Scheme>
static void CheckCpuScheme(Scheme &scheme, const typename Scheme::Key &k0, const typename Scheme::Key &k1) {
  const uint32_t points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  for (uint32_t x : points) {
    auto y0 = Group::From(scheme.Eval(false, k0, x));
    auto y1 = Group::From(scheme.Eval(true, k1, x));
    bool nonzero = comparison ? x < kAlpha : x == kAlpha;
    int4 expected = Group::From(nonzero ? kBeta : int4{0, 0, 0, 0}).Into();
    int4 actual = (y0 + y1).Into();
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "scheme correctness mismatch: %s at input %u\n", comparison ? "DCF" : "DPF", x);
      exit(1);
    }
  }
}


template <typename Group, bool comparison, typename Scheme>
static void CheckCpuFull(Scheme &scheme, const typename Scheme::Key &k0, const typename Scheme::Key &k1) {
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> first(n), second(n);
  scheme.EvalAll(false, k0, cuda::std::span<int4>(first.data(), n));
  scheme.EvalAll(true, k1, cuda::std::span<int4>(second.data(), n));
  for (size_t x = 0; x < n; ++x) {
    const bool selected = comparison ? x < kAlpha : x == kAlpha;
    int4 actual = (Group::From(first[x]) + Group::From(second[x])).Into();
    int4 expected = Group::From(selected ? kBeta : int4{0, 0, 0, 0}).Into();
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "CPU full reconstruction mismatch at input %zu\n", x);
      exit(EXIT_FAILURE);
    }
  }
}

// ============================================================
// CPU DPF benchmarks
// ============================================================

template <typename Group>
static void BM_CpuDpfGen(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, k0, k1);
  for (auto _ : state) {
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

template <typename Group>
static void BM_CpuDpfEval(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dpf.Eval(false, k0, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuDpfEvalAll(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, k0, k1);
  CheckCpuFull<Group, false>(dpf, k0, k1);
  for (auto _ : state) {
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys.data(), n));
    benchmark::DoNotOptimize(ys.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================
// CPU DCF benchmarks
// ============================================================

template <typename Group>
static void BM_CpuDcfGen(benchmark::State &state) {
  using DcfT = fss::Dcf<kInBits, Group, CpuPrg<4>, uint32_t, fss::DcfPred::kLt, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DcfT::Key k0, k1;
  AesCtx<4> ctx;
  DcfT dcf{ctx.prg};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, true>(dcf, k0, k1);
  for (auto _ : state) {
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

template <typename Group>
static void BM_CpuDcfEval(benchmark::State &state) {
  using DcfT = fss::Dcf<kInBits, Group, CpuPrg<4>, uint32_t, fss::DcfPred::kLt, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DcfT::Key k0, k1;
  AesCtx<4> ctx;
  DcfT dcf{ctx.prg};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, true>(dcf, k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dcf.Eval(false, k0, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuDcfEvalAll(benchmark::State &state) {
  using DpfT = fss::Dcf<kInBits, Group, CpuPrg<4>, uint32_t, fss::DcfPred::kLt, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<4> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, true>(dpf, k0, k1);
  CheckCpuFull<Group, true>(dpf, k0, k1);
  for (auto _ : state) {
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys.data(), n));
    benchmark::DoNotOptimize(ys.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

// CPU registration
BENCHMARK(BM_CpuDpfGen<BytesGroup>)->Name("fss/CPU/DPF-bytes/Gen");
BENCHMARK(BM_CpuDpfGen<UintGroup>)->Name("fss/CPU/DPF-uint/Gen");
BENCHMARK(BM_CpuDpfEval<BytesGroup>)->Name("fss/CPU/DPF-bytes/Eval");
BENCHMARK(BM_CpuDpfEval<UintGroup>)->Name("fss/CPU/DPF-uint/Eval");
BENCHMARK(BM_CpuDpfEvalAll<BytesGroup>)->Name("fss/CPU/DPF-bytes/EvalAll");
BENCHMARK(BM_CpuDpfEvalAll<UintGroup>)->Name("fss/CPU/DPF-uint/EvalAll");
BENCHMARK(BM_CpuDcfGen<BytesGroup>)->Name("fss/CPU/DCF-bytes/Gen");
BENCHMARK(BM_CpuDcfGen<UintGroup>)->Name("fss/CPU/DCF-uint/Gen");
BENCHMARK(BM_CpuDcfEval<BytesGroup>)->Name("fss/CPU/DCF-bytes/Eval");
BENCHMARK(BM_CpuDcfEval<UintGroup>)->Name("fss/CPU/DCF-uint/Eval");

BENCHMARK(BM_CpuDcfEvalAll<BytesGroup>)->Name("fss/CPU/DCF-bytes/EvalAll");
BENCHMARK(BM_CpuDcfEvalAll<UintGroup>)->Name("fss/CPU/DCF-uint/EvalAll");

// ============================================================
// CPU HalfTree DPF benchmarks
// ============================================================

static const int4 kHalfTreeHashKey = {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)};

// Same checks as CheckCpuScheme/CheckCpuFull, kept separate to preserve the
// distinct mismatch messages of the HalfTree benchmarks.
template <typename Group, bool comparison, typename Scheme>
static void CheckCpuSchemeWith(Scheme &scheme, const typename Scheme::Key &k0, const typename Scheme::Key &k1) {
  const uint32_t points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  for (uint32_t x : points) {
    auto y0 = Group::From(scheme.Eval(false, k0, x));
    auto y1 = Group::From(scheme.Eval(true, k1, x));
    bool nonzero = comparison ? x < kAlpha : x == kAlpha;
    int4 expected = Group::From(nonzero ? kBeta : int4{0, 0, 0, 0}).Into();
    int4 actual = (y0 + y1).Into();
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "scheme correctness mismatch at input %u\n", x);
      exit(1);
    }
  }
}

template <typename Group, bool comparison, typename Scheme>
static void CheckCpuFullWith(Scheme &scheme, const typename Scheme::Key &k0, const typename Scheme::Key &k1) {
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> first(n), second(n);
  scheme.EvalAll(false, k0, cuda::std::span<int4>(first.data(), n));
  scheme.EvalAll(true, k1, cuda::std::span<int4>(second.data(), n));
  for (size_t x = 0; x < n; ++x) {
    const bool selected = comparison ? x < kAlpha : x == kAlpha;
    int4 actual = (Group::From(first[x]) + Group::From(second[x])).Into();
    int4 expected = Group::From(selected ? kBeta : int4{0, 0, 0, 0}).Into();
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "CPU full reconstruction mismatch at input %zu\n", x);
      exit(EXIT_FAILURE);
    }
  }
}

template <typename Group>
static void BM_CpuHalfTreeDpfGen(benchmark::State &state) {
  using DpfT = fss::HalfTreeDpf<kInBits, Group, CpuPrg<1>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<1> ctx;
  DpfT dpf{ctx.prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuSchemeWith<Group, false>(dpf, k0, k1);
  for (auto _ : state) {
    dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

template <typename Group>
static void BM_CpuHalfTreeDpfEval(benchmark::State &state) {
  using DpfT = fss::HalfTreeDpf<kInBits, Group, CpuPrg<1>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<1> ctx;
  DpfT dpf{ctx.prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuSchemeWith<Group, false>(dpf, k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dpf.Eval(false, k0, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuHalfTreeDpfEvalAll(benchmark::State &state) {
  using DpfT = fss::HalfTreeDpf<kInBits, Group, CpuPrg<1>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<1> ctx;
  DpfT dpf{ctx.prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuSchemeWith<Group, false>(dpf, k0, k1);
  CheckCpuFullWith<Group, false>(dpf, k0, k1);
  for (auto _ : state) {
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys.data(), n));
    benchmark::DoNotOptimize(ys.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_CpuHalfTreeDpfGen<BytesGroup>)->Name("fss/CPU/HalfTreeDPF-bytes/Gen");
BENCHMARK(BM_CpuHalfTreeDpfEval<BytesGroup>)->Name("fss/CPU/HalfTreeDPF-bytes/Eval");
BENCHMARK(BM_CpuHalfTreeDpfEvalAll<BytesGroup>)->Name("fss/CPU/HalfTreeDPF-bytes/EvalAll");

// ============================================================
// CPU packed HalfTree DPF benchmarks (experimental, 1-bit outputs)
// ============================================================

static bool EqualInt4(int4 a, int4 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <typename Prg>
static void CheckCpuPackedFull(Prg &prg) {
  using DpfT = fss::PackedHalfTreeDpf<kInBits, 1, Prg, uint32_t, 0>;
  constexpr size_t blocks = size_t{1} << (kInBits - 7);
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  DpfT dpf{prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, 1);
  std::vector<int4> first(blocks), second(blocks);
  dpf.EvalAll(false, k0, cuda::std::span<int4>(first.data(), blocks));
  dpf.EvalAll(true, k1, cuda::std::span<int4>(second.data(), blocks));
  int4 zero = {0, 0, 0, 0};
  int4 beta_block = DpfT::PackBeta(kAlpha, 1);
  for (size_t i = 0; i < blocks; ++i) {
    int4 diff = fss::util::Xor(first[i], second[i]);
    int4 expected = (i == kAlpha / DpfT::kLanes) ? beta_block : zero;
    if (!EqualInt4(diff, expected)) {
      fprintf(stderr, "packed full reconstruction mismatch at block %zu\n", i);
      exit(EXIT_FAILURE);
    }
  }
}

static void BM_CpuPackedHalfTreeDpfEvalAll(benchmark::State &state) {
  using DpfT = fss::PackedHalfTreeDpf<kInBits, 1, CpuPrg<1>, uint32_t, 0>;
  constexpr size_t n = size_t{1} << kInBits;
  constexpr size_t blocks = n / DpfT::kLanes;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<1> ctx;
  DpfT dpf{ctx.prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, 1);
  CheckCpuPackedFull(ctx.prg);
  std::vector<int4> ys(blocks);
  for (auto _ : state) {
    dpf.EvalAll(false, k0, cuda::std::span<int4>(ys.data(), blocks));
    benchmark::DoNotOptimize(ys.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_CpuPackedHalfTreeDpfEvalAll)->Name("fss/CPU/PackedHalfTreeDPF-bits1/EvalAll");

// ============================================================
// CPU packed-key point evaluation benchmarks (point_eval_cpu)
// ============================================================

// The packed-key fast path must stay bit-exact with the scheme Eval on both
// parties before any timing runs; a mismatch skips the benchmark loudly.
template <typename Scheme, typename PackPoint, typename EvalPoint>
static bool CheckCpuPackedPoint(benchmark::State &state, Scheme &scheme, PackPoint pack_point, EvalPoint eval_point,
    const typename Scheme::Key &k0, const typename Scheme::Key &k1) {
  const auto p0 = pack_point(k0);
  const auto p1 = pack_point(k1);
  const uint32_t points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  for (uint32_t x : points) {
    for (bool b : {false, true}) {
      int4 expected = scheme.Eval(b, b ? k1 : k0, x);
      int4 actual = eval_point(b, b ? p1 : p0, x);
      if (!EqualInt4(expected, actual)) {
        state.SkipWithError("packed-key point evaluation mismatch with the scheme Eval");
        return false;
      }
    }
  }
  return true;
}

// par_depth stays at its default because the pack helpers key on the default
// Dpf / HalfTreeDpf type and par_depth only affects EvalAll, which these
// point benchmarks never call.
template <typename Group>
static void BM_CpuDpfPackedKeyEval(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, k0, k1);
  if (!CheckCpuPackedPoint(state, dpf,
          [](const typename DpfT::Key &key) { return fss::cpu::DpfPackPoint<kInBits, Group, CpuPrg<2>, uint32_t>(key); },
          [&dpf](bool b, const fss::cpu::DpfPointKey<kInBits> &key, uint32_t x) {
            return fss::cpu::DpfEvalPoint<kInBits, Group, CpuPrg<2>, uint32_t>(b, key, x, dpf.prg);
          }, k0, k1)) {
    return;
  }
  const fss::cpu::DpfPointKey<kInBits> p0 = fss::cpu::DpfPackPoint<kInBits, Group, CpuPrg<2>, uint32_t>(k0);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = fss::cpu::DpfEvalPoint<kInBits, Group, CpuPrg<2>, uint32_t>(false, p0, x, ctx.prg);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuDpfPackedKeyOneShotEval(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, k0, k1);
  if (!CheckCpuPackedPoint(state, dpf,
          [](const typename DpfT::Key &key) { return fss::cpu::DpfPackPoint<kInBits, Group, CpuPrg<2>, uint32_t>(key); },
          [&dpf](bool b, const fss::cpu::DpfPointKey<kInBits> &key, uint32_t x) {
            return fss::cpu::DpfEvalPoint<kInBits, Group, CpuPrg<2>, uint32_t>(b, key, x, dpf.prg);
          }, k0, k1)) {
    return;
  }
  uint32_t x = 0;
  for (auto _ : state) {
    const fss::cpu::DpfPointKey<kInBits> p0 = fss::cpu::DpfPackPoint<kInBits, Group, CpuPrg<2>, uint32_t>(k0);
    int4 y = fss::cpu::DpfEvalPoint<kInBits, Group, CpuPrg<2>, uint32_t>(false, p0, x, ctx.prg);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuHalfTreeDpfPackedKeyEval(benchmark::State &state) {
  using DpfT = fss::HalfTreeDpf<kInBits, Group, CpuPrg<1>, uint32_t>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<1> ctx;
  DpfT dpf{ctx.prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuSchemeWith<Group, false>(dpf, k0, k1);
  if (!CheckCpuPackedPoint(state, dpf,
          [](const typename DpfT::Key &key) {
            return fss::cpu::HalfTreeDpfPackPoint<kInBits, Group, CpuPrg<1>, uint32_t>(key);
          },
          [&dpf](bool b, const fss::cpu::HalfTreeDpfPointKey<kInBits> &key, uint32_t x) {
            return fss::cpu::HalfTreeDpfEvalPoint<kInBits, Group, CpuPrg<1>, uint32_t>(b, key, x, dpf.prg, dpf.hash_key);
          }, k0, k1)) {
    return;
  }
  const fss::cpu::HalfTreeDpfPointKey<kInBits> p0 =
      fss::cpu::HalfTreeDpfPackPoint<kInBits, Group, CpuPrg<1>, uint32_t>(k0);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = fss::cpu::HalfTreeDpfEvalPoint<kInBits, Group, CpuPrg<1>, uint32_t>(false, p0, x, ctx.prg, kHalfTreeHashKey);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuHalfTreeDpfPackedKeyOneShotEval(benchmark::State &state) {
  using DpfT = fss::HalfTreeDpf<kInBits, Group, CpuPrg<1>, uint32_t>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  AesCtx<1> ctx;
  DpfT dpf{ctx.prg, kHalfTreeHashKey};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuSchemeWith<Group, false>(dpf, k0, k1);
  if (!CheckCpuPackedPoint(state, dpf,
          [](const typename DpfT::Key &key) {
            return fss::cpu::HalfTreeDpfPackPoint<kInBits, Group, CpuPrg<1>, uint32_t>(key);
          },
          [&dpf](bool b, const fss::cpu::HalfTreeDpfPointKey<kInBits> &key, uint32_t x) {
            return fss::cpu::HalfTreeDpfEvalPoint<kInBits, Group, CpuPrg<1>, uint32_t>(b, key, x, dpf.prg, dpf.hash_key);
          }, k0, k1)) {
    return;
  }
  uint32_t x = 0;
  for (auto _ : state) {
    const fss::cpu::HalfTreeDpfPointKey<kInBits> p0 =
        fss::cpu::HalfTreeDpfPackPoint<kInBits, Group, CpuPrg<1>, uint32_t>(k0);
    int4 y = fss::cpu::HalfTreeDpfEvalPoint<kInBits, Group, CpuPrg<1>, uint32_t>(false, p0, x, ctx.prg, kHalfTreeHashKey);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

BENCHMARK(BM_CpuDpfPackedKeyEval<BytesGroup>)->Name("fss/CPU/DPF-bytes/PackedKey/Eval");
BENCHMARK(BM_CpuDpfPackedKeyEval<UintGroup>)->Name("fss/CPU/DPF-uint/PackedKey/Eval");
BENCHMARK(BM_CpuDpfPackedKeyOneShotEval<BytesGroup>)->Name("fss/CPU/DPF-bytes/PackedKeyOneShot/Eval");
BENCHMARK(BM_CpuDpfPackedKeyOneShotEval<UintGroup>)->Name("fss/CPU/DPF-uint/PackedKeyOneShot/Eval");
BENCHMARK(BM_CpuHalfTreeDpfPackedKeyEval<BytesGroup>)->Name("fss/CPU/HalfTreeDPF-bytes/PackedKey/Eval");
BENCHMARK(BM_CpuHalfTreeDpfPackedKeyOneShotEval<BytesGroup>)->Name("fss/CPU/HalfTreeDPF-bytes/PackedKeyOneShot/Eval");

// ============================================================
// CPU Grotto DCF benchmarks
// ============================================================

using GrottoDcfT = fss::GrottoDcf<kInBits, CpuPrg<2>, uint32_t, 0>;

static void CheckCpuGrottoFull(const typename GrottoDcfT::Key &k0, const typename GrottoDcfT::Key &k1) {
  constexpr size_t n = size_t{1} << kInBits;
  AesCtx<2> ctx;
  GrottoDcfT dcf{ctx.prg};
  auto first = std::make_unique<bool[]>(n);
  auto second = std::make_unique<bool[]>(n);
  dcf.EvalAll(false, k0, cuda::std::span<bool>(first.get(), n));
  dcf.EvalAll(true, k1, cuda::std::span<bool>(second.get(), n));
  for (size_t x = 0; x < n; ++x) {
    if ((first[x] ^ second[x]) != (x >= kAlpha)) {
      fprintf(stderr, "grotto full reconstruction mismatch at input %zu\n", x);
      exit(EXIT_FAILURE);
    }
  }
}

static void BM_CpuGrottoDcfGen(benchmark::State &state) {
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename GrottoDcfT::Key k0, k1;
  AesCtx<2> ctx;
  GrottoDcfT dcf{ctx.prg};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha);
  for (auto _ : state) {
    dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

static void BM_CpuGrottoDcfEval(benchmark::State &state) {
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename GrottoDcfT::Key k0, k1;
  AesCtx<2> ctx;
  GrottoDcfT dcf{ctx.prg};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha);
  auto tree = dcf.Preprocess(false, k0);
  uint32_t x = 0;
  for (auto _ : state) {
    bool y = GrottoDcfT::Eval(tree, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

static void BM_CpuGrottoDcfEvalAll(benchmark::State &state) {
  constexpr size_t n = size_t{1} << kInBits;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename GrottoDcfT::Key k0, k1;
  AesCtx<2> ctx;
  GrottoDcfT dcf{ctx.prg};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha);
  CheckCpuGrottoFull(k0, k1);
  auto ys = std::make_unique<bool[]>(n);
  for (auto _ : state) {
    auto tree = dcf.Preprocess(false, k0);
    dcf.EvalAll(false, k0, cuda::std::span<bool>(ys.get(), n));
    benchmark::DoNotOptimize(tree.p.get());
    benchmark::DoNotOptimize(ys.get());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_CpuGrottoDcfGen)->Name("fss/CPU/GrottoDCF/Gen");
BENCHMARK(BM_CpuGrottoDcfEval)->Name("fss/CPU/GrottoDCF/Eval");
BENCHMARK(BM_CpuGrottoDcfEvalAll)->Name("fss/CPU/GrottoDCF/EvalAll");

// ============================================================
// CPU DMPF / VDMPF benchmarks (eprint 2021/580, t points)
// ============================================================

constexpr int kDmpfMaxPoints = 64;
// ceil(3 * 2^kInBits / m) with m = ChBucket(64, 80) = 112 fits in 2^(kInBits-5)
// for every swept domain, so the inner DPF domain scales with N instead of a
// fixed 2^16 that would dominate small-N EvalAll through padded buckets.
constexpr int kDmpfBucketBits = kInBits - 5;
constexpr int kDmpfNumPoints = 64;

using DmpfT = fss::Dmpf<kInBits, kDmpfMaxPoints, kDmpfBucketBits, BytesGroup, CpuPrg<2>, fss::prp::Aes128Feistel, uint32_t>;
using VdmpfT = fss::Vdmpf<kInBits, kDmpfMaxPoints, kDmpfBucketBits, BytesGroup, CpuPrg<2>, fss::hash::Blake3,
    fss::hash::Blake3, fss::prp::Aes128Feistel, uint32_t>;
// The GPU benchmarks evaluate DMPF with the device-callable PRP; the CPU curves
// above keep Aes128Feistel as their own implementation.
using GpuDmpfT =
    fss::Dmpf<kInBits, kDmpfMaxPoints, kDmpfBucketBits, BytesGroup, fss::prg::ChaCha<2>, fss::prp::ChaChaFeistel, uint32_t>;
using GpuVdmpfT = fss::Vdmpf<kInBits, kDmpfMaxPoints, kDmpfBucketBits, BytesGroup, fss::prg::ChaCha<2>, fss::hash::Blake3,
    fss::hash::Blake3, fss::prp::ChaChaFeistel, uint32_t>;

static const int4 kDmpfSigma = {0x0f1e2d3c, 0x4b5a6978, static_cast<int>(0x8796a5b4u), static_cast<int>(0xc3d2e1f0u)};
static const int4 kDmpfHashIv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};

// Distinct alphas spread over the domain: stride is even so j * (stride + 1)
// stays odd-weighted and injective modulo 2^kInBits for j < kDmpfNumPoints.
static uint32_t DmpfAlpha(int j) {
  uint32_t stride = (1u << kInBits) / kDmpfNumPoints;
  return (j * (stride + 1)) & ((1u << kInBits) - 1);
}

struct DmpfPoints {
  std::array<uint32_t, kDmpfNumPoints> as{};
  std::array<int4, kDmpfNumPoints> bs{};
  DmpfPoints() {
    for (int j = 0; j < kDmpfNumPoints; ++j) {
      as[j] = DmpfAlpha(j);
      bs[j] = {(j + 1) * 11, 0, 0, 0};
    }
  }
};
static const DmpfPoints kDmpfPoints;

static int4 DmpfSeed(int i) {
  return {static_cast<int>(0x01010101u * (i + 1)), static_cast<int>(0x02020202u * (i + 1)),
      static_cast<int>(0x03030303u * (i + 1)), static_cast<int>(0x04040400u * (i + 1))};
}

template <typename Scheme>
static void GenDmpfKeys(Scheme &scheme, typename Scheme::Key &k0, typename Scheme::Key &k1) {
  constexpr int m = Scheme::m;
  cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
  for (int i = 0; i < m; ++i) {
    s0s[i] = {DmpfSeed(2 * i), DmpfSeed(2 * i + 1)};
  }
  int4 sigma = kDmpfSigma;
  if (scheme.Gen(k0, k1, sigma, cuda::std::span<cuda::std::array<int4, 2>, m>(s0s),
      cuda::std::span<const uint32_t>(kDmpfPoints.as), cuda::std::span<const int4>(kDmpfPoints.bs), kDmpfNumPoints) != 0) {
    fprintf(stderr, "dmpf/vdmpf gen failed after retries\n");
    exit(EXIT_FAILURE);
  }
}

static bool IsDmpfAlpha(uint32_t x, int *j) {
  for (int i = 0; i < kDmpfNumPoints; ++i) {
    if (kDmpfPoints.as[i] == x) {
      *j = i;
      return true;
    }
  }
  return false;
}

template <typename Scheme>
static void CheckCpuDmpfFull(Scheme &scheme, const typename Scheme::Key &k0, const typename Scheme::Key &k1) {
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> first(n), second(n);
  scheme.EvalAll(false, k0, cuda::std::span<int4>(first));
  scheme.EvalAll(true, k1, cuda::std::span<int4>(second));
  for (size_t x = 0; x < n; ++x) {
    int4 actual = (BytesGroup::From(first[x]) + BytesGroup::From(second[x])).Into();
    int j;
    int4 expected = IsDmpfAlpha(static_cast<uint32_t>(x), &j) ? kDmpfPoints.bs[j] : int4{0, 0, 0, 0};
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "dmpf full reconstruction mismatch at input %zu\n", x);
      exit(EXIT_FAILURE);
    }
  }
}

static void BM_CpuDmpfGen(benchmark::State &state) {
  AesCtx<2> ctx;
  fss::prp::Aes128Feistel prp;
  DmpfT dmpf{ctx.prg, prp};
  typename DmpfT::Key k0, k1;
  GenDmpfKeys(dmpf, k0, k1);
  for (auto _ : state) {
    GenDmpfKeys(dmpf, k0, k1);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

static void BM_CpuDmpfEval(benchmark::State &state) {
  AesCtx<2> ctx;
  fss::prp::Aes128Feistel prp;
  DmpfT dmpf{ctx.prg, prp};
  typename DmpfT::Key k0, k1;
  GenDmpfKeys(dmpf, k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    uint32_t xs[1] = {x};
    int4 y;
    dmpf.BatchEval(false, k0, cuda::std::span<const uint32_t>(xs), cuda::std::span<int4>(&y, 1));
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

static void BM_CpuDmpfEvalAll(benchmark::State &state) {
  constexpr size_t n = size_t{1} << kInBits;
  AesCtx<2> ctx;
  fss::prp::Aes128Feistel prp;
  DmpfT dmpf{ctx.prg, prp};
  typename DmpfT::Key k0, k1;
  GenDmpfKeys(dmpf, k0, k1);
  CheckCpuDmpfFull(dmpf, k0, k1);
  std::vector<int4> ys(n);
  for (auto _ : state) {
    dmpf.EvalAll(false, k0, cuda::std::span<int4>(ys));
    benchmark::DoNotOptimize(ys.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

static void CheckCpuVdmpfBatch(VdmpfT &vdmpf, const typename VdmpfT::Key &k0, const typename VdmpfT::Key &k1) {
  std::vector<uint32_t> xs;
  for (int j = 0; j < kDmpfNumPoints; ++j) {
    uint32_t a = kDmpfPoints.as[j];
    xs.push_back(a > 0 ? a - 1 : a);
    xs.push_back(a);
    xs.push_back((a + 1) & ((1u << kInBits) - 1));
  }
  xs.push_back(0);
  xs.push_back((1u << kInBits) - 1);
  std::vector<int4> ys0(xs.size()), ys1(xs.size());
  cuda::std::array<int4, 4> pi0, pi1;
  vdmpf.BatchEval(false, k0, cuda::std::span<const uint32_t>(xs), cuda::std::span<int4>(ys0), pi0);
  vdmpf.BatchEval(true, k1, cuda::std::span<const uint32_t>(xs), cuda::std::span<int4>(ys1), pi1);
  if (!VdmpfT::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1))) {
    fprintf(stderr, "vdmpf proof rejected on honest shares\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < xs.size(); ++i) {
    int4 actual = (BytesGroup::From(ys0[i]) + BytesGroup::From(ys1[i])).Into();
    int j;
    int4 expected = IsDmpfAlpha(xs[i], &j) ? kDmpfPoints.bs[j] : int4{0, 0, 0, 0};
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "vdmpf batch reconstruction mismatch at input %u\n", xs[i]);
      exit(EXIT_FAILURE);
    }
  }
}

static void BM_CpuVdmpfGen(benchmark::State &state) {
  AesCtx<2> ctx;
  fss::prp::Aes128Feistel prp;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdmpfT vdmpf{ctx.prg, xor_hash, hash, prp};
  typename VdmpfT::Key k0, k1;
  GenDmpfKeys(vdmpf, k0, k1);
  for (auto _ : state) {
    GenDmpfKeys(vdmpf, k0, k1);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

static void BM_CpuVdmpfEval(benchmark::State &state) {
  AesCtx<2> ctx;
  fss::prp::Aes128Feistel prp;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdmpfT vdmpf{ctx.prg, xor_hash, hash, prp};
  typename VdmpfT::Key k0, k1;
  GenDmpfKeys(vdmpf, k0, k1);
  CheckCpuVdmpfBatch(vdmpf, k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    uint32_t xs[1] = {x};
    int4 y;
    cuda::std::array<int4, 4> pi;
    vdmpf.BatchEval(false, k0, cuda::std::span<const uint32_t>(xs), cuda::std::span<int4>(&y, 1), pi);
    benchmark::DoNotOptimize(y);
    benchmark::DoNotOptimize(pi);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

static void BM_CpuVdmpfEvalAll(benchmark::State &state) {
  constexpr size_t n = size_t{1} << kInBits;
  AesCtx<2> ctx;
  fss::prp::Aes128Feistel prp;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdmpfT vdmpf{ctx.prg, xor_hash, hash, prp};
  typename VdmpfT::Key k0, k1;
  GenDmpfKeys(vdmpf, k0, k1);
  CheckCpuVdmpfBatch(vdmpf, k0, k1);
  std::vector<uint32_t> xs(n);
  for (size_t i = 0; i < n; ++i) {
    xs[i] = static_cast<uint32_t>(i);
  }
  std::vector<int4> ys(n);
  cuda::std::array<int4, 4> pi;
  for (auto _ : state) {
    vdmpf.BatchEval(false, k0, cuda::std::span<const uint32_t>(xs), cuda::std::span<int4>(ys), pi);
    benchmark::DoNotOptimize(ys.data());
    benchmark::DoNotOptimize(pi);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_CpuDmpfGen)->Name("fss/CPU/DMPF-bytes/Gen");
BENCHMARK(BM_CpuDmpfEval)->Name("fss/CPU/DMPF-bytes/Eval");
BENCHMARK(BM_CpuDmpfEvalAll)->Name("fss/CPU/DMPF-bytes/EvalAll");
BENCHMARK(BM_CpuVdmpfGen)->Name("fss/CPU/VDMPF-bytes/Gen");
BENCHMARK(BM_CpuVdmpfEval)->Name("fss/CPU/VDMPF-bytes/Eval");
BENCHMARK(BM_CpuVdmpfEvalAll)->Name("fss/CPU/VDMPF-bytes/EvalAll");

// ============================================================
// CPU VDPF benchmarks (eprint 2021/580 point baseline)
// ============================================================

using VdpfT = fss::Vdpf<kInBits, BytesGroup, CpuPrg<2>, fss::hash::Blake3, fss::hash::Blake3, uint32_t, 0>;

static int4 GroupFromBeta(bool selected) {
  return BytesGroup::From(selected ? kBeta : int4{0, 0, 0, 0}).Into();
}

static void CheckCpuVdpfScheme(const typename VdpfT::Key &k0, const typename VdpfT::Key &k1) {
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  const uint32_t points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  for (uint32_t x : points) {
    auto res0 = vdpf.Eval(false, k0, x);
    auto res1 = vdpf.Eval(true, k1, x);
    int4 actual = (BytesGroup::From(res0.y) + BytesGroup::From(res1.y)).Into();
    int4 expected = GroupFromBeta(x == kAlpha);
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "vdpf correctness mismatch at input %u\n", x);
      exit(1);
    }
    cuda::std::array<int4, 4> pi0, pi1;
    vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(&res0.pi_tilde, 1),
        cuda::std::span<const int4, 4>(k0.cs), pi0);
    vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(&res1.pi_tilde, 1),
        cuda::std::span<const int4, 4>(k0.cs), pi1);
    if (!VdpfT::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1))) {
      fprintf(stderr, "vdpf proof rejected on honest shares at input %u\n", x);
      exit(1);
    }
  }
}

static void CheckCpuVdpfFull(const typename VdpfT::Key &k0, const typename VdpfT::Key &k1) {
  constexpr size_t n = size_t{1} << kInBits;
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  std::vector<int4> first(n), second(n);
  cuda::std::array<int4, 4> pi0, pi1;
  vdpf.EvalAll(false, k0, cuda::std::span<int4>(first.data(), n), pi0);
  vdpf.EvalAll(true, k1, cuda::std::span<int4>(second.data(), n), pi1);
  if (!VdpfT::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1))) {
    fprintf(stderr, "vdpf full-domain proof rejected on honest shares\n");
    exit(1);
  }
  for (size_t x = 0; x < n; ++x) {
    int4 actual = (BytesGroup::From(first[x]) + BytesGroup::From(second[x])).Into();
    int4 expected = GroupFromBeta(x == kAlpha);
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "vdpf full reconstruction mismatch at input %zu\n", x);
      exit(1);
    }
  }
}

// EvalAllMerkle must reconstruct the same shares as EvalAll and both parties'
// Merkle-fold proofs must verify; the accumulator differs from the chain, so
// only Verify() is shared between the two rows.
static void CheckCpuVdpfFullMerkle(const typename VdpfT::Key &k0, const typename VdpfT::Key &k1) {
  constexpr size_t n = size_t{1} << kInBits;
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  std::vector<int4> first(n), second(n);
  cuda::std::array<int4, 4> pi0, pi1;
  vdpf.EvalAllMerkle(false, k0, cuda::std::span<int4>(first.data(), n), pi0);
  vdpf.EvalAllMerkle(true, k1, cuda::std::span<int4>(second.data(), n), pi1);
  if (!VdpfT::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1))) {
    fprintf(stderr, "vdpf merkle full-domain proof rejected on honest shares\n");
    exit(1);
  }
  for (size_t x = 0; x < n; ++x) {
    int4 actual = (BytesGroup::From(first[x]) + BytesGroup::From(second[x])).Into();
    int4 expected = GroupFromBeta(x == kAlpha);
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "vdpf merkle full reconstruction mismatch at input %zu\n", x);
      exit(1);
    }
  }
}

static void BM_CpuVdpfGen(benchmark::State &state) {
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename VdpfT::Key k0, k1;
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  if (vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), kAlpha, kBeta) != 0) {
    fprintf(stderr, "vdpf gen failed after retries\n");
    exit(1);
  }
  CheckCpuVdpfScheme(k0, k1);
  for (auto _ : state) {
    vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), kAlpha, kBeta);
    benchmark::DoNotOptimize(k0);
    benchmark::DoNotOptimize(k1);
  }
}

static void BM_CpuVdpfEval(benchmark::State &state) {
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename VdpfT::Key k0, k1;
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  if (vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), kAlpha, kBeta) != 0) {
    fprintf(stderr, "vdpf gen failed after retries\n");
    exit(1);
  }
  CheckCpuVdpfScheme(k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    auto res = vdpf.Eval(false, k0, x);
    cuda::std::array<int4, 4> pi;
    vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(&res.pi_tilde, 1),
        cuda::std::span<const int4, 4>(k0.cs), pi);
    benchmark::DoNotOptimize(res.y);
    benchmark::DoNotOptimize(pi);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

static void BM_CpuVdpfEvalAll(benchmark::State &state) {
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename VdpfT::Key k0, k1;
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  if (vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), kAlpha, kBeta) != 0) {
    fprintf(stderr, "vdpf gen failed after retries\n");
    exit(1);
  }
  CheckCpuVdpfScheme(k0, k1);
  CheckCpuVdpfFull(k0, k1);
  cuda::std::array<int4, 4> pi;
  for (auto _ : state) {
    vdpf.EvalAll(false, k0, cuda::std::span<int4>(ys.data(), n), pi);
    benchmark::DoNotOptimize(ys.data());
    benchmark::DoNotOptimize(pi);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

// Same shape as BM_CpuVdpfEvalAll but with the Merkle-fold accumulator; the
// proof is produced inside the timed loop and only compared outside it.
static void BM_CpuVdpfEvalAllMerkle(benchmark::State &state) {
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename VdpfT::Key k0, k1;
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<2> ctx;
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  VdpfT vdpf{ctx.prg, xor_hash, hash};
  if (vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), kAlpha, kBeta) != 0) {
    fprintf(stderr, "vdpf gen failed after retries\n");
    exit(1);
  }
  CheckCpuVdpfScheme(k0, k1);
  CheckCpuVdpfFull(k0, k1);
  CheckCpuVdpfFullMerkle(k0, k1);
  cuda::std::array<int4, 4> pi;
  for (auto _ : state) {
    vdpf.EvalAllMerkle(false, k0, cuda::std::span<int4>(ys.data(), n), pi);
    benchmark::DoNotOptimize(ys.data());
    benchmark::DoNotOptimize(pi);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_CpuVdpfGen)->Name("fss/CPU/VDPF-bytes/Gen");
BENCHMARK(BM_CpuVdpfEval)->Name("fss/CPU/VDPF-bytes/Eval");
BENCHMARK(BM_CpuVdpfEvalAll)->Name("fss/CPU/VDPF-bytes/EvalAll");
BENCHMARK(BM_CpuVdpfEvalAllMerkle)->Name("fss/CPU/VDPF-bytes/Merkle/EvalAll");

// ============================================================
// GPU DPF/DCF kernels (ChaCha PRG)
// ============================================================


template <int in_bits, typename Group>
__global__ void DpfGenKernel(typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

  typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Key k0, k1;
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s), alphas[tid], betas[tid]);
  for (int i = 0; i <= in_bits; ++i) cws[tid * (in_bits + 1) + i] = k0.cws[i];
}

template <int in_bits, typename Group>
__global__ void DpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

  typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i <= in_bits; ++i) key.cws[i] = cws[tid * (in_bits + 1) + i];
  ys[tid] = dpf.Eval(party, key, xs[tid]);
}

template <int in_bits, typename Group>
__global__ void DcfGenKernel(typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<4> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

  typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Key k0, k1;
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s), alphas[tid], betas[tid]);
  for (int i = 0; i <= in_bits; ++i) cws[tid * (in_bits + 1) + i] = k0.cws[i];
}

template <int in_bits, typename Group>
__global__ void DcfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<4> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

  typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i <= in_bits; ++i) key.cws[i] = cws[tid * (in_bits + 1) + i];
  ys[tid] = dcf.Eval(party, key, xs[tid]);
}

// ============================================================
// GPU benchmark helpers
// ============================================================

struct GpuData {
  int4 *d_seeds;
  int4 *d_seeds0;
  uint *d_alphas;
  int4 *d_betas;
  uint *d_xs;
  int4 *d_ys;

  GpuData() {
    auto *h_seeds = new int4[kN * 2];
    auto *h_seeds0 = new int4[kN];
    auto *h_alphas = new uint[kN];
    auto *h_betas = new int4[kN];
    auto *h_xs = new uint[kN];

    srand(42);
    for (int i = 0; i < kN; i++) {
      h_seeds[i * 2] = {rand(), rand(), rand(), rand() & ~1};
      h_seeds[i * 2 + 1] = {rand(), rand(), rand(), rand() & ~1};
      h_seeds0[i] = h_seeds[i * 2];
      h_alphas[i] = rand();
      h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
      h_xs[i] = rand();
    }

    auto alloc = [](auto **d, auto *h, int n) {
      CUDA_CHECK(cudaMalloc(d, sizeof(*h) * n));
      CUDA_CHECK(cudaMemcpy(*d, h, sizeof(*h) * n, cudaMemcpyHostToDevice));
    };
    alloc(&d_seeds, h_seeds, kN * 2);
    alloc(&d_seeds0, h_seeds0, kN);
    alloc(&d_alphas, h_alphas, kN);
    alloc(&d_betas, h_betas, kN);
    alloc(&d_xs, h_xs, kN);
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kN));

    delete[] h_seeds;
    delete[] h_seeds0;
    delete[] h_alphas;
    delete[] h_betas;
    delete[] h_xs;
  }

  ~GpuData() {
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_seeds0));
    CUDA_CHECK(cudaFree(d_alphas));
    CUDA_CHECK(cudaFree(d_betas));
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
  }
};


template <typename Group, bool comparison>
__global__ void CheckGpuPointKernel(int4 *outputs) {
  if (threadIdx.x != 0) return;
  const int4 seeds[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)}};
  const int4 beta = {7, 0, 0, 0};
  const uint points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  if constexpr (comparison) {
    fss::Dcf<kInBits, Group, fss::prg::ChaCha<4>, uint> scheme{fss::prg::ChaCha<4>(0x12345678, static_cast<int>(0x9abcdef0u))};
    typename decltype(scheme)::Key k0, k1;
    scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, beta);
    for (int i = 0; i < 5; ++i) {
      outputs[2 * i] = scheme.Eval(false, k0, points[i]);
      outputs[2 * i + 1] = scheme.Eval(true, k1, points[i]);
    }
  } else {
    fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint> scheme{fss::prg::ChaCha<2>(0x12345678, static_cast<int>(0x9abcdef0u))};
    typename decltype(scheme)::Key k0, k1;
    scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, beta);
    for (int i = 0; i < 5; ++i) {
      outputs[2 * i] = scheme.Eval(false, k0, points[i]);
      outputs[2 * i + 1] = scheme.Eval(true, k1, points[i]);
    }
  }
}

template <typename Group, bool comparison>
static void CheckGpuPoint() {
  static bool checked = false;
  if (checked) return;
  int4 outputs[10];
  int4 *device;
  CUDA_CHECK(cudaMalloc(&device, sizeof(outputs)));
  CheckGpuPointKernel<Group, comparison><<<1, 1>>>(device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(outputs, device, sizeof(outputs), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device));
  const uint points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  for (int i = 0; i < 5; ++i) {
    int4 actual = (Group::From(outputs[2 * i]) + Group::From(outputs[2 * i + 1])).Into();
    bool selected = comparison ? points[i] < kAlpha : points[i] == kAlpha;
    int4 expected = Group::From(selected ? kBeta : int4{0, 0, 0, 0}).Into();
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "GPU point reconstruction mismatch at input %u\n", points[i]);
      exit(EXIT_FAILURE);
    }
  }
  checked = true;
}

// ============================================================
// GPU DPF benchmarks
// ============================================================

template <typename Group>
static void BM_GpuDpfGen(benchmark::State &state) {
  CheckGpuPoint<Group, false>();
  using DpfT = fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint>;
  GpuData data;
  typename DpfT::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfT::Cw) * (kInBits + 1) * kN));

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    DpfGenKernel<kInBits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * kN);
  CUDA_CHECK(cudaFree(d_cws));
}

template <typename Group>
static void BM_GpuDpfEval(benchmark::State &state) {
  CheckGpuPoint<Group, false>();
  using DpfT = fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint>;
  GpuData data;
  typename DpfT::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfT::Cw) * (kInBits + 1) * kN));

  DpfGenKernel<kInBits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    DpfEvalKernel<kInBits, Group><<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * kN);
  CUDA_CHECK(cudaFree(d_cws));
}

// ============================================================
// GPU DCF benchmarks
// ============================================================

template <typename Group>
static void BM_GpuDcfGen(benchmark::State &state) {
  CheckGpuPoint<Group, true>();
  using DcfT = fss::Dcf<kInBits, Group, fss::prg::ChaCha<4>, uint>;
  GpuData data;
  typename DcfT::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfT::Cw) * (kInBits + 1) * kN));

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    DcfGenKernel<kInBits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * kN);
  CUDA_CHECK(cudaFree(d_cws));
}

template <typename Group>
static void BM_GpuDcfEval(benchmark::State &state) {
  CheckGpuPoint<Group, true>();
  using DcfT = fss::Dcf<kInBits, Group, fss::prg::ChaCha<4>, uint>;
  GpuData data;
  typename DcfT::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfT::Cw) * (kInBits + 1) * kN));

  DcfGenKernel<kInBits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    DcfEvalKernel<kInBits, Group><<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * kN);
  CUDA_CHECK(cudaFree(d_cws));
}


// Single-key full-domain kernels. Timing covers one party and one launch.
constexpr int Log2BlockSize() {
  int value = kThreadsPerBlock;
  int bits = 0;
  while (value > 1) { value >>= 1; ++bits; }
  return bits;
}
static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0);

static bool EqualOutput(int4 lhs, int4 rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

template <typename Group, bool half_tree>
struct FullScheme;
template <typename Group>
struct FullScheme<Group, false> {
  using Type = fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint>;
};
template <typename Group>
struct FullScheme<Group, true> {
  using Type = fss::HalfTreeDpf<kInBits, Group, fss::prg::ChaCha<1>, uint>;
};

template <typename Group, bool half_tree>
static void BM_GpuFullEvalAll(benchmark::State &state) {
  constexpr int z = kInBits < 17 + int(half_tree) ? kInBits - int(half_tree) : 17;
  constexpr int b1 = z - Log2BlockSize();
  if constexpr (b1 < 0) {
    state.SkipWithError("unsupported: block size exceeds the frontier");
    return;
  } else {
    using Prg = fss::prg::ChaCha<half_tree ? 1 : 2>;
    using Scheme = typename FullScheme<Group, half_tree>::Type;
    constexpr size_t n = size_t{1} << kInBits;
    int active_blocks = 0;
    if constexpr (half_tree) {
      auto *kernel = &fss::gpu::detail::HalfTreeDpfEvalAllKernel<kInBits, z, b1, Group, Prg, typename Scheme::Cw>;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, kThreadsPerBlock, size_t{0}));
    } else {
      auto *kernel = &fss::gpu::detail::DpfEvalAllKernel<kInBits, z, b1, Group, Prg, typename Scheme::Cw>;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, kThreadsPerBlock, size_t{0}));
    }
    if (active_blocks == 0) {
      state.SkipWithError("unsupported: insufficient kernel resources for the block size");
      return;
    }
    const int4 hash_key = {11, 22, 33, 44};
    auto make_scheme = [&](Prg prg) {
      if constexpr (half_tree) return Scheme{prg, hash_key};
      else return Scheme{prg};
    };
    Scheme scheme = make_scheme(Prg(0x12345678, static_cast<int>(0x9abcdef0u)));
    typename Scheme::Key k0, k1;
    scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(kSeeds), kAlpha, kBeta);
    typename Scheme::Cw *device_cws;
    int4 *device_outputs;
    CUDA_CHECK(cudaMalloc(&device_cws, sizeof(k0.cws)));
    CUDA_CHECK(cudaMemcpy(device_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&device_outputs, n * sizeof(int4)));
    auto launch = [&](bool party) {
      if constexpr (half_tree) {
        fss::gpu::HalfTreeDpfEvalAllGpu<z, b1, kThreadsPerBlock>(
            party, (party ? k1 : k0).s0, device_cws, k0.ocw, device_outputs, scheme);
      } else {
        fss::gpu::DpfEvalAllGpu<z, b1, kThreadsPerBlock>(
            party, (party ? k1 : k0).s0, device_cws, device_outputs, scheme);
      }
      CUDA_CHECK(cudaGetLastError());
    };
    // Validate every output share against CPU full evaluation, then reconstruct.
    std::vector<int4> actual(n), expected(n), first(n);
    for (int party = 0; party < 2; ++party) {
      launch(party);
      CUDA_CHECK(cudaMemcpy(actual.data(), device_outputs, n * sizeof(int4), cudaMemcpyDeviceToHost));
      scheme.EvalAll(party, party ? k1 : k0, cuda::std::span<int4>(expected.data(), n));
      for (size_t x = 0; x < n; ++x) {
        if (!EqualOutput(actual[x], expected[x])) {
          fprintf(stderr, "GPU full evaluation mismatch at party %d input %zu\n", party, x);
          exit(EXIT_FAILURE);
        }
        if (party == 0) first[x] = actual[x];
        else {
          int4 sum = (Group::From(first[x]) + Group::From(actual[x])).Into();
          int4 beta = half_tree ? fss::util::SetLsb(kBeta, false) : kBeta;
          int4 wanted = Group::From(x == kAlpha ? beta : int4{0, 0, 0, 0}).Into();
          if (!EqualOutput(sum, wanted)) {
            fprintf(stderr, "GPU full reconstruction mismatch at input %zu\n", x);
            exit(EXIT_FAILURE);
          }
        }
      }
    }
    fprintf(stderr, "GPU full evaluation checked all %zu outputs, both parties, T=%d\n", n, kThreadsPerBlock);
    if (getenv("FSS_BENCH_CHECK_ONLY")) exit(EXIT_SUCCESS);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (auto _ : state) {
      CUDA_CHECK(cudaEventRecord(start));
      launch(false);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      state.SetIterationTime(ms / 1000.0);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["keys_per_launch"] = 1;
    state.counters["threads_per_block"] = kThreadsPerBlock;
    state.counters["domain_outputs"] = n;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_outputs));
    CUDA_CHECK(cudaFree(device_cws));
  }
}

// ============================================================
// GPU full-domain EvalAll for DCF, packed HalfTree DPF, VDPF
// (Merkle-fold proof), and Grotto DCF. Same shape as above: one
// key, one party per timed launch, every output share validated
// against the CPU scheme for both parties before timing.
// ============================================================

template <typename Group>
static void BM_GpuDcfEvalAll(benchmark::State &state) {
  constexpr int z = kInBits < 17 ? kInBits : 17;
  constexpr int b1 = z - Log2BlockSize();
  if constexpr (b1 < 0) {
    state.SkipWithError("unsupported: block size exceeds the frontier");
    return;
  } else {
    using Prg = fss::prg::ChaCha<4>;
    using Scheme = fss::Dcf<kInBits, Group, Prg, uint>;
    constexpr size_t n = size_t{1} << kInBits;
    Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
    typename Scheme::Key k0, k1;
    scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(kSeeds), kAlpha, kBeta);
    typename Scheme::Cw *device_cws;
    int4 *device_outputs;
    CUDA_CHECK(cudaMalloc(&device_cws, sizeof(k0.cws)));
    CUDA_CHECK(cudaMemcpy(device_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&device_outputs, n * sizeof(int4)));
    auto launch = [&](bool party) {
      fss::gpu::DcfEvalAllGpu<z, b1, kThreadsPerBlock>(
          party, (party ? k1 : k0).s0, device_cws, device_outputs, scheme);
      CUDA_CHECK(cudaGetLastError());
    };
    std::vector<int4> actual(n), expected(n), first(n);
    for (int party = 0; party < 2; ++party) {
      launch(party);
      CUDA_CHECK(cudaMemcpy(actual.data(), device_outputs, n * sizeof(int4), cudaMemcpyDeviceToHost));
      scheme.EvalAll(party, party ? k1 : k0, cuda::std::span<int4>(expected.data(), n));
      for (size_t x = 0; x < n; ++x) {
        if (!EqualOutput(actual[x], expected[x])) {
          fprintf(stderr, "GPU DCF full evaluation mismatch at party %d input %zu\n", party, x);
          exit(EXIT_FAILURE);
        }
        if (party == 0) first[x] = actual[x];
        else {
          int4 sum = (Group::From(first[x]) + Group::From(actual[x])).Into();
          int4 wanted = Group::From(x < kAlpha ? kBeta : int4{0, 0, 0, 0}).Into();
          if (!EqualOutput(sum, wanted)) {
            fprintf(stderr, "GPU DCF full reconstruction mismatch at input %zu\n", x);
            exit(EXIT_FAILURE);
          }
        }
      }
    }
    fprintf(stderr, "GPU DCF full evaluation checked all %zu outputs, both parties, T=%d\n", n, kThreadsPerBlock);
    if (getenv("FSS_BENCH_CHECK_ONLY")) exit(EXIT_SUCCESS);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (auto _ : state) {
      CUDA_CHECK(cudaEventRecord(start));
      launch(false);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      state.SetIterationTime(ms / 1000.0);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["keys_per_launch"] = 1;
    state.counters["threads_per_block"] = kThreadsPerBlock;
    state.counters["domain_outputs"] = n;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_outputs));
    CUDA_CHECK(cudaFree(device_cws));
  }
}

static void BM_GpuPackedHalfTreeDpfEvalAll(benchmark::State &state) {
  using Prg = fss::prg::ChaCha<1>;
  using Scheme = fss::PackedHalfTreeDpf<kInBits, 1, Prg, uint>;
  constexpr int kDepth = Scheme::kDepth;
  constexpr int z = kDepth < 16 ? kDepth : 16;
  // A kDepth below the block-size log has no b1 >= 0 with 2^(z - b1) equal to
  // kThreadsPerBlock, so run the whole frontier breadth-parallel in one block
  // of 2^z threads instead.
  constexpr int b1 = z < Log2BlockSize() ? 0 : z - Log2BlockSize();
  constexpr int bs = 1 << (z - b1);
  if constexpr (b1 < 0) {
    state.SkipWithError("unsupported: block size exceeds the frontier");
    return;
  } else {
    constexpr size_t n = size_t{1} << kInBits;
    constexpr size_t blocks = size_t{1} << kDepth;
    const int4 hash_key = {11, 22, 33, 44};
    Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash_key};
    typename Scheme::Key k0, k1;
    scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(kSeeds), kAlpha, 1);
    typename Scheme::Cw *device_cws;
    int4 *device_outputs;
    CUDA_CHECK(cudaMalloc(&device_cws, sizeof(k0.cws)));
    CUDA_CHECK(cudaMemcpy(device_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&device_outputs, blocks * sizeof(int4)));
    auto launch = [&](bool party) {
      fss::gpu::PackedHalfTreeDpfEvalAllGpu<z, b1, bs>(
          party, (party ? k1 : k0).s0, device_cws, k0.fcw, device_outputs, scheme);
      CUDA_CHECK(cudaGetLastError());
    };
    std::vector<int4> actual(blocks), expected(blocks), first(blocks);
    for (int party = 0; party < 2; ++party) {
      launch(party);
      CUDA_CHECK(cudaMemcpy(actual.data(), device_outputs, blocks * sizeof(int4), cudaMemcpyDeviceToHost));
      scheme.EvalAll(party, party ? k1 : k0, cuda::std::span<int4>(expected.data(), blocks));
      for (size_t j = 0; j < blocks; ++j) {
        if (!EqualOutput(actual[j], expected[j])) {
          fprintf(stderr, "GPU packed full evaluation mismatch at party %d block %zu\n", party, j);
          exit(EXIT_FAILURE);
        }
        if (party == 0) first[j] = actual[j];
      }
      if (party == 1) {
        for (size_t x = 0; x < n; ++x) {
          int4 sum = fss::util::Xor(first[x / Scheme::kLanes], actual[x / Scheme::kLanes]);
          uint64_t lane = Scheme::Extract(sum, static_cast<uint>(x));
          if (lane != (x == kAlpha ? 1u : 0u)) {
            fprintf(stderr, "GPU packed full reconstruction mismatch at input %zu\n", x);
            exit(EXIT_FAILURE);
          }
        }
      }
    }
    fprintf(stderr, "GPU packed full evaluation checked all %zu blocks, both parties, T=%d\n", blocks, bs);
    if (getenv("FSS_BENCH_CHECK_ONLY")) exit(EXIT_SUCCESS);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (auto _ : state) {
      CUDA_CHECK(cudaEventRecord(start));
      launch(false);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      state.SetIterationTime(ms / 1000.0);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["keys_per_launch"] = 1;
    state.counters["threads_per_block"] = bs;
    state.counters["domain_outputs"] = n;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_outputs));
    CUDA_CHECK(cudaFree(device_cws));
  }
}

// VdpfEvalAllMerkleGpu keeps the frontier within 8 levels of the domain depth
// and its single-block root fold needs 2^z <= bs^2, so a block size below
// 2^((in_bits - 8) / 2) leaves no feasible tiling; build those configurations
// without this benchmark instead of failing the static asserts below.
#if FSS_BENCH_DOMAIN_BITS <= 8 || \
    (1 << (FSS_BENCH_DOMAIN_BITS - 8)) <= FSS_BENCH_THREADS_PER_BLOCK * FSS_BENCH_THREADS_PER_BLOCK
template <typename Group>
static void BM_GpuVdpfEvalAllMerkle(benchmark::State &state) {
  // VdpfMerkleRootsKernel folds the block roots inside one block, so the
  // frontier depth is capped at twice the block-size log.
  constexpr int z_cap = 2 * Log2BlockSize();
  constexpr int z = kInBits < z_cap ? kInBits : z_cap;
  constexpr int b1 = z - Log2BlockSize();
  static_assert(b1 >= 0, "domain below one block");
  using Prg = fss::prg::ChaCha<2>;
  using Blake3 = fss::hash::Blake3;
  using Scheme = fss::Vdpf<kInBits, Group, Prg, Blake3, Blake3, uint>;
  constexpr size_t n = size_t{1} << kInBits;
  Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash, hash};
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename Scheme::Key k0, k1;
  if (scheme.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), kAlpha, kBeta) != 0) {
    fprintf(stderr, "vdpf gen failed after retries\n");
    exit(EXIT_FAILURE);
  }
  typename Scheme::Cw *device_cws;
  int4 *device_outputs;
  cuda::std::array<int4, 4> *device_pi;
  CUDA_CHECK(cudaMalloc(&device_cws, sizeof(k0.cws)));
  CUDA_CHECK(cudaMemcpy(device_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&device_outputs, n * sizeof(int4)));
  CUDA_CHECK(cudaMalloc(&device_pi, sizeof(cuda::std::array<int4, 4>)));
  auto launch = [&](bool party) {
    fss::gpu::VdpfEvalAllMerkleGpu<z, b1, kThreadsPerBlock>(
        party, (party ? k1 : k0).s0, device_cws, k0.cs, k0.ocw, device_outputs, device_pi, scheme);
    CUDA_CHECK(cudaGetLastError());
  };
  std::vector<int4> actual(n), expected(n), first(n);
  cuda::std::array<int4, 4> pi_gpu[2], pi_host[2];
  for (int party = 0; party < 2; ++party) {
    launch(party);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), device_outputs, n * sizeof(int4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&pi_gpu[party], device_pi, sizeof(pi_gpu[party]), cudaMemcpyDeviceToHost));
    scheme.EvalAllMerkle(party, party ? k1 : k0, cuda::std::span<int4>(expected), pi_host[party]);
    for (size_t x = 0; x < n; ++x) {
      if (!EqualOutput(actual[x], expected[x])) {
        fprintf(stderr, "GPU VDPF full evaluation mismatch at party %d input %zu\n", party, x);
        exit(EXIT_FAILURE);
      }
      if (party == 0) first[x] = actual[x];
      else {
        int4 sum = (Group::From(first[x]) + Group::From(actual[x])).Into();
        int4 wanted = Group::From(x == kAlpha ? kBeta : int4{0, 0, 0, 0}).Into();
        if (!EqualOutput(sum, wanted)) {
          fprintf(stderr, "GPU VDPF full reconstruction mismatch at input %zu\n", x);
          exit(EXIT_FAILURE);
        }
      }
    }
    for (int w = 0; w < 4; ++w) {
      if (!EqualOutput(pi_gpu[party][w], pi_host[party][w])) {
        fprintf(stderr, "GPU VDPF Merkle proof mismatch at party %d word %d\n", party, w);
        exit(EXIT_FAILURE);
      }
    }
  }
  if (!Scheme::Verify(cuda::std::span<const int4, 4>(pi_gpu[0]), cuda::std::span<const int4, 4>(pi_gpu[1]))) {
    fprintf(stderr, "GPU VDPF Merkle proofs do not verify\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "GPU VDPF full evaluation checked all %zu outputs and proofs, both parties, T=%d\n", n,
      kThreadsPerBlock);
  if (getenv("FSS_BENCH_CHECK_ONLY")) exit(EXIT_SUCCESS);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start));
    launch(false);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.counters["keys_per_launch"] = 1;
  state.counters["threads_per_block"] = kThreadsPerBlock;
  state.counters["domain_outputs"] = n;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_outputs));
  CUDA_CHECK(cudaFree(device_pi));
  CUDA_CHECK(cudaFree(device_cws));
}
#endif

static void BM_GpuGrottoDcfEvalAll(benchmark::State &state) {
  constexpr int z = kInBits < 17 ? kInBits : 17;
  constexpr int b1 = z - Log2BlockSize();
  if constexpr (b1 < 0) {
    state.SkipWithError("unsupported: block size exceeds the frontier");
    return;
  } else {
    using Prg = fss::prg::ChaCha<2>;
    using Scheme = fss::GrottoDcf<kInBits, Prg, uint>;
    constexpr size_t n = size_t{1} << kInBits;
    Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
    typename Scheme::Key k0, k1;
    scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(kSeeds), kAlpha);
    typename Scheme::Cw *device_cws;
    bool *device_outputs;
    CUDA_CHECK(cudaMalloc(&device_cws, sizeof(k0.cws)));
    CUDA_CHECK(cudaMemcpy(device_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&device_outputs, n * sizeof(bool)));
    auto launch = [&](bool party) {
      fss::gpu::GrottoDcfEvalAllGpu<z, b1, kThreadsPerBlock>(party, (party ? k1 : k0).s0, device_cws, device_outputs,
          scheme);
      CUDA_CHECK(cudaGetLastError());
    };
    auto actual = std::make_unique<bool[]>(n);
    auto expected = std::make_unique<bool[]>(n);
    auto first = std::make_unique<bool[]>(n);
    for (int party = 0; party < 2; ++party) {
      launch(party);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(actual.get(), device_outputs, n * sizeof(bool), cudaMemcpyDeviceToHost));
      scheme.EvalAll(party, party ? k1 : k0, cuda::std::span<bool>(expected.get(), n));
      for (size_t x = 0; x < n; ++x) {
        if (actual[x] != expected[x]) {
          fprintf(stderr, "GPU Grotto DCF full evaluation mismatch at party %d input %zu\n", party, x);
          exit(EXIT_FAILURE);
        }
        if (party == 0) first[x] = actual[x];
        else if ((first[x] ^ actual[x]) != (x >= kAlpha)) {
          fprintf(stderr, "GPU Grotto DCF full reconstruction mismatch at input %zu\n", x);
          exit(EXIT_FAILURE);
        }
      }
    }
    fprintf(stderr, "GPU Grotto DCF full evaluation checked all %zu outputs, both parties, T=%d\n", n,
        kThreadsPerBlock);
    if (getenv("FSS_BENCH_CHECK_ONLY")) exit(EXIT_SUCCESS);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (auto _ : state) {
      CUDA_CHECK(cudaEventRecord(start));
      launch(false);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      state.SetIterationTime(ms / 1000.0);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["keys_per_launch"] = 1;
    state.counters["threads_per_block"] = kThreadsPerBlock;
    state.counters["domain_outputs"] = n;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_outputs));
    CUDA_CHECK(cudaFree(device_cws));
  }
}

// ============================================================
// GPU DMPF benchmarks (ChaChaFeistel PRP for the device Cuckoo Locate)
// ============================================================

// Generates the GPU DMPF keys once, uploads the active bucket keys, and repacks
// them into the level-major layout the kernels read. Both parties share the
// correction words, so one repack serves both.
struct GpuDmpfKeys {
  using Cw = typename GpuDmpfT::InnerDpf::Cw;

  GpuDmpfT dmpf;
  typename GpuDmpfT::Key k0, k1;
  int m_rt;
  int b_rt;
  int4 *d_s0s[2] = {nullptr, nullptr};
  Cw *d_cws = nullptr;
  int4 *d_cw_s = nullptr;
  int4 *d_out_cw = nullptr;
  fss::gpu::ExtBits<kDmpfBucketBits> *d_extra = nullptr;

  GpuDmpfKeys()
      : dmpf{fss::prg::ChaCha<2>(0x12345678, static_cast<int>(0x9abcdef0u)), fss::prp::ChaChaFeistel()} {
    GenDmpfKeys(dmpf, k0, k1);
    m_rt = k0.m_rt;
    b_rt = k0.b_size_rt;

    std::vector<int4> h_s0s[2];
    std::vector<Cw> h_cws(static_cast<size_t>(m_rt) * (kDmpfBucketBits + 1));
    for (int p = 0; p < 2; ++p) {
      h_s0s[p].resize(m_rt);
      for (int i = 0; i < m_rt; ++i) {
        h_s0s[p][i] = (p ? k1 : k0).bks[i].s0;
      }
    }
    for (int i = 0; i < m_rt; ++i) {
      for (int j = 0; j <= kDmpfBucketBits; ++j) {
        h_cws[static_cast<size_t>(i) * (kDmpfBucketBits + 1) + j] = k0.bks[i].cws[j];
      }
    }

    CUDA_CHECK(cudaMalloc(&d_s0s[0], sizeof(int4) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_s0s[1], sizeof(int4) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_cws, h_cws.size() * sizeof(Cw)));
    CUDA_CHECK(cudaMalloc(&d_cw_s, sizeof(int4) * static_cast<size_t>(kDmpfBucketBits) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_out_cw, sizeof(int4) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_extra, sizeof(*d_extra) * m_rt));
    CUDA_CHECK(cudaMemcpy(d_s0s[0], h_s0s[0].data(), sizeof(int4) * m_rt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s0s[1], h_s0s[1].data(), sizeof(int4) * m_rt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cws, h_cws.data(), h_cws.size() * sizeof(Cw), cudaMemcpyHostToDevice));

    fss::gpu::DpfRepackGpu<kDmpfBucketBits, BytesGroup, fss::prg::ChaCha<2>, uint32_t>(
        d_cws, m_rt, d_cw_s, d_extra, d_out_cw);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  ~GpuDmpfKeys() {
    CUDA_CHECK(cudaFree(d_s0s[0]));
    CUDA_CHECK(cudaFree(d_s0s[1]));
    CUDA_CHECK(cudaFree(d_cws));
    CUDA_CHECK(cudaFree(d_cw_s));
    CUDA_CHECK(cudaFree(d_out_cw));
    CUDA_CHECK(cudaFree(d_extra));
  }

  GpuDmpfKeys(const GpuDmpfKeys &) = delete;
  GpuDmpfKeys &operator=(const GpuDmpfKeys &) = delete;
};

static void BM_GpuDmpfEval(benchmark::State &state) {
  GpuDmpfKeys keys;
  GpuData data;
  int4 *d_ys1;
  CUDA_CHECK(cudaMalloc(&d_ys1, sizeof(int4) * kN));

  // GpuData samples raw 32-bit points while the DMPF domain is 2^kInBits, so
  // keep the batch inside the domain.
  std::vector<uint32_t> xs(kN);
  srand(43);
  for (int i = 0; i < kN; ++i) xs[i] = static_cast<uint32_t>(rand()) & ((1u << kInBits) - 1);
  uint32_t *d_xs;
  CUDA_CHECK(cudaMalloc(&d_xs, sizeof(uint32_t) * kN));
  CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), sizeof(uint32_t) * kN, cudaMemcpyHostToDevice));

  // Validate: both parties reconstruct the sparse multi-point function.
  for (int party = 0; party < 2; ++party) {
    fss::gpu::DmpfBatchEvalGpu(party != 0, d_xs, party ? d_ys1 : data.d_ys, kN, keys.k0.sigma, keys.m_rt, keys.b_rt,
        keys.d_s0s[party], keys.d_cw_s, keys.d_extra, keys.d_out_cw, keys.dmpf);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int4> ys0(kN), ys1(kN);
  CUDA_CHECK(cudaMemcpy(ys0.data(), data.d_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ys1.data(), d_ys1, sizeof(int4) * kN, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) {
    int4 actual = (BytesGroup::From(ys0[i]) + BytesGroup::From(ys1[i])).Into();
    int j;
    int4 expected = IsDmpfAlpha(xs[i], &j) ? kDmpfPoints.bs[j] : int4{0, 0, 0, 0};
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "gpu dmpf batch reconstruction mismatch at input %u\n", xs[i]);
      exit(EXIT_FAILURE);
    }
  }

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fss::gpu::DmpfBatchEvalGpu(false, d_xs, data.d_ys, kN, keys.k0.sigma, keys.m_rt, keys.b_rt, keys.d_s0s[0],
        keys.d_cw_s, keys.d_extra, keys.d_out_cw, keys.dmpf);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * kN);
  CUDA_CHECK(cudaFree(d_xs));
  CUDA_CHECK(cudaFree(d_ys1));
}

static void BM_GpuDmpfEvalAll(benchmark::State &state) {
  constexpr size_t n = size_t{1} << kInBits;
  GpuDmpfKeys keys;
  int4 *d_ys[2];
  int4 *d_bucket_ys;
  CUDA_CHECK(cudaMalloc(&d_ys[0], n * sizeof(int4)));
  CUDA_CHECK(cudaMalloc(&d_ys[1], n * sizeof(int4)));
  CUDA_CHECK(cudaMalloc(&d_bucket_ys, sizeof(int4) * static_cast<size_t>(keys.m_rt) * (size_t{1} << kDmpfBucketBits)));

  // Validate: both parties reconstruct the sparse multi-point function.
  for (int party = 0; party < 2; ++party) {
    fss::gpu::DmpfEvalAllGpu(party != 0, keys.k0.sigma, keys.m_rt, keys.b_rt, keys.d_s0s[party], keys.d_cws,
        d_bucket_ys, d_ys[party], keys.dmpf);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int4> ys0(n), ys1(n);
  CUDA_CHECK(cudaMemcpy(ys0.data(), d_ys[0], n * sizeof(int4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ys1.data(), d_ys[1], n * sizeof(int4), cudaMemcpyDeviceToHost));
  for (size_t x = 0; x < n; ++x) {
    int4 actual = (BytesGroup::From(ys0[x]) + BytesGroup::From(ys1[x])).Into();
    int j;
    int4 expected = IsDmpfAlpha(static_cast<uint32_t>(x), &j) ? kDmpfPoints.bs[j] : int4{0, 0, 0, 0};
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "gpu dmpf full reconstruction mismatch at input %zu\n", x);
      exit(EXIT_FAILURE);
    }
  }

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fss::gpu::DmpfEvalAllGpu(false, keys.k0.sigma, keys.m_rt, keys.b_rt, keys.d_s0s[0], keys.d_cws, d_bucket_ys,
        d_ys[0], keys.dmpf);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.counters["keys_per_launch"] = 1;
  // The DMPF GPU kernels launch fixed 256-thread blocks; the configured block
  // size is not a tiling parameter for them.
  state.counters["threads_per_block"] = 256;
  state.counters["domain_outputs"] = n;
  CUDA_CHECK(cudaFree(d_ys[0]));
  CUDA_CHECK(cudaFree(d_ys[1]));
  CUDA_CHECK(cudaFree(d_bucket_ys));
}

// ============================================================
// GPU VDMPF benchmarks (ChaChaFeistel PRP for the device Cuckoo Locate)
// ============================================================

// Generates the GPU VDMPF keys once and uploads the per-bucket inner VDPF key
// material. Both parties share cws, cs, and ocw, so the repack and the cs and
// ocw buffers serve both; only the seeds differ.
struct GpuVdmpfKeys {
  using Cw = typename GpuVdmpfT::InnerVdpf::Cw;

  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kDmpfHashIv, 2)};
  GpuVdmpfT vdmpf;
  typename GpuVdmpfT::Key k0, k1;
  int m_rt;
  int b_rt;
  int4 *d_s0s[2] = {nullptr, nullptr};
  Cw *d_cws = nullptr;
  int4 *d_cw_s = nullptr;
  int4 *d_ocws = nullptr;
  fss::gpu::ExtBits<kDmpfBucketBits> *d_extra = nullptr;
  cuda::std::array<int4, 4> *d_cs = nullptr;

  GpuVdmpfKeys()
      : vdmpf{fss::prg::ChaCha<2>(0x12345678, static_cast<int>(0x9abcdef0u)), xor_hash, hash,
            fss::prp::ChaChaFeistel()} {
    GenDmpfKeys(vdmpf, k0, k1);
    m_rt = k0.m_rt;
    b_rt = k0.b_size_rt;

    std::vector<int4> h_s0s[2];
    std::vector<Cw> h_cws(static_cast<size_t>(m_rt) * kDmpfBucketBits);
    std::vector<int4> h_ocws(m_rt);
    std::vector<cuda::std::array<int4, 4>> h_cs(GpuVdmpfT::m);
    for (int p = 0; p < 2; ++p) {
      h_s0s[p].resize(m_rt);
      for (int i = 0; i < m_rt; ++i) {
        h_s0s[p][i] = (p ? k1 : k0).bks[i].s0;
      }
    }
    for (int i = 0; i < m_rt; ++i) {
      h_ocws[i] = k0.bks[i].ocw;
      for (int j = 0; j < kDmpfBucketBits; ++j) {
        h_cws[static_cast<size_t>(i) * kDmpfBucketBits + j] = k0.bks[i].cws[j];
      }
    }
    // cs is shared between the parties; the fold covers all m buckets.
    for (int i = 0; i < GpuVdmpfT::m; ++i) h_cs[i] = k0.bks[i].cs;

    CUDA_CHECK(cudaMalloc(&d_s0s[0], sizeof(int4) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_s0s[1], sizeof(int4) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_cws, h_cws.size() * sizeof(Cw)));
    CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_cs, sizeof(*d_cs) * GpuVdmpfT::m));
    CUDA_CHECK(cudaMalloc(&d_cw_s, sizeof(int4) * static_cast<size_t>(kDmpfBucketBits) * m_rt));
    CUDA_CHECK(cudaMalloc(&d_extra, sizeof(*d_extra) * m_rt));
    CUDA_CHECK(cudaMemcpy(d_s0s[0], h_s0s[0].data(), sizeof(int4) * m_rt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s0s[1], h_s0s[1].data(), sizeof(int4) * m_rt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cws, h_cws.data(), h_cws.size() * sizeof(Cw), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ocws, h_ocws.data(), sizeof(int4) * m_rt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cs, h_cs.data(), sizeof(*d_cs) * GpuVdmpfT::m, cudaMemcpyHostToDevice));

    fss::gpu::VdpfRepackGpu<kDmpfBucketBits, BytesGroup, fss::prg::ChaCha<2>, fss::hash::Blake3, fss::hash::Blake3,
        uint32_t>(d_cws, m_rt, d_cw_s, d_extra);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  ~GpuVdmpfKeys() {
    CUDA_CHECK(cudaFree(d_s0s[0]));
    CUDA_CHECK(cudaFree(d_s0s[1]));
    CUDA_CHECK(cudaFree(d_cws));
    CUDA_CHECK(cudaFree(d_ocws));
    CUDA_CHECK(cudaFree(d_cs));
    CUDA_CHECK(cudaFree(d_cw_s));
    CUDA_CHECK(cudaFree(d_extra));
  }

  GpuVdmpfKeys(const GpuVdmpfKeys &) = delete;
  GpuVdmpfKeys &operator=(const GpuVdmpfKeys &) = delete;
};

// Validates both parties' shares and proofs on a batch, then times party 0.
// The wrapper's internal pipeline buffers warm up during validation.
static void RunGpuVdmpfBatch(benchmark::State &state, const uint32_t *h_xs, int n) {
  GpuVdmpfKeys keys;
  int4 *d_ys[2];
  cuda::std::array<int4, 4> *d_pi[2];
  uint32_t *d_xs;
  CUDA_CHECK(cudaMalloc(&d_ys[0], sizeof(int4) * n));
  CUDA_CHECK(cudaMalloc(&d_ys[1], sizeof(int4) * n));
  CUDA_CHECK(cudaMalloc(&d_pi[0], sizeof(*d_pi[0])));
  CUDA_CHECK(cudaMalloc(&d_pi[1], sizeof(*d_pi[1])));
  CUDA_CHECK(cudaMalloc(&d_xs, sizeof(uint32_t) * n));
  CUDA_CHECK(cudaMemcpy(d_xs, h_xs, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

  // Validate: both parties reconstruct the sparse multi-point function and
  // their honest proofs match.
  for (int party = 0; party < 2; ++party) {
    fss::gpu::VdmpfBatchEvalGpu(party != 0, d_xs, d_ys[party], d_pi[party], n, keys.k0.sigma, keys.m_rt, keys.b_rt,
        keys.d_s0s[party], keys.d_cw_s, keys.d_extra, keys.d_cs, keys.d_ocws, keys.vdmpf);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int4> ys0(n), ys1(n);
  cuda::std::array<int4, 4> pi0, pi1;
  CUDA_CHECK(cudaMemcpy(ys0.data(), d_ys[0], sizeof(int4) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ys1.data(), d_ys[1], sizeof(int4) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&pi0, d_pi[0], sizeof(pi0), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&pi1, d_pi[1], sizeof(pi1), cudaMemcpyDeviceToHost));
  if (!GpuVdmpfT::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1))) {
    fprintf(stderr, "gpu vdmpf proof rejected on honest shares\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < n; ++i) {
    int4 actual = (BytesGroup::From(ys0[i]) + BytesGroup::From(ys1[i])).Into();
    int j;
    int4 expected = IsDmpfAlpha(h_xs[i], &j) ? kDmpfPoints.bs[j] : int4{0, 0, 0, 0};
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      fprintf(stderr, "gpu vdmpf batch reconstruction mismatch at input %u\n", h_xs[i]);
      exit(EXIT_FAILURE);
    }
  }

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fss::gpu::VdmpfBatchEvalGpu(false, d_xs, d_ys[0], d_pi[0], n, keys.k0.sigma, keys.m_rt, keys.b_rt, keys.d_s0s[0],
        keys.d_cw_s, keys.d_extra, keys.d_cs, keys.d_ocws, keys.vdmpf);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * n);
  CUDA_CHECK(cudaFree(d_ys[0]));
  CUDA_CHECK(cudaFree(d_ys[1]));
  CUDA_CHECK(cudaFree(d_pi[0]));
  CUDA_CHECK(cudaFree(d_pi[1]));
  CUDA_CHECK(cudaFree(d_xs));
}

static void BM_GpuVdmpfEval(benchmark::State &state) {
  // GpuData samples raw 32-bit points while the VDMPF domain is 2^kInBits, so
  // keep the batch inside the domain.
  std::vector<uint32_t> xs(kN);
  srand(44);
  for (int i = 0; i < kN; ++i) xs[i] = static_cast<uint32_t>(rand()) & ((1u << kInBits) - 1);
  RunGpuVdmpfBatch(state, xs.data(), kN);
}

static void BM_GpuVdmpfEvalAll(benchmark::State &state) {
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<uint32_t> xs(n);
  for (size_t i = 0; i < n; ++i) xs[i] = static_cast<uint32_t>(i);
  RunGpuVdmpfBatch(state, xs.data(), static_cast<int>(n));
  state.counters["keys_per_launch"] = 1;
  // The VDMPF GPU kernels launch fixed 256-thread blocks (and the small fold
  // blocks); the configured block size is not a tiling parameter for them.
  state.counters["threads_per_block"] = 256;
  state.counters["domain_outputs"] = n;
}





// ============================================================
// AesSoft DPF Eval benchmarks (software AES PRG, mul=2)
// ============================================================

static const uint8_t kHostAesSoftKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

static void BM_CpuAesSoftEval(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint32_t, 0>;
  uint32_t te0[256];
  uint8_t sbox[256];
  fss::prg::aes_detail::InitTe0(te0);
  fss::prg::aes_detail::InitSbox(sbox);
  fss::prg::Aes128Soft<2> prg(kHostAesSoftKeys, te0, sbox);
  DpfT dpf{prg};
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), kAlpha, kBeta);
  CheckCpuScheme<BytesGroup, false>(dpf, k0, k1);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dpf.Eval(false, k0, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

BENCHMARK(BM_CpuAesSoftEval)->Name("fss/CPU/DPF-bytes/AesSoft/Eval");

// GPU AesSoft DPF kernels

__constant__ uint8_t kAesSoftKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

__global__ void AesSoftDpfGenKernel(typename fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>::Cw *cws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
  __shared__ uint32_t s_te0[256];
  __shared__ uint8_t s_sbox[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    s_te0[i] = fss::prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
    s_sbox[i] = fss::prg::aes_detail::Sbox(static_cast<uint8_t>(i));
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::Aes128Soft<2> prg(kAesSoftKeys, s_te0, s_sbox);
  fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint> dpf{prg};
  typename fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>::Key k0, k1;
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s), alphas[tid], betas[tid]);
  for (int i = 0; i <= kInBits; ++i) cws[tid * (kInBits + 1) + i] = k0.cws[i];
}

__global__ void AesSoftDpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>::Cw *cws, const uint *xs) {
  __shared__ uint32_t s_te0[256];
  __shared__ uint8_t s_sbox[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    s_te0[i] = fss::prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
    s_sbox[i] = fss::prg::aes_detail::Sbox(static_cast<uint8_t>(i));
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::Aes128Soft<2> prg(kAesSoftKeys, s_te0, s_sbox);
  fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint> dpf{prg};
  typename fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws[tid * (kInBits + 1) + i];
  ys[tid] = dpf.Eval(party, key, xs[tid]);
}

static void BM_GpuAesSoftEval(benchmark::State &state) {
  int gen_active_blocks = 0;
  int eval_active_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &gen_active_blocks, AesSoftDpfGenKernel, kThreadsPerBlock, size_t{0}));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &eval_active_blocks, AesSoftDpfEvalKernel, kThreadsPerBlock, size_t{0}));
  if (gen_active_blocks == 0 || eval_active_blocks == 0) {
    state.SkipWithError("unsupported: insufficient software AES kernel resources for the block size");
    return;
  }
  using DpfT = fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>;
  GpuData data;
  typename DpfT::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfT::Cw) * (kInBits + 1) * kN));

  AesSoftDpfGenKernel<<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    AesSoftDpfEvalKernel<<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  state.SetItemsProcessed(state.iterations() * kN);
  CUDA_CHECK(cudaFree(d_cws));
}

// ============================================================
// GPU benchmark registration
// ============================================================

static void RegisterGpuBenchmarks() {
  if (!HasGpu()) return;
  // DPF
  benchmark::RegisterBenchmark("fss/GPU/DPF-bytes/Gen", BM_GpuDpfGen<BytesGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DPF-uint/Gen", BM_GpuDpfGen<UintGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DPF-bytes/Eval", BM_GpuDpfEval<BytesGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DPF-uint/Eval", BM_GpuDpfEval<UintGroup>)->UseManualTime();
  // DCF
  benchmark::RegisterBenchmark("fss/GPU/DCF-bytes/Gen", BM_GpuDcfGen<BytesGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DCF-uint/Gen", BM_GpuDcfGen<UintGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DCF-bytes/Eval", BM_GpuDcfEval<BytesGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DCF-uint/Eval", BM_GpuDcfEval<UintGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DPF-bytes/EvalAll", BM_GpuFullEvalAll<BytesGroup, false>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/HalfTreeDPF-bytes/EvalAll", BM_GpuFullEvalAll<BytesGroup, true>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DPF-uint/EvalAll", BM_GpuFullEvalAll<UintGroup, false>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/HalfTreeDPF-uint/EvalAll", BM_GpuFullEvalAll<UintGroup, true>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DCF-bytes/EvalAll", BM_GpuDcfEvalAll<BytesGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DCF-uint/EvalAll", BM_GpuDcfEvalAll<UintGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/PackedHalfTreeDPF-bits1/EvalAll", BM_GpuPackedHalfTreeDpfEvalAll)->UseManualTime();
#if FSS_BENCH_DOMAIN_BITS <= 8 || \
    (1 << (FSS_BENCH_DOMAIN_BITS - 8)) <= FSS_BENCH_THREADS_PER_BLOCK * FSS_BENCH_THREADS_PER_BLOCK
  benchmark::RegisterBenchmark("fss/GPU/VDPF-bytes/EvalAll", BM_GpuVdpfEvalAllMerkle<BytesGroup>)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/VDPF-uint/EvalAll", BM_GpuVdpfEvalAllMerkle<UintGroup>)->UseManualTime();
#endif
  benchmark::RegisterBenchmark("fss/GPU/GrottoDCF/EvalAll", BM_GpuGrottoDcfEvalAll)->UseManualTime();
  // DMPF
  benchmark::RegisterBenchmark("fss/GPU/DMPF-bytes/Eval", BM_GpuDmpfEval)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/DMPF-bytes/EvalAll", BM_GpuDmpfEvalAll)->UseManualTime();
  // VDMPF
  benchmark::RegisterBenchmark("fss/GPU/VDMPF-bytes/Eval", BM_GpuVdmpfEval)->UseManualTime();
  benchmark::RegisterBenchmark("fss/GPU/VDMPF-bytes/EvalAll", BM_GpuVdmpfEvalAll)->UseManualTime();
  // AesSoft DPF Eval
  benchmark::RegisterBenchmark("fss/GPU/DPF-bytes/AesSoft/Eval", BM_GpuAesSoftEval)->UseManualTime();
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  // CUDA initialization during static construction can block sanitizer startup.
  RegisterGpuBenchmarks();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
