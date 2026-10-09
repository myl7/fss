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
static void CheckCpuScheme(Scheme &scheme, const int4 seeds[2], const typename Scheme::Cw *cws) {
  const uint32_t points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1, (1u << kInBits) - 1};
  for (uint32_t x : points) {
    auto y0 = Group::From(scheme.Eval(false, seeds[0], cws, x));
    auto y1 = Group::From(scheme.Eval(true, seeds[1], cws, x));
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
static void CheckCpuFull(Scheme &scheme, const typename Scheme::Cw *cws) {
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> first(n), second(n);
  scheme.EvalAll(false, kSeeds[0], cws, first.data());
  scheme.EvalAll(true, kSeeds[1], cws, second.data());
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
  typename DpfT::Cw cws[kInBits + 1];
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, seeds, cws);
  for (auto _ : state) {
    dpf.Gen(cws, seeds, kAlpha, kBeta);
    benchmark::DoNotOptimize(cws);
  }
}

template <typename Group>
static void BM_CpuDpfEval(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Cw cws[kInBits + 1];
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, seeds, cws);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dpf.Eval(false, seeds[0], cws, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuDpfEvalAll(benchmark::State &state) {
  using DpfT = fss::Dpf<kInBits, Group, CpuPrg<2>, uint32_t, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Cw cws[kInBits + 1];
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<2> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<Group, false>(dpf, seeds, cws);
  CheckCpuFull<Group, false>(dpf, cws);
  for (auto _ : state) {
    dpf.EvalAll(false, seeds[0], cws, ys.data());
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
  typename DcfT::Cw cws[kInBits + 1];
  AesCtx<4> ctx;
  DcfT dcf{ctx.prg};
  dcf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<Group, true>(dcf, seeds, cws);
  for (auto _ : state) {
    dcf.Gen(cws, seeds, kAlpha, kBeta);
    benchmark::DoNotOptimize(cws);
  }
}

template <typename Group>
static void BM_CpuDcfEval(benchmark::State &state) {
  using DcfT = fss::Dcf<kInBits, Group, CpuPrg<4>, uint32_t, fss::DcfPred::kLt, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DcfT::Cw cws[kInBits + 1];
  AesCtx<4> ctx;
  DcfT dcf{ctx.prg};
  dcf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<Group, true>(dcf, seeds, cws);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dcf.Eval(false, seeds[0], cws, x);
    benchmark::DoNotOptimize(y);
    x = (x + 1) & ((1u << kInBits) - 1);
  }
}

template <typename Group>
static void BM_CpuDcfEvalAll(benchmark::State &state) {
  using DpfT = fss::Dcf<kInBits, Group, CpuPrg<4>, uint32_t, fss::DcfPred::kLt, 0>;
  int4 seeds[2] = {kSeeds[0], kSeeds[1]};
  typename DpfT::Cw cws[kInBits + 1];
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<int4> ys(n);
  AesCtx<4> ctx;
  DpfT dpf{ctx.prg};
  dpf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<Group, true>(dpf, seeds, cws);
  CheckCpuFull<Group, true>(dpf, cws);
  for (auto _ : state) {
    dpf.EvalAll(false, seeds[0], cws, ys.data());
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
// GPU DPF/DCF kernels (ChaCha PRG)
// ============================================================

__constant__ int kNonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

template <int in_bits, typename Group>
__global__ void DpfGenKernel(typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<2> prg(kNonce);
  fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(cws + tid * (in_bits + 1), s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void DpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<2> prg(kNonce);
  fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

  ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (in_bits + 1), xs[tid]);
}

template <int in_bits, typename Group>
__global__ void DcfGenKernel(typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<4> prg(kNonce);
  fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dcf.Gen(cws + tid * (in_bits + 1), s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void DcfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  fss::prg::ChaCha<4> prg(kNonce);
  fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

  ys[tid] = dcf.Eval(party, seeds[tid], cws + tid * (in_bits + 1), xs[tid]);
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
    fss::Dcf<kInBits, Group, fss::prg::ChaCha<4>, uint> scheme{fss::prg::ChaCha<4>(kNonce)};
    typename decltype(scheme)::Cw cws[kInBits + 1];
    scheme.Gen(cws, seeds, kAlpha, beta);
    for (int i = 0; i < 5; ++i) {
      outputs[2 * i] = scheme.Eval(false, seeds[0], cws, points[i]);
      outputs[2 * i + 1] = scheme.Eval(true, seeds[1], cws, points[i]);
    }
  } else {
    fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint> scheme{fss::prg::ChaCha<2>(kNonce)};
    typename decltype(scheme)::Cw cws[kInBits + 1];
    scheme.Gen(cws, seeds, kAlpha, beta);
    for (int i = 0; i < 5; ++i) {
      outputs[2 * i] = scheme.Eval(false, seeds[0], cws, points[i]);
      outputs[2 * i + 1] = scheme.Eval(true, seeds[1], cws, points[i]);
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
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
          fss::gpu::detail::HalfTreeDpfEvalAllKernel<kInBits, z, b1, Group, Prg>, kThreadsPerBlock, 0));
    } else {
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
          fss::gpu::detail::DpfEvalAllKernel<kInBits, z, b1, Group, Prg>, kThreadsPerBlock, 0));
    }
    if (active_blocks == 0) {
      state.SkipWithError("unsupported: insufficient kernel resources for the block size");
      return;
    }
    const int host_nonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};
    int *device_nonce;
    CUDA_CHECK(cudaMalloc(&device_nonce, sizeof(host_nonce)));
    CUDA_CHECK(cudaMemcpy(device_nonce, host_nonce, sizeof(host_nonce), cudaMemcpyHostToDevice));
    Prg host_prg(host_nonce);
    Prg device_prg(device_nonce);
    const int4 hash_key = {11, 22, 33, 44};
    auto make_scheme = [&](Prg prg) {
      if constexpr (half_tree) return Scheme{prg, hash_key};
      else return Scheme{prg};
    };
    Scheme host_scheme = make_scheme(host_prg);
    Scheme device_scheme = make_scheme(device_prg);
    typename Scheme::Cw cws[kInBits + 1];
    int4 ocw = {0, 0, 0, 0};
    if constexpr (half_tree) host_scheme.Gen(cws, ocw, kSeeds, kAlpha, kBeta);
    else host_scheme.Gen(cws, kSeeds, kAlpha, kBeta);
    typename Scheme::Cw *device_cws;
    int4 *device_outputs;
    CUDA_CHECK(cudaMalloc(&device_cws, sizeof(cws)));
    CUDA_CHECK(cudaMemcpy(device_cws, cws, sizeof(cws), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&device_outputs, n * sizeof(int4)));
    auto launch = [&](bool party) {
      if constexpr (half_tree) {
        fss::gpu::HalfTreeDpfEvalAllGpu<z, b1, kThreadsPerBlock>(
            party, kSeeds[party], device_cws, ocw, device_outputs, device_scheme);
      } else {
        fss::gpu::DpfEvalAllGpu<z, b1, kThreadsPerBlock>(
            party, kSeeds[party], device_cws, device_outputs, device_scheme);
      }
      CUDA_CHECK(cudaGetLastError());
    };
    // Validate every output share against CPU full evaluation, then reconstruct.
    std::vector<int4> actual(n), expected(n), first(n);
    for (int party = 0; party < 2; ++party) {
      launch(party);
      CUDA_CHECK(cudaMemcpy(actual.data(), device_outputs, n * sizeof(int4), cudaMemcpyDeviceToHost));
      if constexpr (half_tree) host_scheme.EvalAll(party, kSeeds[party], cws, ocw, expected.data());
      else host_scheme.EvalAll(party, kSeeds[party], cws, expected.data());
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
    CUDA_CHECK(cudaFree(device_nonce));
  }
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
  typename DpfT::Cw cws[kInBits + 1];
  dpf.Gen(cws, seeds, kAlpha, kBeta);
  CheckCpuScheme<BytesGroup, false>(dpf, seeds, cws);
  uint32_t x = 0;
  for (auto _ : state) {
    int4 y = dpf.Eval(false, seeds[0], cws, x);
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
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(cws + tid * (kInBits + 1), s, alphas[tid], betas[tid]);
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
  ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (kInBits + 1), xs[tid]);
}

static void BM_GpuAesSoftEval(benchmark::State &state) {
  int gen_active_blocks = 0;
  int eval_active_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &gen_active_blocks, AesSoftDpfGenKernel, kThreadsPerBlock, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &eval_active_blocks, AesSoftDpfEvalKernel, kThreadsPerBlock, 0));
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
