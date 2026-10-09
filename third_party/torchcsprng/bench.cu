// Benchmark matching AES-128 MMO PRGs with mul=2.
// CPU timing includes Gen and the seed dependency, with key expansion outside
// the loop. GPU event timing includes per-thread key expansion, Gen, and output
// stores. The fss kernel also initializes its shared tables within that timing.

#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "torchcsprng/aes128_mmo_soft.cuh"
#include <fss/prg/aes128_mmo_soft.cuh>

constexpr int kN = 1 << 20;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;
constexpr int kCheckCount = 32;

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

__constant__ uint8_t kKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

static const uint8_t kHostKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};
static const int4 kCpuSeed = {0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10};

struct FssPrg {
  uint32_t te0[256];
  uint8_t sbox[256];
  fss::prg::Aes128Soft<2> prg;

  FssPrg() : prg(InitTables(), te0, sbox) {}

private:
  const uint8_t (*InitTables())[16] {
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    return kHostKeys;
  }
};

static bool Equal(int4 a, int4 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

static void CheckEqual(int4 actual, int4 expected, const char *backend, int seed, int output) {
  if (Equal(actual, expected)) return;
  fprintf(stderr, "aes correctness mismatch: %s seed %d output %d\n", backend, seed, output);
  exit(1);
}

static void CheckCpu() {
  static const bool checked = [] {
    torchcsprng::Aes128Mmo<2> textbook(kHostKeys);
    FssPrg optimized;
    // AES-128-ECB(seed) XOR seed, independently generated with OpenSSL.
    // int4 words encode the AES input and output in little-endian byte order.
    const int4 known[2] = {
        {static_cast<int>(0xdbd52fdeu), 0x65f1c2f7, static_cast<int>(0x86066080u), 0x0a1f8484},
        {0x19d0f52f, 0x565fc3c4, 0x641d963b, static_cast<int>(0xb3e9d38bu)},
    };
    const int4 seeds[] = {kCpuSeed, {0, 0, 0, 0}, {-1, -1, -1, -1}, {1, 2, 3, 4}};
    for (int i = 0; i < 4; ++i) {
      auto a = textbook.Gen(seeds[i]);
      auto b = optimized.prg.Gen(seeds[i]);
      for (int j = 0; j < 2; ++j) {
        CheckEqual(b[j], a[j], "fss CPU versus textbook CPU", i, j);
        if (i == 0) {
          CheckEqual(a[j], known[j], "textbook CPU versus known output", i, j);
          CheckEqual(b[j], known[j], "fss CPU versus known output", i, j);
        }
      }
    }
    return true;
  }();
  (void)checked;
}

template <bool optimized>
__global__ void AesSoftKernel(int4 *out, const int4 *seeds, int count) {
  __shared__ uint32_t te0[256];
  __shared__ uint8_t sbox[256];
  if constexpr (optimized) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      te0[i] = fss::prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
      sbox[i] = fss::prg::aes_detail::Sbox(static_cast<uint8_t>(i));
    }
    __syncthreads();
  }
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;

  if constexpr (optimized) {
    fss::prg::Aes128Soft<2> prg(kKeys, te0, sbox);
    auto result = prg.Gen(seeds[tid]);
    out[tid * 2] = result[0];
    out[tid * 2 + 1] = result[1];
  } else {
    torchcsprng::Aes128Mmo<2> prg(kKeys);
    auto result = prg.Gen(seeds[tid]);
    out[tid * 2] = result[0];
    out[tid * 2 + 1] = result[1];
  }
}

struct GpuSeeds {
  int4 *d_seeds;
  int4 *d_out;
  int4 check_seeds[kCheckCount];

  GpuSeeds() {
    auto *h_seeds = new int4[kN];
    srand(42);
    for (int i = 0; i < kN; ++i) {
      h_seeds[i] = {rand(), rand(), rand(), rand()};
      if (i < kCheckCount) check_seeds[i] = h_seeds[i];
    }
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int4) * kN));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, sizeof(int4) * kN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int4) * kN * 2));
    delete[] h_seeds;
  }

  ~GpuSeeds() {
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_out));
  }
};

static void CheckGpu(GpuSeeds &data) {
  CheckCpu();
  int4 textbook_out[kCheckCount * 2];
  int4 fss_out[kCheckCount * 2];
  AesSoftKernel<false><<<1, kThreadsPerBlock>>>(data.d_out, data.d_seeds, kCheckCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(textbook_out, data.d_out, sizeof(textbook_out), cudaMemcpyDeviceToHost));
  AesSoftKernel<true><<<1, kThreadsPerBlock>>>(data.d_out, data.d_seeds, kCheckCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(fss_out, data.d_out, sizeof(fss_out), cudaMemcpyDeviceToHost));
  torchcsprng::Aes128Mmo<2> host(kHostKeys);
  for (int i = 0; i < kCheckCount; ++i) {
    auto expected = host.Gen(data.check_seeds[i]);
    for (int j = 0; j < 2; ++j) {
      CheckEqual(textbook_out[i * 2 + j], expected[j], "textbook GPU versus CPU", i, j);
      CheckEqual(fss_out[i * 2 + j], expected[j], "fss GPU versus CPU", i, j);
    }
  }
}

template <bool optimized>
static void BM_AesSoft_GPU(benchmark::State &state) {
  GpuSeeds data;
  CheckGpu(data);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start));
    AesSoftKernel<optimized><<<kNumBlocks, kThreadsPerBlock>>>(data.d_out, data.d_seeds, kN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  state.SetItemsProcessed(state.iterations() * kN);
}

static void DoNotOptimize(int4 v) {
  asm volatile("" : : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

template <typename Prg>
static void RunCpu(benchmark::State &state, Prg &prg) {
  int4 seed = kCpuSeed;
  for (auto _ : state) {
    auto result = prg.Gen(seed);
    DoNotOptimize(result[0]);
    DoNotOptimize(result[1]);
    seed.x = static_cast<int>(static_cast<unsigned>(seed.x) + static_cast<unsigned>(result[0].x));
  }
  state.SetItemsProcessed(state.iterations());
}

static void BM_AesSoft_CPU(benchmark::State &state) {
  CheckCpu();
  torchcsprng::Aes128Mmo<2> prg(kHostKeys);
  RunCpu(state, prg);
}

static void BM_FssAesSoft_CPU(benchmark::State &state) {
  CheckCpu();
  FssPrg context;
  RunCpu(state, context.prg);
}

BENCHMARK(BM_AesSoft_CPU)->Name("torchcsprng/CPU/AesSoft");
BENCHMARK(BM_FssAesSoft_CPU)->Name("fss-prg/CPU/AesSoft");

static void RegisterGpuBenchmarks() {
  if (!HasGpu()) return;
  benchmark::RegisterBenchmark("torchcsprng/GPU/AesSoft", BM_AesSoft_GPU<false>)->UseManualTime();
  benchmark::RegisterBenchmark("fss-prg/GPU/AesSoft", BM_AesSoft_GPU<true>)->UseManualTime();
}

static int gpu_reg_ = (RegisterGpuBenchmarks(), 0);
