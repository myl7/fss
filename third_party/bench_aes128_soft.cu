// Benchmark: T-table AES (Aes128Soft) vs textbook AES (torchcsprng::Aes128Mmo)
// Both do AES-128 MMO: out = AES(key, seed) XOR seed
//
// GPU mode: 1M parallel operations with CUDA event timing.
// CPU mode (fallback when no GPU): single-threaded latency.

#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fss/prg/aes128_mmo_soft.cuh>
#include "torchcsprng/aes128_mmo_soft.cuh"

constexpr int kN = 1 << 20;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

static bool HasGpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// ============================================================
// GPU kernels
// ============================================================

__constant__ uint8_t kKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

template <int mul>
__global__ void AesSoftKernel(int4 *out, const int4 *seeds) {
    __shared__ uint32_t s_te0[256];
    __shared__ uint8_t s_sbox[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_te0[i] = fss::prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
        s_sbox[i] = fss::prg::aes_detail::Sbox(static_cast<uint8_t>(i));
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::Aes128Soft<mul> prg(kKeys, s_te0, s_sbox);
    auto result = prg.Gen(seeds[tid]);
    out[tid * mul] = result[0];
    if constexpr (mul >= 2) out[tid * mul + 1] = result[1];
}

template <int mul>
__global__ void TorchCsprngKernel(int4 *out, const int4 *seeds) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    torchcsprng::Aes128Mmo<mul> prg(kKeys);
    auto result = prg.Gen(seeds[tid]);
    out[tid * mul] = result[0];
    if constexpr (mul >= 2) out[tid * mul + 1] = result[1];
}

// ============================================================
// GPU benchmarks
// ============================================================

struct GpuSeeds {
    int4 *d_seeds;
    int4 *d_out;

    GpuSeeds(int mul) {
        auto *h_seeds = new int4[kN];
        srand(42);
        for (int i = 0; i < kN; i++) {
            h_seeds[i] = {rand(), rand(), rand(), rand()};
        }
        CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int4) * kN));
        CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, sizeof(int4) * kN, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int4) * kN * mul));
        delete[] h_seeds;
    }

    ~GpuSeeds() {
        cudaFree(d_seeds);
        cudaFree(d_out);
    }
};

template <int mul>
static void BM_AesSoft_GPU(benchmark::State &state) {
    GpuSeeds data(mul);
    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        AesSoftKernel<mul><<<kNumBlocks, kThreadsPerBlock>>>(data.d_out, data.d_seeds);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);
}

template <int mul>
static void BM_TorchCsprng_GPU(benchmark::State &state) {
    GpuSeeds data(mul);
    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        TorchCsprngKernel<mul><<<kNumBlocks, kThreadsPerBlock>>>(data.d_out, data.d_seeds);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);
}

// ============================================================
// CPU benchmarks
// ============================================================

static const uint8_t kHostKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

static void DoNotOptimize(int4 v) {
    asm volatile("" : : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

template <int mul>
static void BM_AesSoft_CPU(benchmark::State &state) {
    uint32_t te0[256];
    uint8_t sbox[256];
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    fss::prg::Aes128Soft<mul> prg(kHostKeys, te0, sbox);
    int4 seed = {0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10};
    for (auto _ : state) {
        auto result = prg.Gen(seed);
        DoNotOptimize(result[0]);
        if constexpr (mul >= 2) DoNotOptimize(result[1]);
        seed.x += result[0].x;
    }
    state.SetItemsProcessed(state.iterations());
}

template <int mul>
static void BM_TorchCsprng_CPU(benchmark::State &state) {
    torchcsprng::Aes128Mmo<mul> prg(kHostKeys);
    int4 seed = {0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10};
    for (auto _ : state) {
        auto result = prg.Gen(seed);
        DoNotOptimize(result[0]);
        if constexpr (mul >= 2) DoNotOptimize(result[1]);
        seed.x += result[0].x;
    }
    state.SetItemsProcessed(state.iterations());
}

// ============================================================
// Registration
// ============================================================

// CPU benchmarks (always available)
BENCHMARK(BM_AesSoft_CPU<1>)->Name("CPU/AesSoft/mul1");
BENCHMARK(BM_TorchCsprng_CPU<1>)->Name("CPU/TorchCsprng/mul1");
BENCHMARK(BM_AesSoft_CPU<2>)->Name("CPU/AesSoft/mul2");
BENCHMARK(BM_TorchCsprng_CPU<2>)->Name("CPU/TorchCsprng/mul2");

// GPU benchmarks (registered only if a GPU is present)
static void RegisterGpuBenchmarks() {
    if (!HasGpu()) return;
    benchmark::RegisterBenchmark("GPU/AesSoft/mul1", BM_AesSoft_GPU<1>)->UseManualTime();
    benchmark::RegisterBenchmark("GPU/TorchCsprng/mul1", BM_TorchCsprng_GPU<1>)->UseManualTime();
    benchmark::RegisterBenchmark("GPU/AesSoft/mul2", BM_AesSoft_GPU<2>)->UseManualTime();
    benchmark::RegisterBenchmark("GPU/TorchCsprng/mul2", BM_TorchCsprng_GPU<2>)->UseManualTime();
}

// Use a global constructor to register GPU benchmarks before main.
static int gpu_reg_ = (RegisterGpuBenchmarks(), 0);
