// Benchmark: torchcsprng textbook AES-128 MMO PRG (mul=2, matching DPF usage)
//
// GPU mode: 1M parallel operations with CUDA event timing.
// CPU mode: single-threaded latency.

#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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
// GPU kernel
// ============================================================

__constant__ uint8_t kKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

__global__ void AesSoftKernel(int4 *out, const int4 *seeds) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    torchcsprng::Aes128Mmo<2> prg(kKeys);
    auto result = prg.Gen(seeds[tid]);
    out[tid * 2] = result[0];
    out[tid * 2 + 1] = result[1];
}

// ============================================================
// GPU benchmark
// ============================================================

struct GpuSeeds {
    int4 *d_seeds;
    int4 *d_out;

    GpuSeeds() {
        auto *h_seeds = new int4[kN];
        srand(42);
        for (int i = 0; i < kN; i++) {
            h_seeds[i] = {rand(), rand(), rand(), rand()};
        }
        CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int4) * kN));
        CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, sizeof(int4) * kN, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int4) * kN * 2));
        delete[] h_seeds;
    }

    ~GpuSeeds() {
        cudaFree(d_seeds);
        cudaFree(d_out);
    }
};

static void BM_AesSoft_GPU(benchmark::State &state) {
    GpuSeeds data;
    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        AesSoftKernel<<<kNumBlocks, kThreadsPerBlock>>>(data.d_out, data.d_seeds);
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
// CPU benchmark
// ============================================================

static const uint8_t kHostKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

static void DoNotOptimize(int4 v) {
    asm volatile("" : : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static void BM_AesSoft_CPU(benchmark::State &state) {
    torchcsprng::Aes128Mmo<2> prg(kHostKeys);
    int4 seed = {0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10};
    for (auto _ : state) {
        auto result = prg.Gen(seed);
        DoNotOptimize(result[0]);
        DoNotOptimize(result[1]);
        seed.x += result[0].x;
    }
    state.SetItemsProcessed(state.iterations());
}

// ============================================================
// Registration
// ============================================================

BENCHMARK(BM_AesSoft_CPU)->Name("torchcsprng/CPU/AesSoft");

static void RegisterGpuBenchmarks() {
    if (!HasGpu()) return;
    benchmark::RegisterBenchmark("torchcsprng/GPU/AesSoft", BM_AesSoft_GPU)->UseManualTime();
}

static int gpu_reg_ = (RegisterGpuBenchmarks(), 0);
