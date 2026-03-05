#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>

constexpr int kN = 1 << 20;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

using BytesGroup = fss::group::Bytes;
using UintGroup = fss::group::Uint<uint64_t>;

__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

#define CUDA_CHECK(x)                                                                            \
    do {                                                                                         \
        cudaError_t err = (x);                                                                   \
        if (err != cudaSuccess) {                                                                \
            fprintf(                                                                             \
                stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

// --- DPF Kernels ---

template <int in_bits, typename Group>
__global__ void DpfGenKernel(
    typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<2> prg(kNonce);
    fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dpf.Gen(cws + tid * (in_bits + 1), s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void DpfEvalKernel(
    int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws,
    const uint *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<2> prg(kNonce);
    fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

    ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (in_bits + 1), xs[tid]);
}

// --- DCF Kernels ---

template <int in_bits, typename Group>
__global__ void DcfGenKernel(
    typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<4> prg(kNonce);
    fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dcf.Gen(cws + tid * (in_bits + 1), s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void DcfEvalKernel(
    int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws,
    const uint *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<4> prg(kNonce);
    fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

    ys[tid] = dcf.Eval(party, seeds[tid], cws + tid * (in_bits + 1), xs[tid]);
}

// --- Benchmark helpers ---

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
        cudaFree(d_seeds);
        cudaFree(d_seeds0);
        cudaFree(d_alphas);
        cudaFree(d_betas);
        cudaFree(d_xs);
        cudaFree(d_ys);
    }
};

// --- DPF Gen GPU benchmark ---

template <int in_bits, typename Group>
static void BM_DpfGen(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>;
    GpuData data;

    typename DpfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kN));

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            d_cws, data.d_seeds, data.d_alphas, data.d_betas);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_cws);
}

// --- DPF Eval GPU benchmark ---

template <int in_bits, typename Group>
static void BM_DpfEval(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>;
    GpuData data;

    typename DpfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kN));

    // Pre-generate keys
    DpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
        d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DpfEvalKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_cws);
}

// --- DCF Gen GPU benchmark ---

template <int in_bits, typename Group>
static void BM_DcfGen(benchmark::State &state) {
    using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
    GpuData data;

    typename DcfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kN));

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DcfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            d_cws, data.d_seeds, data.d_alphas, data.d_betas);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_cws);
}

// --- DCF Eval GPU benchmark ---

template <int in_bits, typename Group>
static void BM_DcfEval(benchmark::State &state) {
    using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
    GpuData data;

    typename DcfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kN));

    // Pre-generate keys
    DcfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
        d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DcfEvalKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_cws);
}

// Register all benchmarks
#define REGISTER_BENCHES(in_bits)                                                                    \
    BENCHMARK(BM_DpfGen<in_bits, BytesGroup>)->Name("BM_DpfGen_Bytes/" #in_bits)->UseManualTime();   \
    BENCHMARK(BM_DpfGen<in_bits, UintGroup>)->Name("BM_DpfGen_Uint/" #in_bits)->UseManualTime();     \
    BENCHMARK(BM_DpfEval<in_bits, BytesGroup>)->Name("BM_DpfEval_Bytes/" #in_bits)->UseManualTime(); \
    BENCHMARK(BM_DpfEval<in_bits, UintGroup>)->Name("BM_DpfEval_Uint/" #in_bits)->UseManualTime();   \
    BENCHMARK(BM_DcfGen<in_bits, BytesGroup>)->Name("BM_DcfGen_Bytes/" #in_bits)->UseManualTime();   \
    BENCHMARK(BM_DcfGen<in_bits, UintGroup>)->Name("BM_DcfGen_Uint/" #in_bits)->UseManualTime();     \
    BENCHMARK(BM_DcfEval<in_bits, BytesGroup>)->Name("BM_DcfEval_Bytes/" #in_bits)->UseManualTime(); \
    BENCHMARK(BM_DcfEval<in_bits, UintGroup>)->Name("BM_DcfEval_Uint/" #in_bits)->UseManualTime();

REGISTER_BENCHES(14)
REGISTER_BENCHES(17)
REGISTER_BENCHES(20)
