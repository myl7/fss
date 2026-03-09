#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_soft.cuh>
#include <fss/hash/blake3.cuh>

constexpr int kN = 1 << 20;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

using BytesGroup = fss::group::Bytes;
using UintGroup = fss::group::Uint<uint64_t>;

__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

__constant__ int4 kBlake3Iv[2] = {
    {0x11223344, 0x55667788, static_cast<int>(0x99aabbccu), static_cast<int>(0xddeeff00u)},
    {0x00112233, 0x44556677, static_cast<int>(0x8899aabbu), static_cast<int>(0xccddeeffu)},
};

__constant__ int4 kHalfTreeHashKey = {
    0x12345678, static_cast<int>(0x9abcdef0u), 0x0fedcba9, 0x87654321};

__constant__ uint8_t kAesSoftKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// --- DPF Kernels (ChaCha<2>) ---

template <int in_bits, typename Group>
__global__ void DpfGenKernel(typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
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

// --- DPF Kernels (Aes128Soft<2>) ---

template <int in_bits, typename Group>
__global__ void DpfGenKernelAes(
    typename fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::Aes128Soft<2> prg(kAesSoftKeys);
    fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint> dpf{prg};

    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dpf.Gen(cws + tid * (in_bits + 1), s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void DpfEvalKernelAes(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>::Cw *cws,
    const uint *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::Aes128Soft<2> prg(kAesSoftKeys);
    fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint> dpf{prg};

    ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (in_bits + 1), xs[tid]);
}

// --- DCF Kernels (ChaCha<4>) ---

template <int in_bits, typename Group>
__global__ void DcfGenKernel(typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
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

// --- VDPF Kernels (ChaCha<2> + Blake3) ---

template <int in_bits, typename Group>
using VdpfType =
    fss::Vdpf<in_bits, Group, fss::prg::ChaCha<2>, fss::hash::Blake3, fss::hash::Blake3, uint>;

template <int in_bits, typename Group>
__global__ void VdpfGenKernel(typename VdpfType<in_bits, Group>::Cw *cws,
    cuda::std::array<int4, 4> *cs, int4 *ocws, const int4 *seeds, const uint *alphas,
    const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<2> prg(kNonce);
    fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
    fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
    VdpfType<in_bits, Group> vdpf{prg, xor_hash, hash};

    int4 s0s_arr[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    cuda::std::span<const int4, 2> s0s(s0s_arr);
    vdpf.Gen(cws + tid * in_bits, cs[tid], ocws[tid], s0s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void VdpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename VdpfType<in_bits, Group>::Cw *cws, const cuda::std::array<int4, 4> *cs,
    const int4 *ocws, const uint *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<2> prg(kNonce);
    fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
    fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
    VdpfType<in_bits, Group> vdpf{prg, xor_hash, hash};

    int4 y;
    cuda::std::span<const typename VdpfType<in_bits, Group>::Cw> cws_span(
        cws + tid * in_bits, in_bits);
    cuda::std::span<const int4, 4> cs_span(cs[tid].data(), 4);
    vdpf.Eval(party, seeds[tid], cws_span, cs_span, ocws[tid], xs[tid], y);
    ys[tid] = y;
}

// --- HalfTreeDpf Kernels (ChaCha<1>) ---

template <int in_bits, typename Group>
using HtDpfType = fss::HalfTreeDpf<in_bits, Group, fss::prg::ChaCha<1>, uint>;

template <int in_bits, typename Group>
__global__ void HalfTreeDpfGenKernel(typename HtDpfType<in_bits, Group>::Cw *cws, int4 *ocws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<1> prg(kNonce);
    HtDpfType<in_bits, Group> dpf{prg, kHalfTreeHashKey};

    int4 s0s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dpf.Gen(cws + tid * in_bits, ocws[tid], s0s, alphas[tid], betas[tid]);
}

template <int in_bits, typename Group>
__global__ void HalfTreeDpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename HtDpfType<in_bits, Group>::Cw *cws, const int4 *ocws, const uint *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    fss::prg::ChaCha<1> prg(kNonce);
    HtDpfType<in_bits, Group> dpf{prg, kHalfTreeHashKey};

    ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * in_bits, ocws[tid], xs[tid]);
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

// --- DPF Eval GPU benchmark (ChaCha<2>) ---

template <int in_bits, typename Group>
static void BM_DpfEval(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>;
    GpuData data;

    typename DpfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kN));

    DpfGenKernel<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DpfEvalKernel<in_bits, Group>
            <<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);

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

// --- DPF Gen GPU benchmark (ChaCha<2>) ---

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

        DpfGenKernel<in_bits, Group>
            <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);

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

// --- DPF Eval GPU benchmark (Aes128Soft<2>) ---

template <int in_bits, typename Group>
static void BM_DpfEvalAes(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>;
    GpuData data;

    typename DpfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kN));

    DpfGenKernelAes<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DpfEvalKernelAes<in_bits, Group>
            <<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);

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

// --- DCF Eval GPU benchmark (ChaCha<4>) ---

template <int in_bits, typename Group>
static void BM_DcfEval(benchmark::State &state) {
    using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
    GpuData data;

    typename DcfType::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kN));

    DcfGenKernel<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DcfEvalKernel<in_bits, Group>
            <<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);

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

// --- DCF Gen GPU benchmark (ChaCha<4>) ---

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

        DcfGenKernel<in_bits, Group>
            <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);

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

// --- VDPF Eval GPU benchmark (ChaCha<2> + Blake3) ---

template <int in_bits, typename Group>
static void BM_VdpfEval(benchmark::State &state) {
    using Vdpf = VdpfType<in_bits, Group>;
    GpuData data;

    typename Vdpf::Cw *d_cws;
    cuda::std::array<int4, 4> *d_cs;
    int4 *d_ocws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename Vdpf::Cw) * in_bits * kN));
    CUDA_CHECK(cudaMalloc(&d_cs, sizeof(cuda::std::array<int4, 4>) * kN));
    CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kN));

    VdpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
        d_cws, d_cs, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        VdpfEvalKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            data.d_ys, false, data.d_seeds0, d_cws, d_cs, d_ocws, data.d_xs);

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
    cudaFree(d_cs);
    cudaFree(d_ocws);
}

// --- VDPF Gen GPU benchmark (ChaCha<2> + Blake3) ---

template <int in_bits, typename Group>
static void BM_VdpfGen(benchmark::State &state) {
    using Vdpf = VdpfType<in_bits, Group>;
    GpuData data;

    typename Vdpf::Cw *d_cws;
    cuda::std::array<int4, 4> *d_cs;
    int4 *d_ocws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename Vdpf::Cw) * in_bits * kN));
    CUDA_CHECK(cudaMalloc(&d_cs, sizeof(cuda::std::array<int4, 4>) * kN));
    CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kN));

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        VdpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            d_cws, d_cs, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);

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
    cudaFree(d_cs);
    cudaFree(d_ocws);
}

// --- HalfTreeDpf Eval GPU benchmark (ChaCha<1>) ---

template <int in_bits, typename Group>
static void BM_HalfTreeDpfEval(benchmark::State &state) {
    using HtDpf = HtDpfType<in_bits, Group>;
    GpuData data;

    typename HtDpf::Cw *d_cws;
    int4 *d_ocws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename HtDpf::Cw) * in_bits * kN));
    CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kN));

    HalfTreeDpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
        d_cws, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        HalfTreeDpfEvalKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            data.d_ys, false, data.d_seeds0, d_cws, d_ocws, data.d_xs);

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
    cudaFree(d_ocws);
}

// --- HalfTreeDpf Gen GPU benchmark (ChaCha<1>) ---

template <int in_bits, typename Group>
static void BM_HalfTreeDpfGen(benchmark::State &state) {
    using HtDpf = HtDpfType<in_bits, Group>;
    GpuData data;

    typename HtDpf::Cw *d_cws;
    int4 *d_ocws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename HtDpf::Cw) * in_bits * kN));
    CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kN));

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        HalfTreeDpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(
            d_cws, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);

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
    cudaFree(d_ocws);
}

// --- Register all 12 benchmarks ---

// DPF (ChaCha<2>)
BENCHMARK(BM_DpfEval<20, UintGroup>)->Name("BM_DpfEval_Uint/20")->UseManualTime();
BENCHMARK(BM_DpfEval<14, UintGroup>)->Name("BM_DpfEval_Uint/14")->UseManualTime();
BENCHMARK(BM_DpfEval<17, UintGroup>)->Name("BM_DpfEval_Uint/17")->UseManualTime();
BENCHMARK(BM_DpfGen<20, UintGroup>)->Name("BM_DpfGen_Uint/20")->UseManualTime();
BENCHMARK(BM_DpfEval<20, BytesGroup>)->Name("BM_DpfEval_Bytes/20")->UseManualTime();

// DPF other PRG
BENCHMARK(BM_DpfEvalAes<20, UintGroup>)->Name("BM_DpfEval_Uint_AesSoft/20")->UseManualTime();

// DCF (ChaCha<4>)
BENCHMARK(BM_DcfEval<20, UintGroup>)->Name("BM_DcfEval_Uint/20")->UseManualTime();
BENCHMARK(BM_DcfGen<20, UintGroup>)->Name("BM_DcfGen_Uint/20")->UseManualTime();

// VDPF (ChaCha<2> + Blake3)
BENCHMARK(BM_VdpfEval<20, UintGroup>)->Name("BM_VdpfEval_Uint/20")->UseManualTime();
BENCHMARK(BM_VdpfGen<20, UintGroup>)->Name("BM_VdpfGen_Uint/20")->UseManualTime();

// HalfTreeDpf (ChaCha<1>)
BENCHMARK(BM_HalfTreeDpfEval<20, UintGroup>)->Name("BM_HalfTreeDpfEval_Uint/20")->UseManualTime();
BENCHMARK(BM_HalfTreeDpfGen<20, UintGroup>)->Name("BM_HalfTreeDpfGen_Uint/20")->UseManualTime();
