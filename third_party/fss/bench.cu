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

constexpr int kInBits = 20;
constexpr int kN = 1 << 20;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

using BytesGroup = fss::group::Bytes;

// Wrap Uint<__uint128_t, 2^127> in a plain struct to avoid embedding a 128-bit
// integer literal in __global__ function stubs (nvcc stub gen bug).
struct UintGroup {
    using Impl = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
    Impl impl;

    __host__ __device__ UintGroup operator+(UintGroup rhs) const { return {impl + rhs.impl}; }
    __host__ __device__ UintGroup operator-() const { return {-impl}; }
    __host__ __device__ static UintGroup From(int4 buf) {
        UintGroup g;
        g.impl = Impl::From(buf);
        return g;
    }
    __host__ __device__ int4 Into() const { return impl.Into(); }
};
static_assert(Groupable<UintGroup>);

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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

template <int mul>
struct AesCtx {
    cuda::std::array<EVP_CIPHER_CTX *, mul> ctxs;
    fss::prg::Aes128Mmo<mul> prg;

    AesCtx() : ctxs(MakeCtxs()), prg(ctxs) {}
    ~AesCtx() { fss::prg::Aes128Mmo<mul>::FreeCtxs(ctxs); }

private:
    static cuda::std::array<EVP_CIPHER_CTX *, mul> MakeCtxs() {
        const unsigned char *keys[mul];
        for (int i = 0; i < mul; i++) keys[i] = kAesKeys[i];
        return fss::prg::Aes128Mmo<mul>::CreateCtxs(keys);
    }
};

// ============================================================
// CPU DPF benchmarks
// ============================================================

template <typename Group>
static void BM_CpuDpfGen(benchmark::State &state) {
    using DpfT = fss::Dpf<kInBits, Group, fss::prg::Aes128Mmo<2>, uint32_t, 0>;
    int4 seeds[2] = {kSeeds[0], kSeeds[1]};
    typename DpfT::Cw cws[kInBits + 1];
    AesCtx<2> ctx;
    DpfT dpf{ctx.prg};
    for (auto _ : state) {
        dpf.Gen(cws, seeds, kAlpha, kBeta);
        benchmark::DoNotOptimize(cws);
    }
}

template <typename Group>
static void BM_CpuDpfEval(benchmark::State &state) {
    using DpfT = fss::Dpf<kInBits, Group, fss::prg::Aes128Mmo<2>, uint32_t, 0>;
    int4 seeds[2] = {kSeeds[0], kSeeds[1]};
    typename DpfT::Cw cws[kInBits + 1];
    AesCtx<2> ctx;
    DpfT dpf{ctx.prg};
    dpf.Gen(cws, seeds, kAlpha, kBeta);
    uint32_t x = 0;
    for (auto _ : state) {
        int4 y = dpf.Eval(false, seeds[0], cws, x);
        benchmark::DoNotOptimize(y);
        x = (x + 1) & ((1u << kInBits) - 1);
    }
}

template <typename Group>
static void BM_CpuDpfEvalAll(benchmark::State &state) {
    using DpfT = fss::Dpf<kInBits, Group, fss::prg::Aes128Mmo<2>, uint32_t, 0>;
    int4 seeds[2] = {kSeeds[0], kSeeds[1]};
    typename DpfT::Cw cws[kInBits + 1];
    constexpr size_t n = size_t{1} << kInBits;
    std::vector<int4> ys(n);
    AesCtx<2> ctx;
    DpfT dpf{ctx.prg};
    dpf.Gen(cws, seeds, kAlpha, kBeta);
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
    using DcfT = fss::Dcf<kInBits, Group, fss::prg::Aes128Mmo<4>, uint32_t,
        fss::DcfPred::kLt, 0>;
    int4 seeds[2] = {kSeeds[0], kSeeds[1]};
    typename DcfT::Cw cws[kInBits + 1];
    AesCtx<4> ctx;
    DcfT dcf{ctx.prg};
    for (auto _ : state) {
        dcf.Gen(cws, seeds, kAlpha, kBeta);
        benchmark::DoNotOptimize(cws);
    }
}

template <typename Group>
static void BM_CpuDcfEval(benchmark::State &state) {
    using DcfT = fss::Dcf<kInBits, Group, fss::prg::Aes128Mmo<4>, uint32_t,
        fss::DcfPred::kLt, 0>;
    int4 seeds[2] = {kSeeds[0], kSeeds[1]};
    typename DcfT::Cw cws[kInBits + 1];
    AesCtx<4> ctx;
    DcfT dcf{ctx.prg};
    dcf.Gen(cws, seeds, kAlpha, kBeta);
    uint32_t x = 0;
    for (auto _ : state) {
        int4 y = dcf.Eval(false, seeds[0], cws, x);
        benchmark::DoNotOptimize(y);
        x = (x + 1) & ((1u << kInBits) - 1);
    }
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

// ============================================================
// GPU DPF/DCF kernels (ChaCha PRG)
// ============================================================

__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

template <int in_bits, typename Group>
__global__ void DpfGenKernel(
    typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const int4 *seeds,
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
__global__ void DcfGenKernel(
    typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const int4 *seeds,
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
        cudaFree(d_seeds);
        cudaFree(d_seeds0);
        cudaFree(d_alphas);
        cudaFree(d_betas);
        cudaFree(d_xs);
        cudaFree(d_ys);
    }
};

// ============================================================
// GPU DPF benchmarks
// ============================================================

template <typename Group>
static void BM_GpuDpfGen(benchmark::State &state) {
    using DpfT = fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint>;
    GpuData data;
    typename DpfT::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfT::Cw) * (kInBits + 1) * kN));

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        DpfGenKernel<kInBits, Group>
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

template <typename Group>
static void BM_GpuDpfEval(benchmark::State &state) {
    using DpfT = fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint>;
    GpuData data;
    typename DpfT::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfT::Cw) * (kInBits + 1) * kN));

    DpfGenKernel<kInBits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        DpfEvalKernel<kInBits, Group>
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

// ============================================================
// GPU DCF benchmarks
// ============================================================

template <typename Group>
static void BM_GpuDcfGen(benchmark::State &state) {
    using DcfT = fss::Dcf<kInBits, Group, fss::prg::ChaCha<4>, uint>;
    GpuData data;
    typename DcfT::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfT::Cw) * (kInBits + 1) * kN));

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        DcfGenKernel<kInBits, Group>
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

template <typename Group>
static void BM_GpuDcfEval(benchmark::State &state) {
    using DcfT = fss::Dcf<kInBits, Group, fss::prg::ChaCha<4>, uint>;
    GpuData data;
    typename DcfT::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfT::Cw) * (kInBits + 1) * kN));

    DcfGenKernel<kInBits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        DcfEvalKernel<kInBits, Group>
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

// ============================================================
// AesSoft PRG benchmarks
// ============================================================

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
    uint32_t x = 0;
    for (auto _ : state) {
        int4 y = dpf.Eval(false, seeds[0], cws, x);
        benchmark::DoNotOptimize(y);
        x = (x + 1) & ((1u << kInBits) - 1);
    }
}

BENCHMARK(BM_CpuAesSoftEval)->Name("fss/CPU/AesSoft");

// GPU AesSoft DPF kernels

__constant__ uint8_t kAesSoftKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

__global__ void AesSoftDpfGenKernel(
    typename fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>::Cw *cws,
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
    const typename fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>::Cw *cws,
    const uint *xs) {
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
    using DpfT = fss::Dpf<kInBits, BytesGroup, fss::prg::Aes128Soft<2>, uint>;
    GpuData data;
    typename DpfT::Cw *d_cws;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfT::Cw) * (kInBits + 1) * kN));

    AesSoftDpfGenKernel
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        AesSoftDpfEvalKernel
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
    // AesSoft DPF Eval
    benchmark::RegisterBenchmark("fss/GPU/AesSoft", BM_GpuAesSoftEval)->UseManualTime();
}

static int gpu_reg_ = (RegisterGpuBenchmarks(), 0);
