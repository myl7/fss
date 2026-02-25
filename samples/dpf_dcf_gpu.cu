// Sample: DPF and DCF on GPU
//
// Shows how to run Gen/Eval inside CUDA kernels using ChaCha PRG.
// Each thread handles one independent DPF/DCF instance.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>

constexpr int kN = 1024;  // Number of parallel instances
constexpr int kInBits = 8;
constexpr int kThreadsPerBlock = 256;

using In = uint8_t;
using Group = fss::group::Bytes;
using DpfPrg = fss::prg::ChaCha<2>;
using DcfPrg = fss::prg::ChaCha<4>;
using Dpf = fss::Dpf<kInBits, Group, DpfPrg, In>;
using Dcf = fss::Dcf<kInBits, Group, DcfPrg, In>;

// Nonce in constant memory (accessible from GPU kernels)
__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// --- DPF Kernels ---

__global__ void DpfGenKernel(Dpf::Cw *cws, const int4 *seeds, const In *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    DpfPrg prg(kNonce);
    Dpf dpf{prg};

    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dpf.Gen(cws + tid * (kInBits + 1), s, alphas[tid], betas[tid]);
}

__global__ void DpfEvalKernel(
    int4 *ys, bool party, const int4 *seeds, const Dpf::Cw *cws, const In *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    DpfPrg prg(kNonce);
    Dpf dpf{prg};

    ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (kInBits + 1), xs[tid]);
}

// --- DCF Kernels ---

__global__ void DcfGenKernel(Dcf::Cw *cws, const int4 *seeds, const In *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    DcfPrg prg(kNonce);
    Dcf dcf{prg};

    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dcf.Gen(cws + tid * (kInBits + 1), s, alphas[tid], betas[tid]);
}

__global__ void DcfEvalKernel(
    int4 *ys, bool party, const int4 *seeds, const Dcf::Cw *cws, const In *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    DcfPrg prg(kNonce);
    Dcf dcf{prg};

    ys[tid] = dcf.Eval(party, seeds[tid], cws + tid * (kInBits + 1), xs[tid]);
}

// --- Host helpers ---

// Allocate device memory and copy from host
template <typename T>
static T *ToDevice(const T *host, int n) {
    T *dev;
    CUDA_CHECK(cudaMalloc(&dev, sizeof(T) * n));
    CUDA_CHECK(cudaMemcpy(dev, host, sizeof(T) * n, cudaMemcpyHostToDevice));
    return dev;
}

// Copy device memory to host
template <typename T>
static void ToHost(T *host, const T *dev, int n) {
    CUDA_CHECK(cudaMemcpy(host, dev, sizeof(T) * n, cudaMemcpyDeviceToHost));
}

static void DpfSample() {
    printf("=== DPF GPU Sample (%d instances) ===\n", kN);

    // Prepare host data
    int4 h_seeds[kN * 2];
    int4 h_seeds0[kN], h_seeds1[kN];
    In h_alphas[kN];
    int4 h_betas[kN];
    In h_xs[kN];

    srand(42);
    for (int i = 0; i < kN; i++) {
        h_seeds[i * 2] = {rand(), rand(), rand(), rand() & ~1};
        h_seeds[i * 2 + 1] = {rand(), rand(), rand(), rand() & ~1};
        h_seeds0[i] = h_seeds[i * 2];
        h_seeds1[i] = h_seeds[i * 2 + 1];
        h_alphas[i] = rand() & 0xFF;
        h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
        // Eval at alpha for even indices, random x for odd
        h_xs[i] = (i % 2 == 0) ? h_alphas[i] : (rand() & 0xFF);
    }

    // Copy to device
    int4 *d_seeds = ToDevice(h_seeds, kN * 2);
    int4 *d_seeds0 = ToDevice(h_seeds0, kN);
    int4 *d_seeds1 = ToDevice(h_seeds1, kN);
    In *d_alphas = ToDevice(h_alphas, kN);
    int4 *d_betas = ToDevice(h_betas, kN);
    In *d_xs = ToDevice(h_xs, kN);

    // Allocate output buffers
    Dpf::Cw *d_cws;
    int4 *d_ys;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(Dpf::Cw) * (kInBits + 1) * kN));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kN));

    int blocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // Gen
    DpfGenKernel<<<blocks, kThreadsPerBlock>>>(d_cws, d_seeds, d_alphas, d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Eval party 0
    DpfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, false, d_seeds0, d_cws, d_xs);
    CUDA_CHECK(cudaDeviceSynchronize());
    int4 h_y0s[kN];
    ToHost(h_y0s, d_ys, kN);

    // Eval party 1
    DpfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, true, d_seeds1, d_cws, d_xs);
    CUDA_CHECK(cudaDeviceSynchronize());
    int4 h_y1s[kN];
    ToHost(h_y1s, d_ys, kN);

    // Verify
    int correct = 0;
    for (int i = 0; i < kN; i++) {
        int4 sum = {h_y0s[i].x ^ h_y1s[i].x, h_y0s[i].y ^ h_y1s[i].y, h_y0s[i].z ^ h_y1s[i].z,
            h_y0s[i].w ^ h_y1s[i].w};
        int4 expected = (h_xs[i] == h_alphas[i]) ? h_betas[i] : int4{0, 0, 0, 0};
        if (memcmp(&sum, &expected, sizeof(int4)) == 0) correct++;
    }
    printf("  Verification: %d/%d correct\n", correct, kN);

    cudaFree(d_seeds);
    cudaFree(d_seeds0);
    cudaFree(d_seeds1);
    cudaFree(d_alphas);
    cudaFree(d_betas);
    cudaFree(d_xs);
    cudaFree(d_cws);
    cudaFree(d_ys);
}

static void DcfSample() {
    printf("=== DCF GPU Sample (%d instances) ===\n", kN);

    // Prepare host data
    int4 h_seeds[kN * 2];
    int4 h_seeds0[kN], h_seeds1[kN];
    In h_alphas[kN];
    int4 h_betas[kN];
    In h_xs[kN];

    srand(123);
    for (int i = 0; i < kN; i++) {
        h_seeds[i * 2] = {rand(), rand(), rand(), rand() & ~1};
        h_seeds[i * 2 + 1] = {rand(), rand(), rand(), rand() & ~1};
        h_seeds0[i] = h_seeds[i * 2];
        h_seeds1[i] = h_seeds[i * 2 + 1];
        h_alphas[i] = 1 + (rand() % 254);  // alpha in [1, 254] to allow x < and x > cases
        h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
        h_xs[i] = rand() & 0xFF;
    }

    // Copy to device
    int4 *d_seeds = ToDevice(h_seeds, kN * 2);
    int4 *d_seeds0 = ToDevice(h_seeds0, kN);
    int4 *d_seeds1 = ToDevice(h_seeds1, kN);
    In *d_alphas = ToDevice(h_alphas, kN);
    int4 *d_betas = ToDevice(h_betas, kN);
    In *d_xs = ToDevice(h_xs, kN);

    // Allocate output buffers
    Dcf::Cw *d_cws;
    int4 *d_ys;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(Dcf::Cw) * (kInBits + 1) * kN));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kN));

    int blocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // Gen
    DcfGenKernel<<<blocks, kThreadsPerBlock>>>(d_cws, d_seeds, d_alphas, d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Eval party 0
    DcfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, false, d_seeds0, d_cws, d_xs);
    CUDA_CHECK(cudaDeviceSynchronize());
    int4 h_y0s[kN];
    ToHost(h_y0s, d_ys, kN);

    // Eval party 1
    DcfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, true, d_seeds1, d_cws, d_xs);
    CUDA_CHECK(cudaDeviceSynchronize());
    int4 h_y1s[kN];
    ToHost(h_y1s, d_ys, kN);

    // Verify: y0 + y1 == beta when x < alpha, 0 otherwise
    int correct = 0;
    for (int i = 0; i < kN; i++) {
        int4 sum = {h_y0s[i].x ^ h_y1s[i].x, h_y0s[i].y ^ h_y1s[i].y, h_y0s[i].z ^ h_y1s[i].z,
            h_y0s[i].w ^ h_y1s[i].w};
        int4 expected = (h_xs[i] < h_alphas[i]) ? h_betas[i] : int4{0, 0, 0, 0};
        if (memcmp(&sum, &expected, sizeof(int4)) == 0) correct++;
    }
    printf("  Verification: %d/%d correct\n", correct, kN);

    cudaFree(d_seeds);
    cudaFree(d_seeds0);
    cudaFree(d_seeds1);
    cudaFree(d_alphas);
    cudaFree(d_betas);
    cudaFree(d_xs);
    cudaFree(d_cws);
    cudaFree(d_ys);
}

int main() {
    DpfSample();
    printf("\n");
    DcfSample();
    return 0;
}
