// DPF CUDA Benchmark: 2^20 parallel Gen/Eval with ChaCha PRG
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>

constexpr int kN = 1 << 20;  // 2^20 parallel operations
constexpr int kInBits = 16;  // 16-bit input domain
constexpr int kThreadsPerBlock = 256;

using In = uint16_t;
using Group = fss::group::Bytes;
using Prg = fss::prg::ChaCha<2>;
using Dpf = fss::Dpf<In, kInBits, Group, Prg>;

// Shared nonce for ChaCha
__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

// Generate DPF keys in parallel
__global__ void GenKernel(
    typename Dpf::Cw *cws, const int4 *s0s, const In *alphas, const int4 *betas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    Prg prg(kNonce);
    Dpf dpf{prg};

    int4 seeds[2] = {s0s[tid * 2], s0s[tid * 2 + 1]};
    typename Dpf::Cw *key = cws + tid * (kInBits + 1);

    dpf.Gen(key, seeds, alphas[tid], betas[tid]);
}

// Evaluate DPF at given points in parallel
__global__ void EvalKernel(
    int4 *ys, bool b, const int4 *s0s, const typename Dpf::Cw *cws, const In *xs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    Prg prg(kNonce);
    Dpf dpf{prg};

    const typename Dpf::Cw *key = cws + tid * (kInBits + 1);
    int4 seed = s0s[tid];

    ys[tid] = dpf.Eval(b, seed, key, xs[tid]);
}

static double GetTime() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

int main() {
    printf("DPF Benchmark: %d parallel Gen/Eval (input bits: %d)\n", kN, kInBits);

    srand(42);
    int blocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;
    double t;

    // Allocate host memory
    size_t cw_size = sizeof(typename Dpf::Cw) * (kInBits + 1) * kN;
    int4 *h_s0s = (int4 *)malloc(sizeof(int4) * 2 * kN);
    In *h_alphas = (In *)malloc(sizeof(In) * kN);
    int4 *h_betas = (int4 *)malloc(sizeof(int4) * kN);
    In *h_xs = (In *)malloc(sizeof(In) * kN);
    int4 *h_y0s = (int4 *)malloc(sizeof(int4) * kN);
    int4 *h_y1s = (int4 *)malloc(sizeof(int4) * kN);

    // Initialize random data
    for (int i = 0; i < kN; i++) {
        // Random seeds (with LSB cleared for control bit)
        h_s0s[i * 2] = {rand(), rand(), rand() & ~1, rand()};
        h_s0s[i * 2 + 1] = {rand(), rand(), rand() & ~1, rand()};
        // Random alpha, beta, x
        h_alphas[i] = rand() & ((1 << kInBits) - 1);
        h_betas[i] = {rand(), rand(), rand() & ~1, rand()};
        h_xs[i] = (i == 0) ? h_alphas[0] : (rand() & ((1 << kInBits) - 1));
    }

    // Allocate device memory
    typename Dpf::Cw *d_cws;
    int4 *d_s0s, *d_betas, *d_ys;
    In *d_alphas, *d_xs;

    CUDA_CHECK(cudaMalloc(&d_cws, cw_size));
    CUDA_CHECK(cudaMalloc(&d_s0s, sizeof(int4) * 2 * kN));
    CUDA_CHECK(cudaMalloc(&d_alphas, sizeof(In) * kN));
    CUDA_CHECK(cudaMalloc(&d_betas, sizeof(int4) * kN));
    CUDA_CHECK(cudaMalloc(&d_xs, sizeof(In) * kN));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kN));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_s0s, h_s0s, sizeof(int4) * 2 * kN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas, sizeof(In) * kN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_betas, h_betas, sizeof(int4) * kN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, sizeof(In) * kN, cudaMemcpyHostToDevice));

    // Benchmark Gen
    t = GetTime();
    GenKernel<<<blocks, kThreadsPerBlock>>>(d_cws, d_s0s, d_alphas, d_betas);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Gen: %.3f ms (%.0f ops/s)\n", (GetTime() - t) * 1e3, kN / (GetTime() - t));

    // Benchmark Eval (party 0)
    t = GetTime();
    EvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, false, d_s0s, d_cws, d_xs);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Eval (party 0): %.3f ms (%.0f ops/s)\n", (GetTime() - t) * 1e3, kN / (GetTime() - t));

    CUDA_CHECK(cudaMemcpy(h_y0s, d_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));

    // Benchmark Eval (party 1)
    // Party 1 uses s0s[1] (offset by kN in our layout)
    int4 *d_s1s;
    CUDA_CHECK(cudaMalloc(&d_s1s, sizeof(int4) * kN));
    int4 *h_s1s = (int4 *)malloc(sizeof(int4) * kN);
    for (int i = 0; i < kN; i++) h_s1s[i] = h_s0s[i * 2 + 1];
    CUDA_CHECK(cudaMemcpy(d_s1s, h_s1s, sizeof(int4) * kN, cudaMemcpyHostToDevice));

    t = GetTime();
    EvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, true, d_s1s, d_cws, d_xs);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Eval (party 1): %.3f ms (%.0f ops/s)\n", (GetTime() - t) * 1e3, kN / (GetTime() - t));

    CUDA_CHECK(cudaMemcpy(h_y1s, d_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));

    // Verify: y0 XOR y1 should equal beta at alpha, zero otherwise
    int correct = 0;
    for (int i = 0; i < kN; i++) {
        int4 sum = {h_y0s[i].x ^ h_y1s[i].x, h_y0s[i].y ^ h_y1s[i].y, h_y0s[i].z ^ h_y1s[i].z,
            h_y0s[i].w ^ h_y1s[i].w};
        bool is_alpha = (h_xs[i] == h_alphas[i]);
        bool ok;
        if (is_alpha) {
            ok = (sum.x == h_betas[i].x && sum.y == h_betas[i].y && sum.z == h_betas[i].z &&
                sum.w == h_betas[i].w);
        } else {
            ok = (sum.x == 0 && sum.y == 0 && sum.z == 0 && sum.w == 0);
        }
        if (ok) correct++;
    }
    printf("Verification: %d/%d correct\n", correct, kN);

    // Cleanup
    free(h_s0s);
    free(h_s1s);
    free(h_alphas);
    free(h_betas);
    free(h_xs);
    free(h_y0s);
    free(h_y1s);
    CUDA_CHECK(cudaFree(d_cws));
    CUDA_CHECK(cudaFree(d_s0s));
    CUDA_CHECK(cudaFree(d_s1s));
    CUDA_CHECK(cudaFree(d_alphas));
    CUDA_CHECK(cudaFree(d_betas));
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));

    return 0;
}
