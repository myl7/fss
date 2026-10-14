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

// ChaCha nonce (stored by value, host and device share one instance)
constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

#define CUDA_CHECK(x) \
  do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

// --- DPF Kernels ---

__global__ void DpfGenKernel(Dpf::Key *k0s, Dpf::Key *k1s, const int4 *seeds, const In *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DpfPrg prg(kNonce0, kNonce1);
  Dpf dpf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0s[tid], k1s[tid], cuda::std::span<const int4, 2>(s, 2), alphas[tid], betas[tid]);
}

__global__ void DpfEvalKernel(int4 *ys, bool party, const Dpf::Key *keys, const In *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DpfPrg prg(kNonce0, kNonce1);
  Dpf dpf{prg};

  ys[tid] = dpf.Eval(party, keys[tid], xs[tid]);
}

// --- DCF Kernels ---

__global__ void DcfGenKernel(Dcf::Key *k0s, Dcf::Key *k1s, const int4 *seeds, const In *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DcfPrg prg(kNonce0, kNonce1);
  Dcf dcf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dcf.Gen(k0s[tid], k1s[tid], cuda::std::span<const int4, 2>(s, 2), alphas[tid], betas[tid]);
}

__global__ void DcfEvalKernel(int4 *ys, bool party, const Dcf::Key *keys, const In *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DcfPrg prg(kNonce0, kNonce1);
  Dcf dcf{prg};

  ys[tid] = dcf.Eval(party, keys[tid], xs[tid]);
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
  In h_alphas[kN];
  int4 h_betas[kN];
  In h_xs[kN];

  srand(42);
  for (int i = 0; i < kN; i++) {
    h_seeds[i * 2] = {rand(), rand(), rand(), rand() & ~1};
    h_seeds[i * 2 + 1] = {rand(), rand(), rand(), rand() & ~1};
    h_alphas[i] = rand() & 0xFF;
    h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
    // Eval at alpha for even indices, random x for odd
    h_xs[i] = (i % 2 == 0) ? h_alphas[i] : (rand() & 0xFF);
  }

  // Copy to device
  int4 *d_seeds = ToDevice(h_seeds, kN * 2);
  In *d_alphas = ToDevice(h_alphas, kN);
  int4 *d_betas = ToDevice(h_betas, kN);
  In *d_xs = ToDevice(h_xs, kN);

  // Allocate output buffers
  Dpf::Key *d_k0s, *d_k1s;
  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_k0s, sizeof(Dpf::Key) * kN));
  CUDA_CHECK(cudaMalloc(&d_k1s, sizeof(Dpf::Key) * kN));
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kN));

  int blocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

  // Gen
  DpfGenKernel<<<blocks, kThreadsPerBlock>>>(d_k0s, d_k1s, d_seeds, d_alphas, d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Eval party 0
  DpfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, false, d_k0s, d_xs);
  CUDA_CHECK(cudaDeviceSynchronize());
  int4 h_y0s[kN];
  ToHost(h_y0s, d_ys, kN);

  // Eval party 1
  DpfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, true, d_k1s, d_xs);
  CUDA_CHECK(cudaDeviceSynchronize());
  int4 h_y1s[kN];
  ToHost(h_y1s, d_ys, kN);

  // Verify
  int correct = 0;
  for (int i = 0; i < kN; i++) {
    int4 sum = (Group::From(h_y0s[i]) + Group::From(h_y1s[i])).Into();
    int4 expected = (h_xs[i] == h_alphas[i]) ? h_betas[i] : int4{0, 0, 0, 0};
    if (memcmp(&sum, &expected, sizeof(int4)) == 0) correct++;
  }
  printf("  Verification: %d/%d correct\n", correct, kN);

  cudaFree(d_seeds);
  cudaFree(d_alphas);
  cudaFree(d_betas);
  cudaFree(d_xs);
  cudaFree(d_k0s);
  cudaFree(d_k1s);
  cudaFree(d_ys);
}

static void DcfSample() {
  printf("=== DCF GPU Sample (%d instances) ===\n", kN);

  // Prepare host data
  int4 h_seeds[kN * 2];
  In h_alphas[kN];
  int4 h_betas[kN];
  In h_xs[kN];

  srand(123);
  for (int i = 0; i < kN; i++) {
    h_seeds[i * 2] = {rand(), rand(), rand(), rand() & ~1};
    h_seeds[i * 2 + 1] = {rand(), rand(), rand(), rand() & ~1};
    h_alphas[i] = 1 + (rand() % 254);  // alpha in [1, 254] to allow x < and x > cases
    h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
    h_xs[i] = rand() & 0xFF;
  }

  // Copy to device
  int4 *d_seeds = ToDevice(h_seeds, kN * 2);
  In *d_alphas = ToDevice(h_alphas, kN);
  int4 *d_betas = ToDevice(h_betas, kN);
  In *d_xs = ToDevice(h_xs, kN);

  // Allocate output buffers
  Dcf::Key *d_k0s, *d_k1s;
  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_k0s, sizeof(Dcf::Key) * kN));
  CUDA_CHECK(cudaMalloc(&d_k1s, sizeof(Dcf::Key) * kN));
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kN));

  int blocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

  // Gen
  DcfGenKernel<<<blocks, kThreadsPerBlock>>>(d_k0s, d_k1s, d_seeds, d_alphas, d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Eval party 0
  DcfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, false, d_k0s, d_xs);
  CUDA_CHECK(cudaDeviceSynchronize());
  int4 h_y0s[kN];
  ToHost(h_y0s, d_ys, kN);

  // Eval party 1
  DcfEvalKernel<<<blocks, kThreadsPerBlock>>>(d_ys, true, d_k1s, d_xs);
  CUDA_CHECK(cudaDeviceSynchronize());
  int4 h_y1s[kN];
  ToHost(h_y1s, d_ys, kN);

  // Verify: y0 + y1 == beta when x < alpha, 0 otherwise
  int correct = 0;
  for (int i = 0; i < kN; i++) {
    int4 sum = (Group::From(h_y0s[i]) + Group::From(h_y1s[i])).Into();
    int4 expected = (h_xs[i] < h_alphas[i]) ? h_betas[i] : int4{0, 0, 0, 0};
    if (memcmp(&sum, &expected, sizeof(int4)) == 0) correct++;
  }
  printf("  Verification: %d/%d correct\n", correct, kN);

  cudaFree(d_seeds);
  cudaFree(d_alphas);
  cudaFree(d_betas);
  cudaFree(d_xs);
  cudaFree(d_k0s);
  cudaFree(d_k1s);
  cudaFree(d_ys);
}

int main() {
  DpfSample();
  printf("\n");
  DcfSample();
  return 0;
}
