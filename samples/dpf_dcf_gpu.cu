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
#include <openssl/rand.h>
#include <vector>

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
  dpf.Gen(k0s[tid].cws, s, alphas[tid], betas[tid]);
  k0s[tid].s0 = s[0];
  k1s[tid].s0 = s[1];
  for (int i = 0; i <= kInBits; ++i) k1s[tid].cws[i] = k0s[tid].cws[i];
}

__global__ void DpfEvalKernel(int4 *ys, bool party, const Dpf::Key *keys, const In *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DpfPrg prg(kNonce0, kNonce1);
  Dpf dpf{prg};

  ys[tid] = dpf.Eval(party, keys[tid].s0, keys[tid].cws, xs[tid]);
}

// --- DCF Kernels ---

__global__ void DcfGenKernel(Dcf::Key *k0s, Dcf::Key *k1s, const int4 *seeds, const In *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DcfPrg prg(kNonce0, kNonce1);
  Dcf dcf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dcf.Gen(k0s[tid].cws, s, alphas[tid], betas[tid]);
  k0s[tid].s0 = s[0];
  k1s[tid].s0 = s[1];
  for (int i = 0; i <= kInBits; ++i) k1s[tid].cws[i] = k0s[tid].cws[i];
}

__global__ void DcfEvalKernel(int4 *ys, bool party, const Dcf::Key *keys, const In *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DcfPrg prg(kNonce0, kNonce1);
  Dcf dcf{prg};

  ys[tid] = dcf.Eval(party, keys[tid].s0, keys[tid].cws, xs[tid]);
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

  srand(42);  // Only the public demonstration inputs use rand().
  for (int i = 0; i < kN; i++) {


    h_alphas[i] = rand() & 0xFF;
    h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
    // Eval at alpha for even indices, random x for odd
    h_xs[i] = (i % 2 == 0) ? h_alphas[i] : (rand() & 0xFF);
  }

  if (RAND_bytes(reinterpret_cast<unsigned char *>(h_seeds), sizeof(h_seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    exit(1);
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

  srand(123);  // Only the public demonstration inputs use rand().
  for (int i = 0; i < kN; i++) {


    h_alphas[i] = 1 + (rand() % 254);  // alpha in [1, 254] to allow x < and x > cases
    h_betas[i] = {rand(), rand(), rand(), rand() & ~1};
    h_xs[i] = rand() & 0xFF;
  }

  if (RAND_bytes(reinterpret_cast<unsigned char *>(h_seeds), sizeof(h_seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    exit(1);
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

// Full-domain launches do not require the caller to write an evaluation kernel.
// ChaCha is self-contained, so this host scheme has no borrowed device tables.
template <typename Scheme>
static void FullDomainSample(Scheme scheme, const char *name, bool comparison) {
  constexpr int kKeys = 2;
  constexpr int kOutputs = 1 << kInBits;
  constexpr int kCorrections = kInBits + 1;
  int4 seeds[kKeys][2];
  if (RAND_bytes(reinterpret_cast<unsigned char *>(seeds), sizeof(seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    exit(1);
  }
  typename Scheme::Cw corrections[kKeys][kCorrections];
  int4 party0[kKeys], party1[kKeys];
  const int4 beta = {7, 0, 0, 0};
  for (int i = 0; i < kKeys; ++i) {
    scheme.Gen(corrections[i], seeds[i], static_cast<In>(42 + i), beta);
    party0[i] = seeds[i][0];
    party1[i] = seeds[i][1];
  }
  auto *device_cws = ToDevice(&corrections[0][0], kKeys * kCorrections);
  auto *device_seeds = ToDevice(party0, kKeys);
  int4 *device_y0, *device_y1;
  CUDA_CHECK(cudaMalloc(&device_y0, kKeys * kOutputs * sizeof(int4)));
  CUDA_CHECK(cudaMalloc(&device_y1, kKeys * kOutputs * sizeof(int4)));

  // Corrections and outputs are key-major; the party is shared by the batch.
  CUDA_CHECK(scheme.EvalAllGpuBatch(false,
      cuda::std::span<const int4>(device_seeds, kKeys),
      cuda::std::span<const typename Scheme::Cw>(device_cws, kKeys * kCorrections),
      cuda::std::span<int4>(device_y0, kKeys * kOutputs)));
  for (int i = 0; i < kKeys; ++i) {
    CUDA_CHECK(scheme.EvalAllGpu(true, party1[i],
        cuda::std::span<const typename Scheme::Cw>(device_cws + i * kCorrections, kCorrections),
        cuda::std::span<int4>(device_y1 + i * kOutputs, kOutputs)));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int4> y0(kKeys * kOutputs), y1(kKeys * kOutputs);
  ToHost(y0.data(), device_y0, kKeys * kOutputs);
  ToHost(y1.data(), device_y1, kKeys * kOutputs);
  int mismatches = 0;
  for (int i = 0; i < kKeys; ++i) {
    for (int x = 0; x < kOutputs; ++x) {
      const bool active = comparison ? x < 42 + i : x == 42 + i;
      int4 expected = active ? beta : int4{0, 0, 0, 0};
      int4 actual = (Group::From(y0[i * kOutputs + x]) + Group::From(y1[i * kOutputs + x])).Into();
      if (memcmp(&actual, &expected, sizeof(int4)) != 0) ++mismatches;
    }
  }
  printf("  %s EvalAllGpu/Batch: mismatches: %d\n", name, mismatches);
  CUDA_CHECK(cudaFree(device_cws));
  CUDA_CHECK(cudaFree(device_seeds));
  CUDA_CHECK(cudaFree(device_y0));
  CUDA_CHECK(cudaFree(device_y1));
}

int main() {
  DpfSample();
  printf("\n");
  DcfSample();
  FullDomainSample(Dpf{DpfPrg{kNonce0, kNonce1}}, "DPF", false);
  FullDomainSample(Dcf{DcfPrg{kNonce0, kNonce1}}, "DCF", true);
  return 0;
}
