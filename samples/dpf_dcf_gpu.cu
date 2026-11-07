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

// True when the scheme's correction words carry a separate group-value word
// (DCF); its component Gen and evaluation calls take the second array.
template <typename Scheme>
inline constexpr bool kHasValueWords = requires { std::declval<typename Scheme::Cw &>().v; };

// A device-resident key in the split layout: correction words plus the packed
// control value.
template <int Bits, typename Input>
struct DeviceKey {
  int4 cws[Bits + 1];
  Input tr;
  int4 s0;
};

__global__ void DpfGenKernel(DeviceKey<kInBits, In> *k0s, DeviceKey<kInBits, In> *k1s,
    const int4 *seeds, const In *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DpfPrg prg(kNonce0, kNonce1);
  Dpf dpf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(cuda::std::span<int4>(k0s[tid].cws, kInBits + 1), k0s[tid].tr,
      cuda::std::span<const int4, 2>(s, 2), alphas[tid], betas[tid]);
  k0s[tid].s0 = s[0];
  k1s[tid].s0 = s[1];
  k1s[tid].tr = k0s[tid].tr;
  for (int i = 0; i <= kInBits; ++i) k1s[tid].cws[i] = k0s[tid].cws[i];
}

__global__ void DpfEvalKernel(int4 *ys, bool party, const DeviceKey<kInBits, In> *keys, const In *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DpfPrg prg(kNonce0, kNonce1);
  Dpf dpf{prg};

  ys[tid] = dpf.Eval(party, keys[tid].s0, cuda::std::span<const int4>(keys[tid].cws, kInBits + 1),
      keys[tid].tr, xs[tid]);
}

// --- DCF Kernels ---

// A DCF device key adds the value components after the seed components.
template <int Bits, typename Input>
struct DcfDeviceKey {
  int4 cws[Bits + 1];
  int4 vws[Bits + 1];
  Input tr;
  int4 s0;
};

__global__ void DcfGenKernel(DcfDeviceKey<kInBits, In> *k0s, DcfDeviceKey<kInBits, In> *k1s,
    const int4 *seeds, const In *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DcfPrg prg(kNonce0, kNonce1);
  Dcf dcf{prg};

  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dcf.Gen(cuda::std::span<int4>(k0s[tid].cws, kInBits + 1),
      cuda::std::span<int4>(k0s[tid].vws, kInBits + 1), k0s[tid].tr,
      cuda::std::span<const int4, 2>(s, 2), alphas[tid], betas[tid]);
  k0s[tid].s0 = s[0];
  k1s[tid].s0 = s[1];
  k1s[tid].tr = k0s[tid].tr;
  for (int i = 0; i <= kInBits; ++i) {
    k1s[tid].cws[i] = k0s[tid].cws[i];
    k1s[tid].vws[i] = k0s[tid].vws[i];
  }
}

__global__ void DcfEvalKernel(int4 *ys, bool party, const DcfDeviceKey<kInBits, In> *keys, const In *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kN) return;

  DcfPrg prg(kNonce0, kNonce1);
  Dcf dcf{prg};

  ys[tid] = dcf.Eval(party, keys[tid].s0, cuda::std::span<const int4>(keys[tid].cws, kInBits + 1),
      cuda::std::span<const int4>(keys[tid].vws, kInBits + 1), keys[tid].tr, xs[tid]);
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
  using DpfKey = DeviceKey<kInBits, In>;
  DpfKey *d_k0s, *d_k1s;
  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_k0s, sizeof(DpfKey) * kN));
  CUDA_CHECK(cudaMalloc(&d_k1s, sizeof(DpfKey) * kN));
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
  using DcfKey = DcfDeviceKey<kInBits, In>;
  DcfKey *d_k0s, *d_k1s;
  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_k0s, sizeof(DcfKey) * kN));
  CUDA_CHECK(cudaMalloc(&d_k1s, sizeof(DcfKey) * kN));
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
  constexpr int kCwPerKey = kInBits + 1;
  int4 seeds[kKeys][2];
  if (RAND_bytes(reinterpret_cast<unsigned char *>(seeds), sizeof(seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    exit(1);
  }
  // The batch entry consumes the dense level-major layout: layer i of key k is
  // cws[i * kKeys + k], with the packed control per key in trs[k].
  int4 cws[kKeys][kCwPerKey];
  int4 vws[kKeys][kCwPerKey];
  In trs[kKeys];
  int4 party0[kKeys], party1[kKeys];
  const int4 beta = {7, 0, 0, 0};
  for (int i = 0; i < kKeys; ++i) {
    if constexpr (kHasValueWords<Scheme>) {
      scheme.Gen(cuda::std::span<int4>(cws[i], kCwPerKey), cuda::std::span<int4>(vws[i], kCwPerKey),
          trs[i], cuda::std::span<const int4, 2>(seeds[i], 2), static_cast<In>(42 + i), beta, kKeys);
    } else {
      scheme.Gen(cuda::std::span<int4>(cws[i], kCwPerKey), trs[i],
          cuda::std::span<const int4, 2>(seeds[i], 2), static_cast<In>(42 + i), beta, kKeys);
    }
    party0[i] = seeds[i][0];
    party1[i] = seeds[i][1];
  }
  auto *device_cws = ToDevice(&cws[0][0], kKeys * kCwPerKey);
  auto *device_vws = ToDevice(&vws[0][0], kKeys * kCwPerKey);
  auto *device_trs = ToDevice(trs, kKeys);
  auto *device_seeds = ToDevice(party0, kKeys);
  int4 *device_y0, *device_y1;
  CUDA_CHECK(cudaMalloc(&device_y0, kKeys * kOutputs * sizeof(int4)));
  CUDA_CHECK(cudaMalloc(&device_y1, kKeys * kOutputs * sizeof(int4)));

  // Outputs are key-major; the party is shared by the batch.
  if constexpr (kHasValueWords<Scheme>) {
    CUDA_CHECK(scheme.EvalAllGpuBatch(false,
        cuda::std::span<const int4>(device_seeds, kKeys),
        cuda::std::span<const int4>(device_cws, kKeys * kCwPerKey),
        cuda::std::span<const int4>(device_vws, kKeys * kCwPerKey),
        cuda::std::span<const In>(device_trs, kKeys),
        cuda::std::span<int4>(device_y0, kKeys * kOutputs), 0, kKeys));
    for (int i = 0; i < kKeys; ++i) {
      CUDA_CHECK(scheme.EvalAllGpu(true, party1[i],
          cuda::std::span<const int4>(device_cws + i, kKeys * kCwPerKey - i),
          cuda::std::span<const int4>(device_vws + i, kKeys * kCwPerKey - i), trs[i],
          cuda::std::span<int4>(device_y1 + i * kOutputs, kOutputs), 0, kKeys));
    }
  } else {
    CUDA_CHECK(scheme.EvalAllGpuBatch(false,
        cuda::std::span<const int4>(device_seeds, kKeys),
        cuda::std::span<const int4>(device_cws, kKeys * kCwPerKey),
        cuda::std::span<const In>(device_trs, kKeys),
        cuda::std::span<int4>(device_y0, kKeys * kOutputs), 0, kKeys));
    for (int i = 0; i < kKeys; ++i) {
      CUDA_CHECK(scheme.EvalAllGpu(true, party1[i],
          cuda::std::span<const int4>(device_cws + i, kKeys * kCwPerKey - i), trs[i],
          cuda::std::span<int4>(device_y1 + i * kOutputs, kOutputs), 0, kKeys));
    }
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
