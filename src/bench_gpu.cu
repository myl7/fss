#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/eval_all_gpu.cuh>
#include <fss/point_eval_gpu.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <fss/hash/blake3.cuh>

constexpr int kNumKeys = 1 << 20;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumBlocks = (kNumKeys + kThreadsPerBlock - 1) / kThreadsPerBlock;

using BytesGroup = fss::group::Bytes;
using UintGroup = fss::group::Uint<uint64_t>;


__constant__ int4 kBlake3Iv[2] = {
    {0x11223344, 0x55667788, static_cast<int>(0x99aabbccu), static_cast<int>(0xddeeff00u)},
    {0x00112233, 0x44556677, static_cast<int>(0x8899aabbu), static_cast<int>(0xccddeeffu)},
};

const int4 kHalfTreeHashKeyHost = {0x12345678, static_cast<int>(0x9abcdef0u), 0x0fedcba9, static_cast<int>(0x87654321u)};

__constant__ uint8_t kAesSoftKeys[2][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
};

#define CUDA_CHECK(x) \
  do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "cuda error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

template <typename Fn>
static void RunTimedKernel(benchmark::State &state, Fn &&fn) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start));
    std::forward<Fn>(fn)();
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

// --- DPF Kernels (ChaCha<2>) ---

template <int in_bits, typename Group>
__global__ void DpfGenKernel(typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

  typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Key k0, k1;
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s), alphas[tid], betas[tid]);
  for (int i = 0; i <= in_bits; ++i) cws[tid * (in_bits + 1) + i] = k0.cws[i];
}

template <int in_bits, typename Group>
__global__ void DpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Cw *cws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint> dpf{prg};

  typename fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i <= in_bits; ++i) key.cws[i] = cws[tid * (in_bits + 1) + i];
  ys[tid] = dpf.Eval(party, key, xs[tid]);
}

// --- DPF Kernels (Aes128Soft<2>) ---

template <int in_bits, typename Group>
__global__ void DpfGenKernelAes(typename fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>::Cw *cws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
  __shared__ uint32_t s_te0[256];
  __shared__ uint8_t s_sbox[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    s_te0[i] = fss::prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
    s_sbox[i] = fss::prg::aes_detail::Sbox(static_cast<uint8_t>(i));
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::Aes128Soft<2> prg(kAesSoftKeys, s_te0, s_sbox);
  fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint> dpf{prg};

  typename fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>::Key k0, k1;
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s), alphas[tid], betas[tid]);
  for (int i = 0; i <= in_bits; ++i) cws[tid * (in_bits + 1) + i] = k0.cws[i];
}

template <int in_bits, typename Group>
__global__ void DpfEvalKernelAes(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>::Cw *cws, const uint *xs) {
  __shared__ uint32_t s_te0[256];
  __shared__ uint8_t s_sbox[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    s_te0[i] = fss::prg::aes_detail::ComputeTe0(static_cast<uint8_t>(i));
    s_sbox[i] = fss::prg::aes_detail::Sbox(static_cast<uint8_t>(i));
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::Aes128Soft<2> prg(kAesSoftKeys, s_te0, s_sbox);
  fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint> dpf{prg};

  typename fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i <= in_bits; ++i) key.cws[i] = cws[tid * (in_bits + 1) + i];
  ys[tid] = dpf.Eval(party, key, xs[tid]);
}

// --- DCF Kernels (ChaCha<4>) ---

template <int in_bits, typename Group>
__global__ void DcfGenKernel(typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const int4 *seeds,
    const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<4> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

  typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Key k0, k1;
  int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s), alphas[tid], betas[tid]);
  for (int i = 0; i <= in_bits; ++i) cws[tid * (in_bits + 1) + i] = k0.cws[i];
}

template <int in_bits, typename Group>
__global__ void DcfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Cw *cws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<4> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint> dcf{prg};

  typename fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i <= in_bits; ++i) key.cws[i] = cws[tid * (in_bits + 1) + i];
  ys[tid] = dcf.Eval(party, key, xs[tid]);
}

// --- VDPF Kernels (ChaCha<2> + Blake3) ---

template <int in_bits, typename Group>
using VdpfType = fss::Vdpf<in_bits, Group, fss::prg::ChaCha<2>, fss::hash::Blake3, fss::hash::Blake3, uint>;

template <int in_bits, typename Group>
__global__ void VdpfGenKernel(typename VdpfType<in_bits, Group>::Cw *cws, cuda::std::array<int4, 4> *cs, int4 *ocws,
    const int4 *seeds, const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
  VdpfType<in_bits, Group> vdpf{prg, xor_hash, hash};

  typename VdpfType<in_bits, Group>::Key k0, k1;
  int4 s0s_arr[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  cuda::std::span<const int4, 2> s0s(s0s_arr);
  vdpf.Gen(k0, k1, s0s, alphas[tid], betas[tid]);
  for (int i = 0; i < in_bits; ++i) cws[tid * in_bits + i] = k0.cws[i];
  cs[tid] = k0.cs;
  ocws[tid] = k0.ocw;
}

template <int in_bits, typename Group>
__global__ void VdpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename VdpfType<in_bits, Group>::Cw *cws, const cuda::std::array<int4, 4> *cs, const int4 *ocws,
    const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
  fss::hash::Blake3 hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
  VdpfType<in_bits, Group> vdpf{prg, xor_hash, hash};

  typename VdpfType<in_bits, Group>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i < in_bits; ++i) key.cws[i] = cws[tid * in_bits + i];
  key.cs = cs[tid];
  key.ocw = ocws[tid];
  ys[tid] = vdpf.Eval(party, key, xs[tid]).y;
}

// --- HalfTreeDpf Kernels (ChaCha<1>) ---

template <int in_bits, typename Group>
using HtDpfType = fss::HalfTreeDpf<in_bits, Group, fss::prg::ChaCha<1>, uint>;

template <int in_bits, typename Group>
__global__ void HalfTreeDpfGenKernel(
    typename HtDpfType<in_bits, Group>::Cw *cws, int4 *ocws, const int4 *seeds, const uint *alphas, const int4 *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<1> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  HtDpfType<in_bits, Group> dpf{prg, kHalfTreeHashKeyHost};

  typename HtDpfType<in_bits, Group>::Key k0, k1;
  int4 s0s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s), alphas[tid], betas[tid]);
  for (int i = 0; i < in_bits; ++i) cws[tid * in_bits + i] = k0.cws[i];
  ocws[tid] = k0.ocw;
}

template <int in_bits, typename Group>
__global__ void HalfTreeDpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename HtDpfType<in_bits, Group>::Cw *cws, const int4 *ocws, const uint *xs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<1> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  HtDpfType<in_bits, Group> dpf{prg, kHalfTreeHashKeyHost};

  typename HtDpfType<in_bits, Group>::Key key;
  key.s0 = seeds[tid];
  for (int i = 0; i < in_bits; ++i) key.cws[i] = cws[tid * in_bits + i];
  key.ocw = ocws[tid];
  ys[tid] = dpf.Eval(party, key, xs[tid]);
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
    auto *h_seeds = new int4[kNumKeys * 2];
    auto *h_seeds0 = new int4[kNumKeys];
    auto *h_alphas = new uint[kNumKeys];
    auto *h_betas = new int4[kNumKeys];
    auto *h_xs = new uint[kNumKeys];

    srand(42);
    for (int i = 0; i < kNumKeys; i++) {
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
    alloc(&d_seeds, h_seeds, kNumKeys * 2);
    alloc(&d_seeds0, h_seeds0, kNumKeys);
    alloc(&d_alphas, h_alphas, kNumKeys);
    alloc(&d_betas, h_betas, kNumKeys);
    alloc(&d_xs, h_xs, kNumKeys);
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * kNumKeys));

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
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kNumKeys));

  DpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    DpfEvalKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
}

// --- DPF Gen GPU benchmark (ChaCha<2>) ---

template <int in_bits, typename Group>
static void BM_DpfGen(benchmark::State &state) {
  using DpfType = fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>;
  GpuData data;

  typename DpfType::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kNumKeys));

  RunTimedKernel(state, [&] {
    DpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
}

// --- DPF Eval GPU benchmark (Aes128Soft<2>) ---

template <int in_bits, typename Group>
static void BM_DpfEvalAes(benchmark::State &state) {
  using DpfType = fss::Dpf<in_bits, Group, fss::prg::Aes128Soft<2>, uint>;
  GpuData data;

  typename DpfType::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kNumKeys));

  DpfGenKernelAes<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    DpfEvalKernelAes<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
}

// --- DCF Eval GPU benchmark (ChaCha<4>) ---

template <int in_bits, typename Group>
static void BM_DcfEval(benchmark::State &state) {
  using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
  GpuData data;

  typename DcfType::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kNumKeys));

  DcfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    DcfEvalKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, data.d_xs);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
}

// --- DCF Gen GPU benchmark (ChaCha<4>) ---

template <int in_bits, typename Group>
static void BM_DcfGen(benchmark::State &state) {
  using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
  GpuData data;

  typename DcfType::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kNumKeys));

  RunTimedKernel(state, [&] {
    DcfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

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
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename Vdpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cs, sizeof(cuda::std::array<int4, 4>) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  VdpfGenKernel<in_bits, Group>
      <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_cs, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    VdpfEvalKernel<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, d_cs, d_ocws, data.d_xs);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

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
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename Vdpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cs, sizeof(cuda::std::array<int4, 4>) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  RunTimedKernel(state, [&] {
    VdpfGenKernel<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_cs, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

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
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename HtDpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  HalfTreeDpfGenKernel<in_bits, Group>
      <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    HalfTreeDpfEvalKernel<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(data.d_ys, false, data.d_seeds0, d_cws, d_ocws, data.d_xs);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

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
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename HtDpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  RunTimedKernel(state, [&] {
    HalfTreeDpfGenKernel<in_bits, Group>
        <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
  cudaFree(d_ocws);
}


// --- GPU full-domain eval benchmarks ---

template <int in_bits, typename Group>
static void BM_HalfTreeDpfEvalAllGpu(benchmark::State &state) {
  using HtDpf = HtDpfType<in_bits, Group>;
  GpuData data;

  // Keys per launch. One launch covers 2^b1 * kKeysPerLaunch blocks, which fills the
  // device (110 SMs) many times over and amortizes tail waves; the single-key
  // path leaves the tail ~1.5 waves and runs ~25% slower per key. Large
  // Multiple keys per launch also amortize the ~0.2us inter-launch gap over
  // the kNumKeys-key iteration.
  constexpr int kKeysPerLaunch = 1024;

  typename HtDpf::Cw *d_cws;
  int4 *d_ocws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename HtDpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  HalfTreeDpfGenKernel<in_bits, Group>
      <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  // EvalAllGpuBatch takes party-major seeds; GpuData keeps them key-major
  // (seeds[i*2 + b]). Re-lay all kNumKeys keys party-major.
  int4 *d_s0s;
  CUDA_CHECK(cudaMalloc(&d_s0s, sizeof(int4) * kNumKeys * 2));
  for (int b = 0; b < 2; ++b)
    CUDA_CHECK(cudaMemcpy2D(d_s0s + (size_t)b * kNumKeys, sizeof(int4), data.d_seeds + b, 2 * sizeof(int4),
        sizeof(int4), kNumKeys, cudaMemcpyDeviceToDevice));

  fss::prg::ChaCha<1> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  HtDpfType<in_bits, Group> dpf{prg, kHalfTreeHashKeyHost};

  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * (1 << in_bits) * kKeysPerLaunch));

  // Warm the clocks once; the harness's first timed iteration otherwise pays
  // the boost ramp from idle (idle 180 MHz vs sustained ~2.6 GHz costs ~10%).
  fss::gpu::HalfTreeDpfEvalAllGpuBatch<17, 9, 256>(
      false, d_s0s, d_cws, d_ocws, kKeysPerLaunch, d_ys, dpf);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    for (int off = 0; off < kNumKeys; off += kKeysPerLaunch)
      fss::gpu::HalfTreeDpfEvalAllGpuBatch<17, 9, 256>(
          false, d_s0s + off, d_cws + off * in_bits, d_ocws + off, kKeysPerLaunch, d_ys, dpf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys * (1 << in_bits));

  cudaFree(d_cws);
  cudaFree(d_ocws);
  cudaFree(d_s0s);
  cudaFree(d_ys);
}

template <int in_bits, typename Group>
static void BM_DpfEvalAllGpu(benchmark::State &state) {
  using DpfType = fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>;
  GpuData data;

  // See BM_HalfTreeDpfEvalAllGpu: process several keys per launch so the grid
  // fills the device and tail waves / launch overhead amortize.
  constexpr int kKeysPerLaunch = 1024;

  typename DpfType::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kNumKeys));

  DpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  int4 *d_s0s;
  CUDA_CHECK(cudaMalloc(&d_s0s, sizeof(int4) * kNumKeys * 2));
  for (int b = 0; b < 2; ++b)
    CUDA_CHECK(cudaMemcpy2D(d_s0s + (size_t)b * kNumKeys, sizeof(int4), data.d_seeds + b, 2 * sizeof(int4),
        sizeof(int4), kNumKeys, cudaMemcpyDeviceToDevice));

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  DpfType dpf{prg};

  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * (1 << in_bits) * kKeysPerLaunch));

  // Warm the clocks once, see BM_HalfTreeDpfEvalAllGpu.
  fss::gpu::DpfEvalAllGpuBatch<17, 9, 256>(false, d_s0s, d_cws, kKeysPerLaunch, d_ys, dpf);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    for (int off = 0; off < kNumKeys; off += kKeysPerLaunch)
      fss::gpu::DpfEvalAllGpuBatch<17, 9, 256>(
          false, d_s0s + off, d_cws + off * (in_bits + 1), kKeysPerLaunch, d_ys, dpf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys * (1 << in_bits));

  cudaFree(d_cws);
  cudaFree(d_s0s);
  cudaFree(d_ys);
}

// --- GPU full-domain eval benchmarks (ChaCha<4>) ---

template <int in_bits, typename Group>
static void BM_DcfEvalAllGpu(benchmark::State &state) {
  using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
  GpuData data;

  // See BM_HalfTreeDpfEvalAllGpu: process several keys per launch so the grid
  // fills the device and tail waves / launch overhead amortize.
  constexpr int kKeysPerLaunch = 1024;

  typename DcfType::Cw *d_cws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kNumKeys));

  DcfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());

  int4 *d_s0s;
  CUDA_CHECK(cudaMalloc(&d_s0s, sizeof(int4) * kNumKeys * 2));
  for (int b = 0; b < 2; ++b)
    CUDA_CHECK(cudaMemcpy2D(d_s0s + (size_t)b * kNumKeys, sizeof(int4), data.d_seeds + b, 2 * sizeof(int4),
        sizeof(int4), kNumKeys, cudaMemcpyDeviceToDevice));

  fss::prg::ChaCha<4> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  DcfType dcf{prg};

  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * (1 << in_bits) * kKeysPerLaunch));

  // Warm the clocks once, see BM_HalfTreeDpfEvalAllGpu.
  fss::gpu::DcfEvalAllGpuBatch<16, 8, 256>(false, d_s0s, d_cws, kKeysPerLaunch, d_ys, dcf);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    for (int off = 0; off < kNumKeys; off += kKeysPerLaunch)
      fss::gpu::DcfEvalAllGpuBatch<16, 8, 256>(
          false, d_s0s + off, d_cws + off * (in_bits + 1), kKeysPerLaunch, d_ys, dcf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys * (1 << in_bits));

  cudaFree(d_cws);
  cudaFree(d_s0s);
  cudaFree(d_ys);
}

// --- PackedHalfTree DPF full-domain GPU benchmarks (ChaCha<1>) ---

template <int in_bits, int out_bits>
__global__ void PackedHalfTreeDpfGenKernel(
    typename fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint>::Cw *cws, int4 *fcws,
    const int4 *seeds, const uint *alphas, const uint64_t *betas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumKeys) return;

  fss::prg::ChaCha<1> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint> dpf{prg, kHalfTreeHashKeyHost};

  using Scheme = fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint>;
  typename Scheme::Key k0, k1;
  int4 s0s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s), alphas[tid], betas[tid]);
  for (int i = 0; i < Scheme::kDepth; ++i) cws[tid * Scheme::kDepth + i] = k0.cws[i];
  fcws[tid] = k0.fcw;
}

template <int in_bits, int out_bits>
static void BM_PackedHalfTreeDpfEvalAllGpu(benchmark::State &state) {
  using PackedType = fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint>;
  constexpr int kDepth = PackedType::kDepth;
  // bs=256 frontier tiling: Z=min(kDepth,16), so 2^(Z-B1)=bs gives B1=Z-8.
  constexpr int Z = kDepth <= 16 ? kDepth : 16;
  constexpr int B1 = Z - 8;
  GpuData data;

  // See BM_HalfTreeDpfEvalAllGpu: process several keys per launch so the grid
  // fills the device and tail waves / launch overhead amortize.
  constexpr int kKeysPerLaunch = 1024;

  // One-bit betas like the CPU packed bench: the kLanes packing gain is the
  // point of the scheme.
  std::vector<uint64_t> betas(kNumKeys, 1);
  uint64_t *d_betas_u64;
  CUDA_CHECK(cudaMalloc(&d_betas_u64, sizeof(uint64_t) * kNumKeys));
  CUDA_CHECK(cudaMemcpy(d_betas_u64, betas.data(), sizeof(uint64_t) * kNumKeys, cudaMemcpyHostToDevice));

  typename PackedType::Cw *d_cws;
  int4 *d_fcws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename PackedType::Cw) * kDepth * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_fcws, sizeof(int4) * kNumKeys));

  PackedHalfTreeDpfGenKernel<in_bits, out_bits>
      <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_fcws, data.d_seeds, data.d_alphas, d_betas_u64);
  CUDA_CHECK(cudaDeviceSynchronize());

  int4 *d_s0s;
  CUDA_CHECK(cudaMalloc(&d_s0s, sizeof(int4) * kNumKeys * 2));
  for (int b = 0; b < 2; ++b)
    CUDA_CHECK(cudaMemcpy2D(d_s0s + (size_t)b * kNumKeys, sizeof(int4), data.d_seeds + b, 2 * sizeof(int4),
        sizeof(int4), kNumKeys, cudaMemcpyDeviceToDevice));

  fss::prg::ChaCha<1> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  PackedType dpf{prg, kHalfTreeHashKeyHost};

  int4 *d_ys;
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * (1 << kDepth) * kKeysPerLaunch));

  // Warm the clocks once, see BM_HalfTreeDpfEvalAllGpu.
  fss::gpu::PackedHalfTreeDpfEvalAllGpuBatch<Z, B1, 256>(false, d_s0s, d_cws, d_fcws, kKeysPerLaunch, d_ys, dpf);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    for (int off = 0; off < kNumKeys; off += kKeysPerLaunch)
      fss::gpu::PackedHalfTreeDpfEvalAllGpuBatch<Z, B1, 256>(
          false, d_s0s + off, d_cws + off * kDepth, d_fcws + off, kKeysPerLaunch, d_ys, dpf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys * (1 << in_bits));

  cudaFree(d_betas_u64);
  cudaFree(d_cws);
  cudaFree(d_fcws);
  cudaFree(d_s0s);
  cudaFree(d_ys);
}

// --- VDPF full-domain GPU benchmark (ChaCha<2> + Blake3) ---
//
// One key per iteration, matching the CPU VDPF EvalAll benches. The timed op
// is the complete VdpfEvalAllMerkleGpu: the frontier kernel computes the
// shares and folds each thread's leaves into a Merkle subtree, and a second
// kernel folds the block roots into the 64B proof. The paper's sequential
// proof chain is replaced by the order-independent fold analyzed in
// doc/vdpf_merkle_fold_security.md; no per-point data leaves the device.

template <int in_bits, typename Group>
static void BM_VdpfEvalAllMerkleGpu(benchmark::State &state) {
  using Vdpf = VdpfType<in_bits, Group>;
  constexpr size_t n = size_t{1} << in_bits;

  // Blake3 copies the IV by value at construction.
  const int4 iv[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444444},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)},
  };
  fss::hash::Blake3 blake3{cuda::std::span<const int4, 2>(iv, 2)};
  Vdpf vdpf{fss::prg::ChaCha<2>(0x12345678, static_cast<int>(0x9abcdef0u)), blake3, blake3};

  int4 s0s[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
  };
  uint alpha = 42;
  int4 beta = {7, 0, 0, 0};
  typename Vdpf::Key k0, k1;
  if (vdpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta) != 0) {
    fprintf(stderr, "vdpf gen retry condition hit for fixed seeds\n");
    exit(1);
  }

  typename Vdpf::Cw *d_cws;
  int4 *d_ys;
  cuda::std::array<int4, 4> *d_pi;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(k0.cws)));
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * n));
  CUDA_CHECK(cudaMalloc(&d_pi, sizeof(cuda::std::array<int4, 4>)));
  CUDA_CHECK(cudaMemcpy(d_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));

  cuda::std::array<int4, 4> pi;
  // Warm the clocks once, see BM_HalfTreeDpfEvalAllGpu.
  fss::gpu::VdpfEvalAllMerkleGpu<16, 8, 256>(false, k0.s0, d_cws, k0.cs, k0.ocw, d_ys, d_pi, vdpf);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&pi, d_pi, sizeof(pi), cudaMemcpyDeviceToHost));

  RunTimedKernel(state, [&] {
    fss::gpu::VdpfEvalAllMerkleGpu<16, 8, 256>(false, k0.s0, d_cws, k0.cs, k0.ocw, d_ys, d_pi, vdpf);
  });
  state.SetItemsProcessed(state.iterations() * n);
  benchmark::DoNotOptimize(pi.data());

  cudaFree(d_cws);
  cudaFree(d_ys);
  cudaFree(d_pi);
}

// --- GPU point eval benchmarks (level-major layout) ---
//
// Same workload as the per-key Eval benchmarks above (one output per key at
// one random point each), but the correction words are relaid level-major
// once after Gen: level i of every key adjacent, plus the packed control bits
// and the output correction words. Each level load is then one coalesced 128B
// transaction per warp instead of 32 lines of 128B each, which removes the
// DRAM bound for DPF and DCF point evaluation. The relayout is a one-time
// preprocessing cost outside the timed loop, same as Gen.

template <int in_bits, typename Group>
static void BM_DpfEvalPointGpu(benchmark::State &state) {
  using DpfType = fss::Dpf<in_bits, Group, fss::prg::ChaCha<2>, uint>;
  GpuData data;

  typename DpfType::Cw *d_cws;
  int4 *d_cw_s, *d_out_cw;
  uint32_t *d_extra;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DpfType::Cw) * (in_bits + 1) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cw_s, sizeof(int4) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_extra, sizeof(uint32_t) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_out_cw, sizeof(int4) * kNumKeys));

  DpfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());
  fss::gpu::DpfRelayoutGpu<in_bits, Group, fss::prg::ChaCha<2>, uint>(d_cws, kNumKeys, d_cw_s, d_extra, d_out_cw);
  CUDA_CHECK(cudaDeviceSynchronize());

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  DpfType dpf{prg};

  RunTimedKernel(state, [&] {
    fss::gpu::DpfEvalPointGpu<1, in_bits, Group, fss::prg::ChaCha<2>, uint>(
        false, data.d_seeds0, d_cw_s, d_extra, d_out_cw, data.d_xs, data.d_ys, kNumKeys, dpf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
  cudaFree(d_cw_s);
  cudaFree(d_extra);
  cudaFree(d_out_cw);
}

template <int in_bits, typename Group>
static void BM_DcfEvalPointGpu(benchmark::State &state) {
  using DcfType = fss::Dcf<in_bits, Group, fss::prg::ChaCha<4>, uint>;
  GpuData data;

  typename DcfType::Cw *d_cws;
  int4 *d_cw_s, *d_cw_v, *d_out_cw;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename DcfType::Cw) * (in_bits + 1) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cw_s, sizeof(int4) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cw_v, sizeof(int4) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_out_cw, sizeof(int4) * kNumKeys));

  DcfGenKernel<in_bits, Group><<<kNumBlocks, kThreadsPerBlock>>>(d_cws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());
  fss::gpu::DcfRelayoutGpu<in_bits, Group, fss::prg::ChaCha<4>, uint>(d_cws, kNumKeys, d_cw_s, d_cw_v, d_out_cw);
  CUDA_CHECK(cudaDeviceSynchronize());

  fss::prg::ChaCha<4> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  DcfType dcf{prg};

  RunTimedKernel(state, [&] {
    fss::gpu::DcfEvalPointGpu<4, in_bits, Group, fss::prg::ChaCha<4>, uint>(
        false, data.d_seeds0, d_cw_s, d_cw_v, d_out_cw, data.d_xs, data.d_ys, kNumKeys, dcf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
  cudaFree(d_cw_s);
  cudaFree(d_cw_v);
  cudaFree(d_out_cw);
}

template <int in_bits, typename Group>
static void BM_HalfTreeDpfEvalPointGpu(benchmark::State &state) {
  using HtDpf = HtDpfType<in_bits, Group>;
  GpuData data;

  typename HtDpf::Cw *d_cws;
  int4 *d_cw_s;
  uint32_t *d_extra;
  int4 *d_ocws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename HtDpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cw_s, sizeof(int4) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_extra, sizeof(uint32_t) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  HalfTreeDpfGenKernel<in_bits, Group>
      <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());
  fss::gpu::HalfTreeDpfRelayoutGpu<in_bits, Group, fss::prg::ChaCha<1>, uint>(d_cws, kNumKeys, d_cw_s, d_extra);
  CUDA_CHECK(cudaDeviceSynchronize());

  fss::prg::ChaCha<1> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  HtDpfType<in_bits, Group> dpf{prg, kHalfTreeHashKeyHost};

  RunTimedKernel(state, [&] {
    fss::gpu::HalfTreeDpfEvalPointGpu<4, in_bits, Group, fss::prg::ChaCha<1>, uint>(
        false, data.d_seeds0, d_cw_s, d_extra, d_ocws, data.d_xs, data.d_ys, kNumKeys, dpf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
  cudaFree(d_cw_s);
  cudaFree(d_extra);
  cudaFree(d_ocws);
}

template <int in_bits, typename Group>
static void BM_VdpfEvalPointGpu(benchmark::State &state) {
  using Vdpf = VdpfType<in_bits, Group>;
  GpuData data;

  typename Vdpf::Cw *d_cws;
  int4 *d_cw_s, *d_pi;
  uint32_t *d_extra;
  cuda::std::array<int4, 4> *d_cs;
  int4 *d_ocws;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename Vdpf::Cw) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_cw_s, sizeof(int4) * in_bits * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_extra, sizeof(uint32_t) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_pi, sizeof(int4) * kNumKeys * 4));
  CUDA_CHECK(cudaMalloc(&d_cs, sizeof(cuda::std::array<int4, 4>) * kNumKeys));
  CUDA_CHECK(cudaMalloc(&d_ocws, sizeof(int4) * kNumKeys));

  VdpfGenKernel<in_bits, Group>
      <<<kNumBlocks, kThreadsPerBlock>>>(d_cws, d_cs, d_ocws, data.d_seeds, data.d_alphas, data.d_betas);
  CUDA_CHECK(cudaDeviceSynchronize());
  fss::gpu::VdpfRelayoutGpu<in_bits, Group, fss::prg::ChaCha<2>, fss::hash::Blake3, fss::hash::Blake3, uint>(
      d_cws, kNumKeys, d_cw_s, d_extra);
  CUDA_CHECK(cudaDeviceSynchronize());

  fss::prg::ChaCha<2> prg(0x12345678, static_cast<int>(0x9abcdef0u));
  fss::hash::Blake3 xor_hash{cuda::std::span<const int4, 2>(kBlake3Iv)};
  Vdpf vdpf{prg, xor_hash, xor_hash};

  RunTimedKernel(state, [&] {
    fss::gpu::VdpfEvalPointGpu<1, in_bits, Group, fss::prg::ChaCha<2>, fss::hash::Blake3, fss::hash::Blake3, uint>(
        false, data.d_seeds0, d_cw_s, d_extra, d_cs, d_ocws, data.d_xs, data.d_ys, d_pi, kNumKeys, vdpf);
  });
  state.SetItemsProcessed(state.iterations() * kNumKeys);

  cudaFree(d_cws);
  cudaFree(d_cw_s);
  cudaFree(d_extra);
  cudaFree(d_pi);
  cudaFree(d_cs);
  cudaFree(d_ocws);
}

// --- Grotto DCF full-domain GPU benchmarks (ChaCha<2>) ---

template <int in_bits>
static void BM_GrottoDcfPreprocessGpu(benchmark::State &state) {
  using Prg = fss::prg::ChaCha<2>;
  using GdcfType = fss::GrottoDcf<in_bits, Prg, uint>;
  constexpr size_t N = size_t{1} << in_bits;

  GdcfType dcf{fss::prg::ChaCha<2>(0x12345678, static_cast<int>(0x9abcdef0u))};

  int4 s0s[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
  };
  uint alpha = 42;
  typename GdcfType::Key k0, k1;
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s), alpha);

  typename GdcfType::Cw *d_cws;
  bool *d_p;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(k0.cws)));
  CUDA_CHECK(cudaMalloc(&d_p, sizeof(bool) * (2 * N - 1)));
  CUDA_CHECK(cudaMemcpy(d_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));

  // Warm the clocks once, see BM_HalfTreeDpfEvalAllGpu.
  fss::gpu::GrottoDcfPreprocessGpu<16, 8, 256>(false, k0.s0, d_cws, d_p, dcf);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    fss::gpu::GrottoDcfPreprocessGpu<16, 8, 256>(false, k0.s0, d_cws, d_p, dcf);
  });
  state.SetItemsProcessed(state.iterations() * N);

  cudaFree(d_cws);
  cudaFree(d_p);
}

template <int in_bits>
static void BM_GrottoDcfEvalAllGpu(benchmark::State &state) {
  using Prg = fss::prg::ChaCha<2>;
  using GdcfType = fss::GrottoDcf<in_bits, Prg, uint>;
  constexpr size_t N = size_t{1} << in_bits;

  GdcfType dcf{fss::prg::ChaCha<2>(0x12345678, static_cast<int>(0x9abcdef0u))};

  int4 s0s[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
  };
  uint alpha = 42;
  typename GdcfType::Key k0, k1;
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s), alpha);

  typename GdcfType::Cw *d_cws;
  bool *d_ys;
  CUDA_CHECK(cudaMalloc(&d_cws, sizeof(k0.cws)));
  CUDA_CHECK(cudaMalloc(&d_ys, sizeof(bool) * N));
  CUDA_CHECK(cudaMemcpy(d_cws, k0.cws, sizeof(k0.cws), cudaMemcpyHostToDevice));

  // Warm the clocks once, see BM_HalfTreeDpfEvalAllGpu.
  fss::gpu::GrottoDcfEvalAllGpu<16, 8, 256>(false, k0.s0, d_cws, d_ys, dcf);
  CUDA_CHECK(cudaDeviceSynchronize());

  RunTimedKernel(state, [&] {
    fss::gpu::GrottoDcfEvalAllGpu<16, 8, 256>(false, k0.s0, d_cws, d_ys, dcf);
  });
  state.SetItemsProcessed(state.iterations() * N);

  cudaFree(d_cws);
  cudaFree(d_ys);
}

// --- Register all 18 benchmarks ---

// DPF (ChaCha<2>)
BENCHMARK(BM_DpfEval<20, UintGroup>)->Name("BM_DpfEval_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_DpfEval<14, UintGroup>)->Name("BM_DpfEval_Uint_ChaCha/14")->UseManualTime();
BENCHMARK(BM_DpfEval<17, UintGroup>)->Name("BM_DpfEval_Uint_ChaCha/17")->UseManualTime();
BENCHMARK(BM_DpfGen<20, UintGroup>)->Name("BM_DpfGen_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_DpfEval<20, BytesGroup>)->Name("BM_DpfEval_Bytes_ChaCha/20")->UseManualTime();

// DPF other PRG
BENCHMARK(BM_DpfEvalAes<20, UintGroup>)->Name("BM_DpfEval_Uint_AesSoft/20")->UseManualTime();

// DCF (ChaCha<4>)
BENCHMARK(BM_DcfEval<20, UintGroup>)->Name("BM_DcfEval_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_DcfGen<20, UintGroup>)->Name("BM_DcfGen_Uint_ChaCha/20")->UseManualTime();

// VDPF (ChaCha<2> + Blake3)
BENCHMARK(BM_VdpfEval<20, UintGroup>)->Name("BM_VdpfEval_Uint_ChaCha_Blake3/20")->UseManualTime();
BENCHMARK(BM_VdpfGen<20, UintGroup>)->Name("BM_VdpfGen_Uint_ChaCha_Blake3/20")->UseManualTime();

// HalfTreeDpf (ChaCha<1>)
BENCHMARK(BM_HalfTreeDpfEval<20, UintGroup>)->Name("BM_HalfTreeDpfEval_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_HalfTreeDpfGen<20, UintGroup>)->Name("BM_HalfTreeDpfGen_Uint_ChaCha/20")->UseManualTime();

// GPU full-domain eval
BENCHMARK(BM_HalfTreeDpfEvalAllGpu<20, UintGroup>)->Name("BM_HalfTreeDpfEvalAllGpu_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_DpfEvalAllGpu<20, UintGroup>)->Name("BM_DpfEvalAllGpu_Uint_ChaCha/20")->UseManualTime();

// GPU point eval (level-major layout)
BENCHMARK(BM_DpfEvalPointGpu<20, UintGroup>)->Name("BM_DpfEvalPointGpu_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_DcfEvalPointGpu<20, UintGroup>)->Name("BM_DcfEvalPointGpu_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_HalfTreeDpfEvalPointGpu<20, UintGroup>)
    ->Name("BM_HalfTreeDpfEvalPointGpu_Uint_ChaCha/20")
    ->UseManualTime();
BENCHMARK(BM_VdpfEvalPointGpu<20, UintGroup>)->Name("BM_VdpfEvalPointGpu_Uint_ChaCha_Blake3/20")->UseManualTime();

// GPU full-domain eval (Grotto DCF)
BENCHMARK(BM_GrottoDcfPreprocessGpu<20>)->Name("BM_GrottoDcfPreprocessGpu_ChaCha/20")->UseManualTime();
BENCHMARK(BM_GrottoDcfEvalAllGpu<20>)->Name("BM_GrottoDcfEvalAllGpu_ChaCha/20")->UseManualTime();
BENCHMARK(BM_DcfEvalAllGpu<20, UintGroup>)->Name("BM_DcfEvalAllGpu_Uint_ChaCha/20")->UseManualTime();
BENCHMARK(BM_PackedHalfTreeDpfEvalAllGpu<20, 1>)->Name("BM_PackedHalfTreeDpfEvalAllGpu_ChaCha/20")->UseManualTime();
BENCHMARK(BM_VdpfEvalAllMerkleGpu<20, UintGroup>)->Name("BM_VdpfEvalAllMerkleGpu_Uint_ChaCha_Blake3/20")->UseManualTime();
