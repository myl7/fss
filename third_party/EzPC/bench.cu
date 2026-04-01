// Benchmark: EzPC/GPU-MPC DPF and DCF (GPU-only)
// DPF gen/eval/evalAll and DCF gen/eval with bin=20, AES-128 PRG.

#include <benchmark/benchmark.h>

// EzPC headers (order matters, do not reorder)
#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "fss/gpu_dpf.h"
#include "fss/dcf/gpu_dcf.h"
#include <sytorch/tensor.h>

// OneGB is declared extern in gpu_file_utils.h but defined in sigma_comms.cpp
// (network layer). We define it here to avoid pulling in networking code.
size_t OneGB = 1024ULL * 1024 * 1024;

using T = u64;

static constexpr int kBin = 20;
static constexpr int kBout = 1;
// batch size is now a benchmark parameter (state.range(0))

// Global state initialized once.
static AESGlobalContext *g_aes = nullptr;

static void EnsureInit() {
  if (g_aes) return;
  initGPUMemPool();
  g_aes = new AESGlobalContext;
  initAESContext(g_aes);
}

// ---------------------------------------------------------------------------
// Helpers from fss/gpu_lut.cu needed by dpfEvalAll kernel.
// Copied here to avoid pulling in the full gpu_lut.h dependency chain.
// ---------------------------------------------------------------------------

__device__ void storeAESBlock(AESBlock *x, int idx, AESBlock y,
                              int N, int threadId) {
  x[idx * N + threadId] = y;
}

__device__ AESBlock loadAESBlock(AESBlock *x, int idx,
                                 int N, int threadId) {
  return x[idx * N + threadId];
}

// ---------------------------------------------------------------------------
// gpuDpfEvalAll: copied from tests/fss/dpf_eval_all.cu because this function
// is not part of the EzPC library (only in the test binary).
// ---------------------------------------------------------------------------

template <typename TIn>
__global__ void dpfEvalAll(int party, int bin, int N, TIn *X,
                           AESBlock *scw_g, AESBlock *stack_g,
                           AESBlock *l0_g, AESBlock *l1_g, u32 *tR_g,
                           u32 *U, AESGlobalContext gaes) {
  AESSharedContext saes;
  loadSbox(&gaes, &saes);
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId < N) {
    storeAESBlock(stack_g, 0, scw_g[threadId], N, threadId);
    auto x = (u64)X[threadId];
    gpuMod(x, bin);
    auto l0_cw = l0_g[threadId];
    auto l1_cw = l1_g[threadId];
    auto tR = tR_g[threadId];
    u32 pathStack = 0;
    int depth = 1;
    T u = 0;
    while (depth > 0) {
      auto seed = loadAESBlock(stack_g, depth - 1, N, threadId);
      auto bit = pathStack & 1ULL;
      if (depth == bin - LOG_AES_BLOCK_LEN) {
        auto lastBlock = expandDPFTreeNode(
            bin, party, seed, 0, l0_cw, l1_cw, 0,
            uint8_t(bit), depth - 1, &saes);
        T c = party == SERVER1 ? -1 : 1;
        for (u64 i = 0; i < AES_BLOCK_LEN_IN_BITS; i++) {
          auto w = c * T(lastBlock & 1);
          u += w;
          lastBlock >>= 1;
        }
        while (pathStack & 1ULL) {
          pathStack >>= 1;
          depth--;
        }
        pathStack ^= 1;
      } else {
        auto tR_l = (tR >> (depth - 1)) & 1;
        auto newSeed = expandDPFTreeNode(
            bin, party, seed,
            loadAESBlock(scw_g, depth, N, threadId),
            0, 0, tR_l, uint8_t(bit), depth - 1, &saes);
        storeAESBlock(stack_g, depth, newSeed, N, threadId);
        depth++;
        pathStack <<= 1;
      }
    }
    gpuMod(u, 1);
    writeVCW(1, U, u64(u), 0, N);
  }
}

template <typename TIn>
u32 *gpuDpfEvalAll(GPUDPFKey k0, int party, TIn *d_X,
                   AESGlobalContext *g, Stats *s) {
  auto k = *(k0.dpfTreeKey);
  assert(k0.bin >= 8 && k0.B == 1);

  const int tbSz = 256;
  int tb = (k.N - 1) / tbSz + 1;
  AESBlock *d_scw, *d_stack, *d_l0, *d_l1;
  u32 *d_tR;

  assert(k.memSzScw % (k.bin - LOG_AES_BLOCK_LEN) == 0);

  d_scw = (AESBlock *)moveToGPU((uint8_t *)k.scw, k.memSzScw, s);
  d_stack = (AESBlock *)gpuMalloc(k.memSzScw);
  d_l0 = (AESBlock *)moveToGPU((uint8_t *)k.l0, k.memSzL, s);
  d_l1 = (AESBlock *)moveToGPU((uint8_t *)k.l1, k.memSzL, s);
  d_tR = (u32 *)moveToGPU((uint8_t *)k.tR, k.memSzT, s);
  auto d_U = (u32 *)gpuMalloc(k.memSzOut);

  dpfEvalAll<TIn><<<tb, tbSz>>>(party, k.bin, k.N, d_X, d_scw,
                                d_stack, d_l0, d_l1, d_tR, d_U, *g);
  checkCudaErrors(cudaDeviceSynchronize());

  gpuFree(d_scw);
  gpuFree(d_stack);
  gpuFree(d_l0);
  gpuFree(d_l1);
  gpuFree(d_tR);

  return d_U;
}

// ---------------------------------------------------------------------------
// DPF Benchmarks
// ---------------------------------------------------------------------------

// --- DPF Gen ---

static void BM_DpfGen(benchmark::State &state) {
  const int N = state.range(0);
  EnsureInit();
  initGPURandomness();
  auto *d_rin = randomGEOnGpu<T>(N, kBin);

  for (auto _ : state) {
    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 2 * OneGB);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gpuKeyGenDPF(&curPtr, /*party=*/0, kBin, N, d_rin, g_aes,
                 /*evalAll=*/false);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cpuFree(startPtr);
  }
  state.SetItemsProcessed(state.iterations() * N);
  gpuFree(d_rin);
  destroyGPURandomness();
}

// --- DPF Eval (point eval) ---

static void BM_DpfEval(benchmark::State &state) {
  const int N = state.range(0);
  EnsureInit();
  initGPURandomness();
  auto *d_rin = randomGEOnGpu<T>(N, kBin);
  auto *d_X = randomGEOnGpu<T>(N, kBin);

  // Generate keys for eval.
  u8 *startPtr, *curPtr;
  getKeyBuf(&startPtr, &curPtr, 2 * OneGB);
  gpuKeyGenDPF(&curPtr, 0, kBin, N, d_rin, g_aes, false);
  auto k = readGPUDPFKey(&startPtr);

  Stats s;
  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    auto *d_O = gpuDpf(k, 0, d_X, g_aes, &s);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuFree(d_O);
  }
  state.SetItemsProcessed(state.iterations() * N);
  gpuFree(d_rin);
  gpuFree(d_X);
  destroyGPURandomness();
}

// --- DPF EvalAll ---

static void BM_DpfEvalAll(benchmark::State &state) {
  const int N = state.range(0);
  EnsureInit();
  initGPURandomness();
  auto *d_rin = randomGEOnGpu<T>(N, kBin);

  // Generate keys with evalAll=true (different tR packing).
  u8 *startPtr, *curPtr;
  getKeyBuf(&startPtr, &curPtr, 2 * OneGB);
  gpuKeyGenDPF(&curPtr, 0, kBin, N, d_rin, g_aes, true);
  auto k = readGPUDPFKey(&startPtr);

  Stats s;
  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    auto *d_O = gpuDpfEvalAll(k, 0, d_rin, g_aes, &s);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuFree(d_O);
  }
  state.SetItemsProcessed(state.iterations() * N);
  gpuFree(d_rin);
  destroyGPURandomness();
}

// ---------------------------------------------------------------------------
// DCF Benchmarks
// ---------------------------------------------------------------------------

// --- DCF Gen ---

static void BM_DcfGen(benchmark::State &state) {
  const int N = state.range(0);
  EnsureInit();
  initGPURandomness();
  auto *d_rin = randomGEOnGpu<T>(N, kBin);

  for (auto _ : state) {
    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 2 * OneGB);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dcf::gpuKeyGenDCF(&curPtr, /*party=*/0, kBin, kBout, N, d_rin,
                      T(1), g_aes);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cpuFree(startPtr);
  }
  state.SetItemsProcessed(state.iterations() * N);
  gpuFree(d_rin);
  destroyGPURandomness();
}

// --- DCF Eval ---

static void BM_DcfEval(benchmark::State &state) {
  const int N = state.range(0);
  EnsureInit();
  initGPURandomness();
  auto *d_rin = randomGEOnGpu<T>(N, kBin);
  auto *d_X = randomGEOnGpu<T>(N, kBin);

  // Generate keys for eval.
  u8 *startPtr, *curPtr;
  getKeyBuf(&startPtr, &curPtr, 2 * OneGB);
  dcf::gpuKeyGenDCF(&curPtr, 0, kBin, kBout, N, d_rin, T(1), g_aes);
  auto k = dcf::readGPUDCFKey(&startPtr);

  Stats s;
  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    auto *d_O = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(
        k, 0, d_X, g_aes, &s);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuFree(d_O);
  }
  state.SetItemsProcessed(state.iterations() * N);
  gpuFree(d_rin);
  gpuFree(d_X);
  destroyGPURandomness();
}

static constexpr int kN = 1 << 18;  // batch size

BENCHMARK(BM_DpfGen)->Name("EzPC/GPU/DPF/Gen")->Arg(kN)->UseManualTime();
BENCHMARK(BM_DpfEval)->Name("EzPC/GPU/DPF/Eval")->Arg(kN)->UseManualTime();
BENCHMARK(BM_DpfEvalAll)->Name("EzPC/GPU/DPF/EvalAll")->Arg(kN)->UseManualTime();
BENCHMARK(BM_DcfGen)->Name("EzPC/GPU/DCF/Gen")->Arg(kN)->UseManualTime();
BENCHMARK(BM_DcfEval)->Name("EzPC/GPU/DCF/Eval")->Arg(kN)->UseManualTime();
