// Benchmark: fss-v0.7.0 (C) DPF and DCF on GPU with Salsa20 PRG
// DPF/DCF gen/eval with in_bits=20, uint (u128_le) and bytes (XOR) groups.
//
// group_add/group_neg/group_zero are link-time pluggable symbols, so we build
// separate executables per scheme and group type.
// Salsa20 DPF uses BLOCK_NUM=1 and DCF uses BLOCK_NUM=2.

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include <dpf.h>
#include <dcf.h>
}

// prg_init is defined in salsa20.cu without extern "C" linkage.
extern void prg_init(const uint8_t *state, int state_len);

#include "check.h"

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
#ifndef FSS_BENCH_NUM_KEYS
#define FSS_BENCH_NUM_KEYS (1 << 20)
#endif
#ifndef FSS_BENCH_THREADS_PER_BLOCK
#define FSS_BENCH_THREADS_PER_BLOCK 256
#endif
static constexpr int kInBits = FSS_BENCH_DOMAIN_BITS;
static constexpr int kInBytes = (kInBits + 7) / 8;
static constexpr int kAlphaVal = 12345;

static constexpr int kThreadsPerBlock = FSS_BENCH_THREADS_PER_BLOCK;

// Number of independent gen/eval instances for Gen and Eval benchmarks.
static constexpr int kN = FSS_BENCH_NUM_KEYS;
static constexpr int kNumBlocks = (kN + kThreadsPerBlock - 1) / kThreadsPerBlock;

#define CUDA_CHECK(x)                                                      \
    do {                                                                    \
        cudaError_t err = (x);                                             \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "cuda error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                     \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// DPF kernels
// ---------------------------------------------------------------------------

#ifdef FSS070_BENCH_DPF
// --- DPF ---

__global__ void DpfGenKernel(uint8_t *cws_dev, uint8_t *cw_np1_dev,
                             const uint8_t *alpha_dev,
                             const uint8_t *beta_dev, uint8_t *sbuf_dev) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    uint8_t *sbuf = sbuf_dev + (size_t)tid * kLambda * 6;
    uint8_t *cws = cws_dev + (size_t)tid * kDpfCwLen * kInBits;
    uint8_t *cw_np1 = cw_np1_dev + (size_t)tid * kLambda;

    DpfKey k = {cws, cw_np1};
    PointFunc pf = {{(uint8_t *)alpha_dev + (size_t)tid * kInBytes, kInBits},
                    (uint8_t *)beta_dev + (size_t)tid * kLambda};
    dpf_gen(k, pf, sbuf);
}

__global__ void DpfEvalKernel(uint8_t *sbuf_dev, uint8_t b,
                              const uint8_t *cws_dev,
                              const uint8_t *cw_np1_dev,
                              const uint8_t *x_dev) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    uint8_t *sbuf = sbuf_dev + (size_t)tid * kLambda * 3;
    const uint8_t *cws = cws_dev + (size_t)tid * kDpfCwLen * kInBits;
    const uint8_t *cw_np1 = cw_np1_dev + (size_t)tid * kLambda;

    DpfKey k = {(uint8_t *)cws, (uint8_t *)cw_np1};
    Bits x_bits = {(uint8_t *)x_dev + (size_t)tid * kInBytes, kInBits};
    dpf_eval(sbuf, b, k, x_bits);
}


// ---------------------------------------------------------------------------
// DCF kernels
// ---------------------------------------------------------------------------

#else
// --- DCF ---

__global__ void DcfGenKernel(uint8_t *cws_dev, uint8_t *cw_np1_dev,
                             const uint8_t *alpha_dev,
                             const uint8_t *beta_dev, uint8_t *sbuf_dev) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    uint8_t *sbuf = sbuf_dev + (size_t)tid * kLambda * 10;
    uint8_t *cws = cws_dev + (size_t)tid * kDcfCwLen * kInBits;
    uint8_t *cw_np1 = cw_np1_dev + (size_t)tid * kLambda;

    DcfKey k = {cws, cw_np1};
    CmpFunc cf = {{(uint8_t *)alpha_dev + (size_t)tid * kInBytes, kInBits},
                  (uint8_t *)beta_dev + (size_t)tid * kLambda,
                  kLtAlpha};
    dcf_gen(k, cf, sbuf);
}

__global__ void DcfEvalKernel(uint8_t *sbuf_dev, uint8_t b,
                              const uint8_t *cws_dev,
                              const uint8_t *cw_np1_dev,
                              const uint8_t *x_dev) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    uint8_t *sbuf = sbuf_dev + (size_t)tid * kLambda * 6;
    const uint8_t *cws = cws_dev + (size_t)tid * kDcfCwLen * kInBits;
    const uint8_t *cw_np1 = cw_np1_dev + (size_t)tid * kLambda;

    DcfKey k = {(uint8_t *)cws, (uint8_t *)cw_np1};
    Bits x_bits = {(uint8_t *)x_dev + (size_t)tid * kInBytes, kInBits};
    dcf_eval(sbuf, b, k, x_bits);
}


// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

#endif

namespace {

constexpr int kCheckCount = 32;
constexpr int kCheckCwBytes = kCheckedCwLen * kInBits;
constexpr int kCheckCwStride = ((kCheckCwBytes + 32 + 15) / 16) * 16;
constexpr int kCheckGenStride = kCheckedGenBytes + 32;
constexpr int kCheckEvalStride = kCheckedEvalBytes + 32;
constexpr int kCheckLastStride = kLambda + 32;

__global__ void CorrectnessKernel(uint8_t *cws, uint8_t *last, uint8_t *gen,
                                   uint8_t *eval, uint8_t *outputs,
                                   uint8_t *alpha, uint8_t *query,
                                   uint8_t *beta) {
  const int tid = threadIdx.x;
  if (tid >= kCheckCount) return;
  uint8_t *scratch = gen + tid * kCheckGenStride + 16;
  uint8_t *eval_scratch = eval + tid * kCheckEvalStride + 16;
  CheckedKey key{cws + tid * kCheckCwStride + 16,
                 last + tid * kCheckLastStride + 16};
  for (int i = 0; i < 2 * kLambda; ++i) scratch[i] = i * 17 + tid + 3;
  scratch[15] &= 0x7f;
  scratch[31] &= 0x7f;
  CheckedGen(key, Bits{alpha + tid * kInBytes, kInBits},
             beta, scratch);
  for (int party = 0; party < 2; ++party) {
    for (int i = 0; i < kLambda; ++i) {
      eval_scratch[i] = (i + party * kLambda) * 17 + tid + 3;
    }
    eval_scratch[15] &= 0x7f;
    CheckedEval(eval_scratch, party, key,
                Bits{query + tid * kInBytes, kInBits});
    for (int i = 0; i < kLambda; ++i) {
      outputs[(tid * 2 + party) * kLambda + i] = eval_scratch[i];
    }
  }
}

void CheckCorrectnessGpu() {
  static bool checked = false;
  if (checked) return;
  std::vector<uint8_t> cws(kCheckCount * kCheckCwStride, 0xa5);
  std::vector<uint8_t> last(kCheckCount * kCheckLastStride, 0xa5);
  std::vector<uint8_t> gen(kCheckCount * kCheckGenStride, 0xa5);
  std::vector<uint8_t> eval(kCheckCount * kCheckEvalStride, 0xa5);
  std::vector<uint8_t> outputs(kCheckCount * 2 * kLambda);
  std::vector<uint8_t> alpha(kCheckCount * kInBytes);
  std::vector<uint8_t> query(kCheckCount * kInBytes);
  std::vector<uint8_t> beta(kLambda, 0xff);
  beta[15] &= 0x7f;
  for (int tid = 0; tid < kCheckCount; ++tid) {
    for (int i = 0; i < kInBytes; ++i) {
      alpha[tid * kInBytes + i] = kCheckedAlpha[tid % 10] >> (8 * i);
      query[tid * kInBytes + i] = kCheckedQuery[tid % 10] >> (8 * i);
    }
  }
  auto upload = [](const std::vector<uint8_t> &host) {
    uint8_t *device;
    CUDA_CHECK(cudaMalloc(&device, host.size()));
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
    return device;
  };
  auto *d_cws = upload(cws);
  auto *d_last = upload(last);
  auto *d_gen = upload(gen);
  auto *d_eval = upload(eval);
  auto *d_outputs = upload(outputs);
  auto *d_alpha = upload(alpha);
  auto *d_query = upload(query);
  auto *d_beta = upload(beta);
  CorrectnessKernel<<<1, kCheckCount>>>(d_cws, d_last, d_gen, d_eval, d_outputs,
                                        d_alpha, d_query, d_beta);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  auto download = [](std::vector<uint8_t> &host, uint8_t *device) {
    CUDA_CHECK(cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
  };
  download(cws, d_cws);
  download(last, d_last);
  download(gen, d_gen);
  download(eval, d_eval);
  download(outputs, d_outputs);
  CUDA_CHECK(cudaFree(d_alpha));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_beta));
  auto canary = [](const std::vector<uint8_t> &host, int stride, int payload) {
    for (int tid = 0; tid < kCheckCount; ++tid) {
      for (int i = 0; i < stride; ++i) {
        if ((i < 16 || i >= payload + 16) && host[tid * stride + i] != 0xa5) {
          fprintf(stderr, "historical GPU buffer canary changed at input %d\n", tid);
          exit(EXIT_FAILURE);
        }
      }
    }
  };
  canary(cws, kCheckCwStride, kCheckCwBytes);
  canary(last, kCheckLastStride, kLambda);
  canary(gen, kCheckGenStride, kCheckedGenBytes);
  canary(eval, kCheckEvalStride, kCheckedEvalBytes);
  for (int tid = 0; tid < kCheckCount; ++tid) {
    CheckShares(kCheckedAlpha[tid % 10], kCheckedQuery[tid % 10],
                outputs.data() + tid * 2 * kLambda,
                outputs.data() + (tid * 2 + 1) * kLambda, beta.data());
  }
  checked = true;
  fprintf(stderr, "historical GPU reconstruction, MSB, and adjacent canary checks passed\n");
}

void InitPrgGpu() {
    // Salsa20 prg_init copies nonces to __constant__ memory via
    // cudaMemcpyToSymbol. DPF uses BLOCK_NUM=1 (8 bytes nonce),
    // DCF uses BLOCK_NUM=2 (16 bytes nonce).
    uint8_t nonces[16] = {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                          0x0f, 0xed, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21};
    prg_init(nonces, FSS070_PRG_BLOCK_NUM * 8);
    CUDA_CHECK(cudaGetLastError());
    CheckCorrectnessGpu();
    if (getenv("FSS070_CHECK_ONLY")) exit(EXIT_SUCCESS);
}

// Fill host buffer with pseudo-random bytes using a simple LCG.
void FillRandom(uint8_t *buf, size_t len, uint32_t seed = 42) {
    for (size_t i = 0; i < len; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = static_cast<uint8_t>(seed >> 16);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// DPF benchmarks
// ---------------------------------------------------------------------------

#ifdef FSS070_BENCH_DPF
// --- DPF ---

static void BM_DpfGen(benchmark::State &state) {
    InitPrgGpu();

    // Allocate per-instance data: alpha, beta, s0s packed in sbuf.
    size_t alpha_total = (size_t)kN * kInBytes;
    size_t beta_total = (size_t)kN * kLambda;
    size_t sbuf_total = (size_t)kN * kLambda * 6;
    size_t cws_total = (size_t)kN * kDpfCwLen * kInBits;
    size_t cw_np1_total = (size_t)kN * kLambda;

    auto *h_alpha = new uint8_t[alpha_total];
    auto *h_beta = new uint8_t[beta_total];
    auto *h_sbuf = new uint8_t[sbuf_total];
    FillRandom(h_alpha, alpha_total, 1);
    FillRandom(h_beta, beta_total, 2);
    // s0s stored in first 2*kLambda of each sbuf chunk; clear control bits.
    FillRandom(h_sbuf, sbuf_total, 3);
    for (int i = 0; i < kN; i++) {
        h_sbuf[(size_t)i * kLambda * 6 + kLambda - 1] &= 0x7f;
        h_sbuf[(size_t)i * kLambda * 6 + 2 * kLambda - 1] &= 0x7f;
    }

    uint8_t *d_alpha, *d_beta, *d_sbuf, *d_cws, *d_cw_np1;
    CUDA_CHECK(cudaMalloc(&d_alpha, alpha_total));
    CUDA_CHECK(cudaMalloc(&d_beta, beta_total));
    CUDA_CHECK(cudaMalloc(&d_sbuf, sbuf_total));
    CUDA_CHECK(cudaMalloc(&d_cws, cws_total));
    CUDA_CHECK(cudaMalloc(&d_cw_np1, cw_np1_total));
    CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha, alpha_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, beta_total, cudaMemcpyHostToDevice));

    for (auto _ : state) {
        CUDA_CHECK(cudaMemcpy(d_sbuf, h_sbuf, sbuf_total, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DpfGenKernel<<<kNumBlocks, kThreadsPerBlock>>>(
            d_cws, d_cw_np1, d_alpha, d_beta, d_sbuf);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_sbuf);
    cudaFree(d_cws);
    cudaFree(d_cw_np1);
    delete[] h_alpha;
    delete[] h_beta;
    delete[] h_sbuf;
}
BENCHMARK(BM_DpfGen)->Name(BENCH_NAME_PREFIX "/DPF/Gen")->UseManualTime();

static void BM_DpfEval(benchmark::State &state) {
    InitPrgGpu();

    size_t alpha_total = (size_t)kN * kInBytes;
    size_t beta_total = (size_t)kN * kLambda;
    size_t gen_sbuf_total = (size_t)kN * kLambda * 6;
    size_t eval_sbuf_total = (size_t)kN * kLambda * 3;
    size_t cws_total = (size_t)kN * kDpfCwLen * kInBits;
    size_t cw_np1_total = (size_t)kN * kLambda;
    size_t x_total = (size_t)kN * kInBytes;

    auto *h_alpha = new uint8_t[alpha_total];
    auto *h_beta = new uint8_t[beta_total];
    auto *h_sbuf = new uint8_t[gen_sbuf_total];
    auto *h_eval_sbuf = new uint8_t[eval_sbuf_total];
    auto *h_x = new uint8_t[x_total];
    FillRandom(h_alpha, alpha_total, 1);
    FillRandom(h_beta, beta_total, 2);
    FillRandom(h_sbuf, gen_sbuf_total, 3);
    FillRandom(h_x, x_total, 4);
    for (int i = 0; i < kN; i++) {
        h_sbuf[(size_t)i * kLambda * 6 + kLambda - 1] &= 0x7f;
        h_sbuf[(size_t)i * kLambda * 6 + 2 * kLambda - 1] &= 0x7f;
        // Copy seed s0 into eval sbuf for each instance.
        for (int j = 0; j < kLambda; j++) {
            h_eval_sbuf[(size_t)i * kLambda * 3 + j] =
                h_sbuf[(size_t)i * kLambda * 6 + j];
        }
    }

    uint8_t *d_alpha, *d_beta, *d_sbuf, *d_cws, *d_cw_np1, *d_eval_sbuf, *d_x;
    CUDA_CHECK(cudaMalloc(&d_alpha, alpha_total));
    CUDA_CHECK(cudaMalloc(&d_beta, beta_total));
    CUDA_CHECK(cudaMalloc(&d_sbuf, gen_sbuf_total));
    CUDA_CHECK(cudaMalloc(&d_cws, cws_total));
    CUDA_CHECK(cudaMalloc(&d_cw_np1, cw_np1_total));
    CUDA_CHECK(cudaMalloc(&d_eval_sbuf, eval_sbuf_total));
    CUDA_CHECK(cudaMalloc(&d_x, x_total));
    CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha, alpha_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, beta_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sbuf, h_sbuf, gen_sbuf_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_total, cudaMemcpyHostToDevice));

    // Generate keys first.
    DpfGenKernel<<<kNumBlocks, kThreadsPerBlock>>>(
        d_cws, d_cw_np1, d_alpha, d_beta, d_sbuf);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        CUDA_CHECK(cudaMemcpy(d_eval_sbuf, h_eval_sbuf, eval_sbuf_total,
                              cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DpfEvalKernel<<<kNumBlocks, kThreadsPerBlock>>>(
            d_eval_sbuf, 0, d_cws, d_cw_np1, d_x);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_sbuf);
    cudaFree(d_cws);
    cudaFree(d_cw_np1);
    cudaFree(d_eval_sbuf);
    cudaFree(d_x);
    delete[] h_alpha;
    delete[] h_beta;
    delete[] h_sbuf;
    delete[] h_eval_sbuf;
    delete[] h_x;
}
BENCHMARK(BM_DpfEval)->Name(BENCH_NAME_PREFIX "/DPF/Eval")->UseManualTime();


// ---------------------------------------------------------------------------
// DCF benchmarks
// ---------------------------------------------------------------------------

#else
// --- DCF ---

static void BM_DcfGen(benchmark::State &state) {
    InitPrgGpu();

    size_t alpha_total = (size_t)kN * kInBytes;
    size_t beta_total = (size_t)kN * kLambda;
    size_t sbuf_total = (size_t)kN * kLambda * 10;
    size_t cws_total = (size_t)kN * kDcfCwLen * kInBits;
    size_t cw_np1_total = (size_t)kN * kLambda;

    auto *h_alpha = new uint8_t[alpha_total];
    auto *h_beta = new uint8_t[beta_total];
    auto *h_sbuf = new uint8_t[sbuf_total];
    FillRandom(h_alpha, alpha_total, 1);
    FillRandom(h_beta, beta_total, 2);
    FillRandom(h_sbuf, sbuf_total, 3);
    for (int i = 0; i < kN; i++) {
        h_sbuf[(size_t)i * kLambda * 10 + kLambda - 1] &= 0x7f;
        h_sbuf[(size_t)i * kLambda * 10 + 2 * kLambda - 1] &= 0x7f;
    }

    uint8_t *d_alpha, *d_beta, *d_sbuf, *d_cws, *d_cw_np1;
    CUDA_CHECK(cudaMalloc(&d_alpha, alpha_total));
    CUDA_CHECK(cudaMalloc(&d_beta, beta_total));
    CUDA_CHECK(cudaMalloc(&d_sbuf, sbuf_total));
    CUDA_CHECK(cudaMalloc(&d_cws, cws_total));
    CUDA_CHECK(cudaMalloc(&d_cw_np1, cw_np1_total));
    CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha, alpha_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, beta_total, cudaMemcpyHostToDevice));

    for (auto _ : state) {
        CUDA_CHECK(cudaMemcpy(d_sbuf, h_sbuf, sbuf_total, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DcfGenKernel<<<kNumBlocks, kThreadsPerBlock>>>(
            d_cws, d_cw_np1, d_alpha, d_beta, d_sbuf);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_sbuf);
    cudaFree(d_cws);
    cudaFree(d_cw_np1);
    delete[] h_alpha;
    delete[] h_beta;
    delete[] h_sbuf;
}
BENCHMARK(BM_DcfGen)->Name(BENCH_NAME_PREFIX "/DCF/Gen")->UseManualTime();

static void BM_DcfEval(benchmark::State &state) {
    InitPrgGpu();

    size_t alpha_total = (size_t)kN * kInBytes;
    size_t beta_total = (size_t)kN * kLambda;
    size_t gen_sbuf_total = (size_t)kN * kLambda * 10;
    size_t eval_sbuf_total = (size_t)kN * kLambda * 6;
    size_t cws_total = (size_t)kN * kDcfCwLen * kInBits;
    size_t cw_np1_total = (size_t)kN * kLambda;
    size_t x_total = (size_t)kN * kInBytes;

    auto *h_alpha = new uint8_t[alpha_total];
    auto *h_beta = new uint8_t[beta_total];
    auto *h_sbuf = new uint8_t[gen_sbuf_total];
    auto *h_eval_sbuf = new uint8_t[eval_sbuf_total];
    auto *h_x = new uint8_t[x_total];
    FillRandom(h_alpha, alpha_total, 1);
    FillRandom(h_beta, beta_total, 2);
    FillRandom(h_sbuf, gen_sbuf_total, 3);
    FillRandom(h_x, x_total, 4);
    for (int i = 0; i < kN; i++) {
        h_sbuf[(size_t)i * kLambda * 10 + kLambda - 1] &= 0x7f;
        h_sbuf[(size_t)i * kLambda * 10 + 2 * kLambda - 1] &= 0x7f;
        for (int j = 0; j < kLambda; j++) {
            h_eval_sbuf[(size_t)i * kLambda * 6 + j] =
                h_sbuf[(size_t)i * kLambda * 10 + j];
        }
    }

    uint8_t *d_alpha, *d_beta, *d_sbuf, *d_cws, *d_cw_np1, *d_eval_sbuf, *d_x;
    CUDA_CHECK(cudaMalloc(&d_alpha, alpha_total));
    CUDA_CHECK(cudaMalloc(&d_beta, beta_total));
    CUDA_CHECK(cudaMalloc(&d_sbuf, gen_sbuf_total));
    CUDA_CHECK(cudaMalloc(&d_cws, cws_total));
    CUDA_CHECK(cudaMalloc(&d_cw_np1, cw_np1_total));
    CUDA_CHECK(cudaMalloc(&d_eval_sbuf, eval_sbuf_total));
    CUDA_CHECK(cudaMalloc(&d_x, x_total));
    CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha, alpha_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, beta_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sbuf, h_sbuf, gen_sbuf_total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_total, cudaMemcpyHostToDevice));

    // Generate keys first.
    DcfGenKernel<<<kNumBlocks, kThreadsPerBlock>>>(
        d_cws, d_cw_np1, d_alpha, d_beta, d_sbuf);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        CUDA_CHECK(cudaMemcpy(d_eval_sbuf, h_eval_sbuf, eval_sbuf_total,
                              cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        DcfEvalKernel<<<kNumBlocks, kThreadsPerBlock>>>(
            d_eval_sbuf, 0, d_cws, d_cw_np1, d_x);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    state.SetItemsProcessed(state.iterations() * kN);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_sbuf);
    cudaFree(d_cws);
    cudaFree(d_cw_np1);
    cudaFree(d_eval_sbuf);
    cudaFree(d_x);
    delete[] h_alpha;
    delete[] h_beta;
    delete[] h_sbuf;
    delete[] h_eval_sbuf;
    delete[] h_x;
}
BENCHMARK(BM_DcfEval)->Name(BENCH_NAME_PREFIX "/DCF/Eval")->UseManualTime();


#endif
