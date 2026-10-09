#include <benchmark/benchmark.h>
#include "dpf_base/dpf.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define FUSES_MATMUL 0
#define MM 1
#include "dpf_gpu/dpf/dpf_hybrid.cu"

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
static_assert(FSS_BENCH_DOMAIN_BITS >= 8 && FSS_BENCH_DOMAIN_BITS <= 20,
              "domain bits must be between 8 and 20");

namespace {
constexpr int kBits = FSS_BENCH_DOMAIN_BITS;
constexpr int kDomain = 1 << kBits;
constexpr int kAlpha = 12345 & (kDomain - 1);
constexpr uint128_t kBeta = 7;

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cuda operation failed: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// The upstream helper uses free() on objects allocated with new.
void DeleteTree(SeedsCodewords* key) {
  if (key == nullptr) return;
  DeleteTree(key->sub);
  delete key;
}

void BM_EvalAllFull(benchmark::State& state) {
  std::mt19937 random(42);
  // The upstream CHACHA20 enum selects its chacha20_12 implementation.
  auto* tree = GenerateSeedsAndCodewordsLog(kAlpha, kBeta, kDomain, random, CHACHA);
  SeedsCodewordsFlat cpu[2]{};
  SeedsCodewordsFlatGPU host[2]{};
  for (int party = 0; party < 2; ++party) {
    FlattenCodewords(tree, party, &cpu[party]);
    host[party] = SeedsCodewordsFlatGPUFromCPU(cpu[party]);
  }
  DeleteTree(tree);
  SeedsCodewordsFlatGPU* key;
  uint128_t_gpu* output;
  Check(cudaMalloc(&key, sizeof(host[0])));
  Check(cudaMalloc(&output, kDomain * sizeof(*output)));
  const size_t stack_bytes = size_t(kBits) * Z * 2 * sizeof(uint128_t_gpu);
  Check(cudaMalloc(&DPF_HYBRID_STACK_1, stack_bytes));
  Check(cudaMalloc(&DPF_HYBRID_STACK_2, stack_bytes));
  std::vector<uint128_t_gpu> result[2];
  for (int party = 0; party < 2; ++party) {
    result[party].resize(kDomain);
    Check(cudaMemcpy(key, &host[party], sizeof(host[party]), cudaMemcpyHostToDevice));
    dpf_hybrid<CHACHA20>(key, output, nullptr, 1, kDomain, nullptr);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(result[party].data(), output, kDomain * sizeof(*output),
                     cudaMemcpyDeviceToHost));
  }
  // Check every reconstructed value, plus CPU share references at n=8 and
  // evenly spaced points for larger domains. Native output is bit reversed.
  for (int x = 0; x < kDomain; ++x) {
    const int index = brev_cpu(x) >> (32 - kBits);
    const uint128_t a = uint128_from_gpu(result[0][index]);
    const uint128_t b = uint128_from_gpu(result[1][index]);
    const bool reference = kBits == 8 || x == kAlpha ||
                           x % (kDomain / 256) == 0 || x == kDomain - 1;
    if (a - b != (x == kAlpha ? kBeta : 0) ||
        (reference && (a != EvaluateFlat(&cpu[0], x, CHACHA) ||
                       b != EvaluateFlat(&cpu[1], x, CHACHA)))) {
      std::fprintf(stderr, "full-domain correctness check failed at input %d\n", x);
      std::exit(EXIT_FAILURE);
    }
  }
  Check(cudaMemcpy(key, &host[0], sizeof(host[0]), cudaMemcpyHostToDevice));
  cudaEvent_t start, stop;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&stop));
  for (auto _ : state) {
    Check(cudaEventRecord(start));
    dpf_hybrid<CHACHA20>(key, output, nullptr, 1, kDomain, nullptr);
    Check(cudaGetLastError());
    Check(cudaEventRecord(stop));
    Check(cudaEventSynchronize(stop));
    float ms;
    Check(cudaEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }
  state.SetItemsProcessed(state.iterations() * kDomain);
  state.counters["domain_bits"] = kBits;
  state.counters["domain_outputs"] = kDomain;
  state.counters["keys"] = 1;
  state.counters["threads_per_block"] = DPF_HYBRID_THREADS_PER_BLOCK;
  state.counters["output_bits"] = 128;
  state.SetLabel("native hybrid; ChaCha12; bit-reversed uint128 output; kernel only");
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(stop));
  Check(cudaFree(key));
  Check(cudaFree(output));
  Check(cudaFree(DPF_HYBRID_STACK_1));
  Check(cudaFree(DPF_HYBRID_STACK_2));
}
BENCHMARK(BM_EvalAllFull)->Name("GPU-DPF/GPU/DPF/EvalAllFull")->UseManualTime();
}  // namespace
