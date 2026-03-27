#include <benchmark/benchmark.h>
#include "fss-client.h"
#include "fss-server.h"

#include <cstdlib>

constexpr uint32_t kLogDomainSize = 20;
constexpr uint32_t kNumParties = 2;
constexpr uint64_t kAlpha = 12345;
constexpr uint64_t kBeta = 7;

static void FreeKeys(ServerKeyEq& k0, ServerKeyEq& k1) {
  for (int i = 0; i < 2; i++) {
    free(k0.cw[i]);
    free(k1.cw[i]);
  }
}

static void BM_Gen(benchmark::State& state) {
  Fss fClient;
  initializeClient(&fClient, kLogDomainSize, kNumParties);

  ServerKeyEq k0, k1;
  for (auto _ : state) {
    generateTreeEq(&fClient, &k0, &k1, kAlpha, kBeta);
    benchmark::DoNotOptimize(&k0);
    benchmark::DoNotOptimize(&k1);
    FreeKeys(k0, k1);
  }
}
BENCHMARK(BM_Gen)->Name("libfss/CPU/DPF/Gen");

static void BM_EvalAll(benchmark::State& state) {
  Fss fClient;
  initializeClient(&fClient, kLogDomainSize, kNumParties);

  ServerKeyEq k0, k1;
  generateTreeEq(&fClient, &k0, &k1, kAlpha, kBeta);

  Fss fServer;
  initializeServer(&fServer, &fClient);

  const uint64_t n = 1u << kLogDomainSize;
  for (auto _ : state) {
    for (uint64_t x = 0; x < n; x++) {
      mpz_class result = evaluateEq(&fServer, &k0, x);
      benchmark::DoNotOptimize(result);
    }
  }

  FreeKeys(k0, k1);
}
BENCHMARK(BM_EvalAll)->Name("libfss/CPU/DPF/EvalAll");

static void BM_Eval(benchmark::State& state) {
  Fss fClient;
  initializeClient(&fClient, kLogDomainSize, kNumParties);

  ServerKeyEq k0, k1;
  generateTreeEq(&fClient, &k0, &k1, kAlpha, kBeta);

  Fss fServer;
  initializeServer(&fServer, &fClient);

  const uint64_t mask = (1u << kLogDomainSize) - 1;
  uint64_t x = 0;
  for (auto _ : state) {
    mpz_class result = evaluateEq(&fServer, &k0, x);
    benchmark::DoNotOptimize(result);
    x = (x + 1) & mask;
  }

  FreeKeys(k0, k1);
}
BENCHMARK(BM_Eval)->Name("libfss/CPU/DPF/Eval");
