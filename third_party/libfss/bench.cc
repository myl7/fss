#include <benchmark/benchmark.h>
#include "fss-client.h"
#include "fss-server.h"

#include <cstdlib>
#include <cstdio>

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
static_assert(FSS_BENCH_DOMAIN_BITS >= 8 && FSS_BENCH_DOMAIN_BITS <= 20,
              "domain bits must be between 8 and 20");

constexpr uint32_t kLogDomainSize = FSS_BENCH_DOMAIN_BITS;
constexpr uint32_t kNumParties = 2;
constexpr uint64_t kAlpha = 12345 & ((1u << kLogDomainSize) - 1);
constexpr uint64_t kBeta = 7;

static void FreeKeys(ServerKeyEq& k0, ServerKeyEq& k1) {
  for (int i = 0; i < 2; i++) {
    free(k0.cw[i]);
    free(k1.cw[i]);
  }
}

static void CheckCorrectness() {
  static bool checked = false;
  if (checked) return;
  Fss client, server;
  initializeClient(&client, kLogDomainSize, kNumParties);
  initializeServer(&server, &client);
  ServerKeyEq first, second;
  generateTreeEq(&client, &first, &second, kAlpha, kBeta);
  for (uint64_t x : {kAlpha, kAlpha ^ 1, uint64_t{0},
                     (uint64_t{1} << kLogDomainSize) - 1}) {
    mpz_class result = evaluateEq(&server, &first, x) -
                       evaluateEq(&server, &second, x);
    result = (result % server.prime + server.prime) % server.prime;
    if (result != (x == kAlpha ? kBeta : 0)) {
      fprintf(stderr, "share reconstruction failed\n");
      abort();
    }
  }
  FreeKeys(first, second);
  checked = true;
}

static void BM_Gen(benchmark::State& state) {
  CheckCorrectness();
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
  CheckCorrectness();
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
  CheckCorrectness();
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

static void FreeKeys(ServerKeyLt& k0, ServerKeyLt& k1) {
  for (int i = 0; i < 2; ++i) {
    free(k0.cw[i]);
    free(k1.cw[i]);
  }
}

static void CheckComparisonCorrectness() {
  static bool checked = false;
  if (checked) return;
  Fss client, server;
  initializeClient(&client, kLogDomainSize, kNumParties);
  initializeServer(&server, &client);
  ServerKeyLt first, second;
  generateTreeLt(&client, &first, &second, kAlpha, kBeta);
  for (uint64_t x : {kAlpha, kAlpha - 1, kAlpha + 1, uint64_t{0},
                     (uint64_t{1} << kLogDomainSize) - 1}) {
    const uint64_t result = evaluateLt(&server, &first, x) -
                            evaluateLt(&server, &second, x);
    if (result != (x < kAlpha ? kBeta : 0)) {
      fprintf(stderr, "comparison share reconstruction failed\n");
      abort();
    }
  }
  FreeKeys(first, second);
  checked = true;
}

static void BM_DcfGen(benchmark::State& state) {
  CheckComparisonCorrectness();
  Fss client;
  initializeClient(&client, kLogDomainSize, kNumParties);
  ServerKeyLt first, second;
  for (auto _ : state) {
    generateTreeLt(&client, &first, &second, kAlpha, kBeta);
    benchmark::DoNotOptimize(&first);
    benchmark::DoNotOptimize(&second);
    FreeKeys(first, second);
  }
}
BENCHMARK(BM_DcfGen)->Name("libfss/CPU/DCF/Gen");

static void BM_DcfEval(benchmark::State& state) {
  CheckComparisonCorrectness();
  Fss client, server;
  initializeClient(&client, kLogDomainSize, kNumParties);
  initializeServer(&server, &client);
  ServerKeyLt first, second;
  generateTreeLt(&client, &first, &second, kAlpha, kBeta);
  uint64_t x = 0;
  for (auto _ : state) {
    uint64_t result = evaluateLt(&server, &first, x);
    benchmark::DoNotOptimize(result);
    x = (x + 1) & ((uint64_t{1} << kLogDomainSize) - 1);
  }
  FreeKeys(first, second);
}
BENCHMARK(BM_DcfEval)->Name("libfss/CPU/DCF/Eval");
