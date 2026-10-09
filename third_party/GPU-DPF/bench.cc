#include <benchmark/benchmark.h>
#include "dpf_base/dpf.h"

#include <random>
#include <cstdio>
#include <cstdlib>

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
static_assert(FSS_BENCH_DOMAIN_BITS >= 8 && FSS_BENCH_DOMAIN_BITS <= 20,
              "domain bits must be between 8 and 20");

constexpr int kN = 1 << FSS_BENCH_DOMAIN_BITS;
constexpr int kAlpha = 12345 & (kN - 1);
constexpr uint128_t kBeta = 7;

static void CheckCorrectness() {
  static bool checked = false;
  if (checked) return;
  std::mt19937 gen(42);
  SeedsCodewords* tree =
      GenerateSeedsAndCodewordsLog(kAlpha, kBeta, kN, gen, AES128);
  SeedsCodewordsFlat first, second;
  FlattenCodewords(tree, 0, &first);
  FlattenCodewords(tree, 1, &second);
  for (int x : {kAlpha, kAlpha ^ 1, 0, kN - 1}) {
    const uint128_t result = EvaluateFlat(&first, x, AES128) -
                             EvaluateFlat(&second, x, AES128);
    if (result != (x == kAlpha ? kBeta : 0)) {
      fprintf(stderr, "share reconstruction failed\n");
      abort();
    }
  }
  FreeSeedsCodewords(tree);
  checked = true;
}

static void BM_Gen(benchmark::State& state) {
  CheckCorrectness();
  std::mt19937 gen(42);
  for (auto _ : state) {
    SeedsCodewords* s =
        GenerateSeedsAndCodewordsLog(kAlpha, kBeta, kN, gen, AES128);
    SeedsCodewordsFlat* sf = new SeedsCodewordsFlat;
    FlattenCodewords(s, 0, sf);
    benchmark::DoNotOptimize(sf);
    delete sf;
    FreeSeedsCodewords(s);
  }
}
BENCHMARK(BM_Gen)->Name("GPU-DPF/CPU/DPF/Gen");

static void BM_EvalAll(benchmark::State& state) {
  CheckCorrectness();
  std::mt19937 gen(42);
  SeedsCodewords* s =
      GenerateSeedsAndCodewordsLog(kAlpha, kBeta, kN, gen, AES128);
  SeedsCodewordsFlat* sf = new SeedsCodewordsFlat;
  FlattenCodewords(s, 0, sf);

  for (auto _ : state) {
    for (int i = 0; i < kN; i++) {
      uint128_t v = EvaluateFlat(sf, i, AES128);
      benchmark::DoNotOptimize(v);
    }
  }

  delete sf;
  FreeSeedsCodewords(s);
}
BENCHMARK(BM_EvalAll)->Name("GPU-DPF/CPU/DPF/EvalAll");

static void BM_Eval(benchmark::State& state) {
  CheckCorrectness();
  std::mt19937 gen(42);
  SeedsCodewords* s =
      GenerateSeedsAndCodewordsLog(kAlpha, kBeta, kN, gen, AES128);
  SeedsCodewordsFlat* sf = new SeedsCodewordsFlat;
  FlattenCodewords(s, 0, sf);

  int x = 0;
  for (auto _ : state) {
    uint128_t v = EvaluateFlat(sf, x, AES128);
    benchmark::DoNotOptimize(v);
    x = (x + 1) & (kN - 1);
  }

  delete sf;
  FreeSeedsCodewords(s);
}
BENCHMARK(BM_Eval)->Name("GPU-DPF/CPU/DPF/Eval");

