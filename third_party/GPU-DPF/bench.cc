#include <benchmark/benchmark.h>
#include "dpf_base/dpf.h"

#include <random>

constexpr int kN = 1 << 20;
constexpr int kAlpha = 12345;
constexpr uint128_t kBeta = 7;

static void BM_Gen(benchmark::State& state) {
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

