// Benchmark adapter for the Kales et al. dpf-cpp reference implementation
// (https://github.com/dkales/dpf-cpp), the DPF of "Revisiting User Privacy
// for Certificate Transparency" (EuroS&P 2019): an AES-NI MMO PRG with GF(2)
// output, whose full-domain EvalFull/EvalFull8 return packed-bit truth
// tables. Timing boundaries follow upstream bench.cpp: one Gen, one party's
// point Eval, or one full-domain Eval per iteration, with the returned
// table's allocation inside the timed region. Caveat carried from upstream:
// Gen seeds its PRNG from a fixed test block, so Gen measures the DPF
// expansion without any seed randomness.
#include <benchmark/benchmark.h>

#include "dpf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>

namespace {

constexpr size_t kAlpha = 42;
constexpr int kCheckBits = 16;

// Reconstruction must give 1 at alpha and 0 elsewhere, EvalFull8 must agree
// with EvalFull, and the XOR of the parties' packed tables must hold exactly
// the alpha bit (the tables pack point x into byte x/8, bit x%8). Runs once
// per process, before any timing.
void CheckCorrectness() {
  static bool checked = false;
  if (checked) {
    return;
  }
  const size_t logn = kCheckBits;
  const std::pair<std::vector<uint8_t>, std::vector<uint8_t>> keys =
      DPF::Gen(kAlpha, logn);
  const size_t points[] = {0, kAlpha - 1, kAlpha, kAlpha + 1,
                           (size_t{1} << logn) - 1};
  for (const size_t x : points) {
    const bool shared =
        DPF::Eval(keys.first, x, logn) ^ DPF::Eval(keys.second, x, logn);
    if (shared != (x == kAlpha)) {
      fprintf(stderr, "dpf-cpp reconstruction mismatch at input %zu\n", x);
      exit(EXIT_FAILURE);
    }
  }
  const std::vector<uint8_t> table0 = DPF::EvalFull(keys.first, logn);
  const std::vector<uint8_t> table1 = DPF::EvalFull(keys.second, logn);
  const std::vector<uint8_t> table8 = DPF::EvalFull8(keys.first, logn);
  if (table8 != table0) {
    fprintf(stderr, "dpf-cpp EvalFull8 disagrees with EvalFull\n");
    exit(EXIT_FAILURE);
  }
  size_t set_bits = 0;
  size_t bit_position = 0;
  for (size_t i = 0; i < table0.size(); ++i) {
    const uint8_t xor_byte = table0[i] ^ table1[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (xor_byte & (uint8_t{1} << bit)) {
        ++set_bits;
        bit_position = 8 * i + bit;
      }
    }
  }
  if (set_bits != 1 || bit_position != kAlpha) {
    fprintf(stderr, "dpf-cpp full-domain reconstruction mismatch\n");
    exit(EXIT_FAILURE);
  }
  checked = true;
}

void BM_DpfCppGen(benchmark::State &state) {
  CheckCorrectness();
  const size_t logn = static_cast<size_t>(state.range(0));
  for (auto _ : state) {
    const std::pair<std::vector<uint8_t>, std::vector<uint8_t>> keys =
        DPF::Gen(kAlpha, logn);
    benchmark::DoNotOptimize(keys.first.data());
    benchmark::DoNotOptimize(keys.second.data());
  }
}
BENCHMARK(BM_DpfCppGen)->Name("dpf-cpp/CPU/Gen")->Arg(16)->Arg(20);

void BM_DpfCppEval(benchmark::State &state) {
  CheckCorrectness();
  const size_t logn = static_cast<size_t>(state.range(0));
  const std::pair<std::vector<uint8_t>, std::vector<uint8_t>> keys =
      DPF::Gen(kAlpha, logn);
  size_t x = 0;
  for (auto _ : state) {
    bool share = DPF::Eval(keys.first, x, logn);
    benchmark::DoNotOptimize(share);
    x = (x + 1) & ((size_t{1} << logn) - 1);
  }
}
BENCHMARK(BM_DpfCppEval)->Name("dpf-cpp/CPU/Eval")->Arg(16)->Arg(20);

void BM_DpfCppEvalAll(benchmark::State &state) {
  CheckCorrectness();
  const size_t logn = static_cast<size_t>(state.range(0));
  const std::pair<std::vector<uint8_t>, std::vector<uint8_t>> keys =
      DPF::Gen(kAlpha, logn);
  const size_t domain = size_t{1} << logn;
  for (auto _ : state) {
    const std::vector<uint8_t> table = DPF::EvalFull(keys.first, logn);
    benchmark::DoNotOptimize(table.data());
  }
  state.SetItemsProcessed(state.iterations() * domain);
}
BENCHMARK(BM_DpfCppEvalAll)->Name("dpf-cpp/CPU/EvalAll")
    ->Arg(16)
    ->Arg(18)
    ->Arg(20)
    ->Arg(24);

// 8-way batched full-domain evaluation; upstream requires logn >= 10 for the
// vectorized recursion, which every benchmarked size satisfies.
void BM_DpfCppEvalAll8(benchmark::State &state) {
  CheckCorrectness();
  const size_t logn = static_cast<size_t>(state.range(0));
  const std::pair<std::vector<uint8_t>, std::vector<uint8_t>> keys =
      DPF::Gen(kAlpha, logn);
  const size_t domain = size_t{1} << logn;
  for (auto _ : state) {
    const std::vector<uint8_t> table = DPF::EvalFull8(keys.first, logn);
    benchmark::DoNotOptimize(table.data());
  }
  state.SetItemsProcessed(state.iterations() * domain);
}
BENCHMARK(BM_DpfCppEvalAll8)->Name("dpf-cpp/CPU/EvalAll8")
    ->Arg(16)
    ->Arg(18)
    ->Arg(20)
    ->Arg(24);

}  // namespace
