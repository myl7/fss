// Benchmark: Google distributed_point_functions DPF and DCF
// DPF uses XOR uint128 output. DCF uses additive uint128 output.
//
// Build: cd third_party/distributed_point_functions && bazel build -c opt :bench_dpf_google
// Run:   bazel-bin/bench_dpf_google

#include <benchmark/benchmark.h>

#include "absl/numeric/int128.h"
#include "dcf/distributed_comparison_function.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/xor_wrapper.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace distributed_point_functions;

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
static_assert(FSS_BENCH_DOMAIN_BITS >= 8 && FSS_BENCH_DOMAIN_BITS <= 20,
              "domain bits must be between 8 and 20");

constexpr int kLogDomainSize = FSS_BENCH_DOMAIN_BITS;
constexpr absl::uint128 kAlpha = 12345 & ((1u << kLogDomainSize) - 1);

DpfParameters MakeDpfParams() {
    DpfParameters params;
    params.set_log_domain_size(kLogDomainSize);
    params.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(128);
    return params;
}

DcfParameters MakeDcfParams() {
    DcfParameters params;
    *params.mutable_parameters() = MakeDpfParams();
    // The native DCF zero-value conversion supports integer outputs.
    params.mutable_parameters()->mutable_value_type()->mutable_integer()->set_bitsize(128);
    return params;
}

using T = XorWrapper<absl::uint128>;

void CheckCorrectness() {
  static bool checked = false;
  if (checked) return;
  auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
  auto dcf = DistributedComparisonFunction::Create(MakeDcfParams()).value();
  auto dpf_keys = dpf->GenerateKeys(kAlpha, T{7}).value();
  auto dcf_keys = dcf->GenerateKeys(kAlpha, absl::uint128{7}).value();
  for (absl::uint128 x : {kAlpha, kAlpha - 1, kAlpha + 1,
                          absl::uint128{0},
                          (absl::uint128{1} << kLogDomainSize) - 1}) {
    auto points = absl::MakeConstSpan(&x, 1);
    T point = dpf->EvaluateAt<T>(dpf_keys.first, 0, points).value()[0] +
              dpf->EvaluateAt<T>(dpf_keys.second, 0, points).value()[0];
    absl::uint128 comparison =
        dcf->Evaluate<absl::uint128>(dcf_keys.first, x).value() +
        dcf->Evaluate<absl::uint128>(dcf_keys.second, x).value();
    if (point != T{x == kAlpha ? 7 : 0} ||
        comparison != absl::uint128{x < kAlpha ? 7 : 0}) {
      fprintf(stderr,
              "share reconstruction failed: query=%llu, dpf=%llx:%llx, "
              "dcf=%llx:%llx\n",
              static_cast<unsigned long long>(absl::Uint128Low64(x)),
              static_cast<unsigned long long>(absl::Uint128High64(point.value())),
              static_cast<unsigned long long>(absl::Uint128Low64(point.value())),
              static_cast<unsigned long long>(absl::Uint128High64(comparison)),
              static_cast<unsigned long long>(absl::Uint128Low64(comparison)));
      abort();
    }
  }
  checked = true;
}

// ---------------------------------------------------------------------------
// DPF benchmarks
// ---------------------------------------------------------------------------

void BM_DpfGen(benchmark::State& state) {
    CheckCorrectness();
    auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
    absl::uint128 alpha = kAlpha;
    T beta{7};

    for (auto _ : state) {
        auto keys = dpf->GenerateKeys(alpha, beta).value();
        benchmark::DoNotOptimize(keys);
    }
}
BENCHMARK(BM_DpfGen)->Name("google_dpf/CPU/DPF/Gen");

void BM_DpfEvalAll(benchmark::State& state) {
    CheckCorrectness();
    auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
    auto [key0, key1] = dpf->GenerateKeys(kAlpha, T{7}).value();

    for (auto _ : state) {
        auto ctx = dpf->CreateEvaluationContext(key0).value();
        auto result = dpf->EvaluateUntil<T>(
            0, absl::Span<const absl::uint128>(), ctx).value();
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_DpfEvalAll)->Name("google_dpf/CPU/DPF/EvalAll");

void BM_DpfEval(benchmark::State& state) {
    CheckCorrectness();
    auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
    auto [key0, key1] = dpf->GenerateKeys(kAlpha, T{7}).value();

    absl::uint128 x = 0;
    for (auto _ : state) {
        auto result = dpf->EvaluateAt<T>(key0, 0, absl::MakeConstSpan(&x, 1)).value();
        benchmark::DoNotOptimize(result[0]);
        x = (x + 1) & ((absl::uint128{1} << kLogDomainSize) - 1);
    }
}
BENCHMARK(BM_DpfEval)->Name("google_dpf/CPU/DPF/Eval");

// ---------------------------------------------------------------------------
// DCF benchmarks
// ---------------------------------------------------------------------------

void BM_DcfGen(benchmark::State& state) {
    CheckCorrectness();
    auto dcf = DistributedComparisonFunction::Create(MakeDcfParams()).value();
    absl::uint128 alpha = kAlpha;
    absl::uint128 beta{7};

    for (auto _ : state) {
        auto keys = dcf->GenerateKeys(alpha, beta).value();
        benchmark::DoNotOptimize(keys);
    }
}
BENCHMARK(BM_DcfGen)->Name("google_dpf/CPU/DCF/Gen");

void BM_DcfEval(benchmark::State& state) {
    CheckCorrectness();
    auto dcf = DistributedComparisonFunction::Create(MakeDcfParams()).value();
    auto [key0, key1] = dcf->GenerateKeys(kAlpha, absl::uint128{7}).value();

    absl::uint128 x = 0;
    for (auto _ : state) {
        auto result = dcf->Evaluate<absl::uint128>(key0, x).value();
        benchmark::DoNotOptimize(result);
        x = (x + 1) & ((absl::uint128{1} << kLogDomainSize) - 1);
    }
}
BENCHMARK(BM_DcfEval)->Name("google_dpf/CPU/DCF/Eval");

}  // namespace
