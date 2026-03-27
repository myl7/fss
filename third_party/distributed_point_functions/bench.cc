// Benchmark: Google distributed_point_functions DPF and DCF
// DPF/DCF gen/eval with log_domain_size=20, XorWrapper<uint128> output.
//
// Build: cd third_party/distributed_point_functions && bazel build -c opt :bench_dpf_google
// Run:   bazel-bin/bench_dpf_google

#include <benchmark/benchmark.h>

#include "absl/numeric/int128.h"
#include "dcf/distributed_comparison_function.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/xor_wrapper.h"

namespace {

using namespace distributed_point_functions;

constexpr int kLogDomainSize = 20;

DpfParameters MakeDpfParams() {
    DpfParameters params;
    params.set_log_domain_size(kLogDomainSize);
    params.mutable_value_type()->mutable_xor_wrapper()->set_bitsize(128);
    return params;
}

DcfParameters MakeDcfParams() {
    DcfParameters params;
    *params.mutable_parameters() = MakeDpfParams();
    return params;
}

using T = XorWrapper<absl::uint128>;

// ---------------------------------------------------------------------------
// DPF benchmarks
// ---------------------------------------------------------------------------

void BM_DpfGen(benchmark::State& state) {
    auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
    absl::uint128 alpha = 12345;
    T beta{7};

    for (auto _ : state) {
        auto keys = dpf->GenerateKeys(alpha, beta).value();
        benchmark::DoNotOptimize(keys);
    }
}
BENCHMARK(BM_DpfGen)->Name("google_dpf/CPU/DPF/Gen");

void BM_DpfEvalAll(benchmark::State& state) {
    auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
    auto [key0, key1] = dpf->GenerateKeys(absl::uint128{12345}, T{7}).value();

    for (auto _ : state) {
        auto ctx = dpf->CreateEvaluationContext(key0).value();
        auto result = dpf->EvaluateUntil<T>(
            0, absl::Span<const absl::uint128>(), ctx).value();
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_DpfEvalAll)->Name("google_dpf/CPU/DPF/EvalAll");

void BM_DpfEval(benchmark::State& state) {
    auto dpf = DistributedPointFunction::Create(MakeDpfParams()).value();
    auto [key0, key1] = dpf->GenerateKeys(absl::uint128{12345}, T{7}).value();

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
    auto dcf = DistributedComparisonFunction::Create(MakeDcfParams()).value();
    absl::uint128 alpha = 12345;
    T beta{7};

    for (auto _ : state) {
        auto keys = dcf->GenerateKeys(alpha, beta).value();
        benchmark::DoNotOptimize(keys);
    }
}
BENCHMARK(BM_DcfGen)->Name("google_dpf/CPU/DCF/Gen");

void BM_DcfEval(benchmark::State& state) {
    auto dcf = DistributedComparisonFunction::Create(MakeDcfParams()).value();
    auto [key0, key1] = dcf->GenerateKeys(absl::uint128{12345}, T{7}).value();

    absl::uint128 x = 0;
    for (auto _ : state) {
        auto result = dcf->Evaluate<T>(key0, x).value();
        benchmark::DoNotOptimize(result);
        x = (x + 1) & ((absl::uint128{1} << kLogDomainSize) - 1);
    }
}
BENCHMARK(BM_DcfEval)->Name("google_dpf/CPU/DCF/Eval");

}  // namespace
