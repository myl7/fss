#pragma once

#include <benchmark/benchmark.h>
#include <optional>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

namespace fss_bench {

inline constexpr int kWorkSchemaVersion = 1;

inline int64_t CheckedMul(int64_t a, int64_t b) {
  int64_t product = 0;
  if (__builtin_mul_overflow(a, b, &product)) {
    fprintf(stderr, "work counter product overflows int64: %lld * %lld\n", static_cast<long long>(a),
        static_cast<long long>(b));
    exit(1);
  }
  return product;
}

inline int FullDomainCallsPerIteration() {
  static const int value = [] {
    const char *env = getenv("FSS_BENCH_FULL_LAUNCHES");
    if (env == nullptr || env[0] == '\0') return 1;
    char *end = nullptr;
    const long parsed = strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed < 1 || parsed > 1024) {
      fprintf(stderr, "invalid FSS_BENCH_FULL_LAUNCHES '%s': expected an integer in 1..1024\n", env);
      exit(1);
    }
    return static_cast<int>(parsed);
  }();
  return value;
}

// Work counters for ONE completed framework iteration. keys counts key pairs
// for Gen rows and participating-party keys for Eval rows, queries counts
// evaluated input points, logical_outputs counts scheme outputs (packed rows
// count output bits, not storage blocks; proofs are not logical outputs),
// and output_bytes counts the logical output storage. Inapplicable
// dimensions default to 0; api_calls stays absent unless the row names a
// meaningful host API boundary, which raw custom kernels do not have.
struct IterationWork {
  int64_t keys = 0;
  int64_t queries = 0;
  int64_t logical_outputs = 0;
  int64_t output_bytes = 0;
  std::optional<int64_t> api_calls;
};

inline void SetWork(benchmark::State &state, const IterationWork &work) {
  state.counters["work_schema_version"] = kWorkSchemaVersion;
  state.counters["keys_per_iteration"] = work.keys;
  state.counters["queries_per_iteration"] = work.queries;
  state.counters["logical_outputs_per_iteration"] = work.logical_outputs;
  state.counters["output_bytes_per_iteration"] = work.output_bytes;
  if (work.api_calls) state.counters["api_calls_per_iteration"] = *work.api_calls;
}

// Describes the proof blocks a proof-heavy row writes per iteration. Proofs
// are not logical function outputs, so such rows report logical_outputs 0.
inline void SetProofWork(benchmark::State &state, int64_t proofs, int64_t proof_bytes) {
  state.counters["proofs_per_iteration"] = proofs;
  state.counters["proof_bytes_per_iteration"] = proof_bytes;
}

// Records the corrected per-point hash (pi_tilde) storage bytes a row writes
// per iteration, not accumulated proof bytes.
inline void SetPointHashWork(benchmark::State &state, int64_t point_hash_bytes) {
  state.counters["point_hash_bytes_per_iteration"] = point_hash_bytes;
}

// Describes the domain leaves a preprocessing row prepares per iteration.
// Preprocessing is not a public query evaluation, so such rows report
// queries 0.
inline void SetPreparedLeaves(benchmark::State &state, int64_t leaves) {
  state.counters["prepared_leaves_per_iteration"] = leaves;
}

// Records the input domain the GPU fixture's alphas and xs are masked to, so
// a receipt states which fixture identity produced it. The public Gen/Eval
// preconditions require those values in [0, 2^domain_bits).
inline void SetFixtureDomain(benchmark::State &state, int64_t domain_bits) {
  state.counters["fixture_domain_bits"] = domain_bits;
}

// Batch full-domain rows walk a fixed fixture prefix: one iteration issues
// calls_per_iteration batch API calls over the first
// batch_keys * calls_per_iteration keys of the fixture_keys keys, in order
// from key 0, reusing the same prefix every iteration.
inline void SetBatchWork(benchmark::State &state, int64_t batch_keys, int64_t calls_per_iteration,
    int64_t fixture_keys) {
  state.counters["batch_keys"] = batch_keys;
  state.counters["calls_per_iteration"] = calls_per_iteration;
  state.counters["fixture_keys"] = fixture_keys;
}

// Steady-clock host wall time for GPU rows, separate from the manual
// iteration time the framework reports: Start() begins at the first event
// submission, Stop() ends after the stop-event synchronize, and Report()
// averages the accumulated time over the measured iterations.
class HostWallTimer {
 public:
  void Start() { start_ = std::chrono::steady_clock::now(); }
  void Stop() {
    total_ns_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_).count();
  }
  void Report(benchmark::State &state) {
    const int64_t iterations = static_cast<int64_t>(state.iterations());
    if (iterations > 0) state.counters["host_wall_ns_per_iteration"] = static_cast<double>(total_ns_) / iterations;
  }

 private:
  std::chrono::steady_clock::time_point start_{};
  int64_t total_ns_ = 0;
};

}  // namespace fss_bench
