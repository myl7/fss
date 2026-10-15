// Benchmark adapter for the header-only dpf++ library
// (https://github.com/rh3nry/dpfplusplus), the anonymous pre-release of the
// DPF++ headers that upstream plans to publish as libdpf++; see
// doc/bench-dpfplusplus.md for provenance and license status. Timing
// boundaries follow the upstream API: dpf_key::gen draws its root seeds with
// arc4random_buf inside the call, and evalfull mallocs and frees its traversal
// hint array inside the call, so both are inside the timed region. Eval
// times one key.eval point evaluation per iteration. EvalAll times one
// party's evalfull and counts every domain output as work.
#include <benchmark/benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "dpf.hpp"

using dpfkey_t = dpf::dpf_key<bool>;        // node_t=__m128i, prg=dpf::prg::aes
using dpfkey64_t = dpf::dpf_key<uint64_t>;  // wider point values

static constexpr size_t kAlpha = 42;
static constexpr size_t kU64DomainBits = 12;
static constexpr uint64_t kU64Val = 0xdeadbeefcafebabeULL;

[[noreturn]] static void Fail(const char *what, size_t nbits) {
  fprintf(stderr, "dpf++ %s failed (n=2^%zu)\n", what, nbits);
  exit(EXIT_FAILURE);
}

// One-off self-check: both parties' evalfull shares must XOR to exactly the
// alpha bit, and point evaluation must agree at and away from the target.
static void CheckDomain(size_t nbits) {
  const size_t nitems = size_t{1} << nbits;
  const size_t target = kAlpha & (nitems - 1);
  auto keys = dpfkey_t::gen(nitems, target);
  const dpfkey_t &k0 = keys.first;
  const dpfkey_t &k1 = keys.second;

  const size_t probe = (target + 7919) % nitems;
  if (!(k0.eval(target) ^ k1.eval(target))) Fail("point eval at target", nbits);
  if (k0.eval(probe) ^ k1.eval(probe)) Fail("point eval away from target", nbits);

  // bool leaves are packed 128 per __m128i node; +1 block of slack as in the
  // upstream full_bytes() sizing.
  const size_t blocks = nitems / 128 + 1;
  std::vector<__m128i> v0(blocks), v1(blocks);
  k0.evalfull(reinterpret_cast<bool *>(v0.data()));
  k1.evalfull(reinterpret_cast<bool *>(v1.data()));
  const uint64_t *w0 = reinterpret_cast<const uint64_t *>(v0.data());
  const uint64_t *w1 = reinterpret_cast<const uint64_t *>(v1.data());
  size_t set_bits = 0, set_at = 0;
  for (size_t x = 0; x < nitems; ++x) {
    if (((w0[x / 64] ^ w1[x / 64]) >> (x % 64)) & 1) {
      ++set_bits;
      set_at = x;
    }
  }
  if (set_bits != 1 || set_at != target) Fail("full-domain reconstruction", nbits);
}

// Optional wider-valued variant: uint64_t point shares reconstruct the value
// at the target and zero elsewhere.
static void CheckUint64Points() {
  const size_t nitems = size_t{1} << kU64DomainBits;
  auto keys = dpfkey64_t::gen(nitems, 777, kU64Val);
  const uint64_t at_target = keys.first.eval(777) ^ keys.second.eval(777);
  const uint64_t at_probe = keys.first.eval(778) ^ keys.second.eval(778);
  if (at_target != kU64Val || at_probe != 0) Fail("uint64_t point eval", kU64DomainBits);
}

static void BM_DpfppGen(benchmark::State &state) {
  const size_t nbits = static_cast<size_t>(state.range(0));
  const size_t nitems = size_t{1} << nbits;
  CheckDomain(nbits);
  CheckUint64Points();
  for (auto _ : state) {
    auto keys = dpfkey_t::gen(nitems, kAlpha);
    benchmark::DoNotOptimize(&keys);
  }
}

static void BM_DpfppEval(benchmark::State &state) {
  const size_t nbits = static_cast<size_t>(state.range(0));
  const size_t nitems = size_t{1} << nbits;
  CheckDomain(nbits);
  auto keys = dpfkey_t::gen(nitems, kAlpha);
  const dpfkey_t &key = keys.first;
  uint64_t x = 0;
  for (auto _ : state) {
    const bool share = key.eval(x);
    benchmark::DoNotOptimize(share);
    x = (x + 1) & (nitems - 1);
  }
}

static void BM_DpfppEvalAll(benchmark::State &state) {
  const size_t nbits = static_cast<size_t>(state.range(0));
  const size_t nitems = size_t{1} << nbits;
  CheckDomain(nbits);
  auto keys = dpfkey_t::gen(nitems, kAlpha);
  const dpfkey_t &key = keys.first;
  std::vector<__m128i> out(nitems / 128 + 1);
  for (auto _ : state) {
    key.evalfull(reinterpret_cast<bool *>(out.data()));
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nitems));
}

BENCHMARK(BM_DpfppGen)->Name("dpfplusplus/CPU/Gen")->Arg(16)->Arg(20);
BENCHMARK(BM_DpfppEval)->Name("dpfplusplus/CPU/Eval")->Arg(16)->Arg(20);
BENCHMARK(BM_DpfppEvalAll)->Name("dpfplusplus/CPU/EvalAll")->Arg(16)->Arg(18)->Arg(20)->Arg(24);
