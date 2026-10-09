// Benchmark: fss-v0.7.0 (C) DPF and DCF
// DPF/DCF gen/eval with in_bits=20, uint (u128_le) and bytes (XOR) groups.
//
// group_add/group_neg/group_zero are link-time pluggable symbols, so we build
// separate executables per scheme and group type.
// AES DPF uses BLOCK_NUM=2 and AES DCF uses BLOCK_NUM=4.

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <cstring>
#include <random>

// fss-v0.7.0 C headers
extern "C" {
#include <dpf.h>
#include <dcf.h>

void prg_init(const uint8_t *state, int state_len);
}

#include "check.h"

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
static_assert(FSS_BENCH_DOMAIN_BITS >= 8 && FSS_BENCH_DOMAIN_BITS <= 20,
              "domain bits must be between 8 and 20");

static constexpr int kInBits = FSS_BENCH_DOMAIN_BITS;
static constexpr int kInBytes = (kInBits + 7) / 8;
static constexpr int kDomainSize = 1 << kInBits;
// Alpha packed as little-endian bytes (value 12345)
static constexpr uint32_t kAlphaVal = 12345 & (kDomainSize - 1);

namespace {

#ifdef FSS070_BENCH_DPF
// RAII wrapper for DPF key buffers.
struct DpfKeyBuf {
  DpfKey key;
  DpfKeyBuf() {
    key.cws = static_cast<uint8_t *>(malloc(kDpfCwLen * kInBits));
    key.cw_np1 = static_cast<uint8_t *>(malloc(kLambda));
  }
  ~DpfKeyBuf() {
    free(key.cws);
    free(key.cw_np1);
  }
  DpfKeyBuf(const DpfKeyBuf &) = delete;
  DpfKeyBuf &operator=(const DpfKeyBuf &) = delete;
};

#else
// RAII wrapper for DCF key buffers.
struct DcfKeyBuf {
  DcfKey key;
  DcfKeyBuf() {
    key.cws = static_cast<uint8_t *>(malloc(kDcfCwLen * kInBits));
    key.cw_np1 = static_cast<uint8_t *>(malloc(kLambda));
  }
  ~DcfKeyBuf() {
    free(key.cws);
    free(key.cw_np1);
  }
  DcfKeyBuf(const DcfKeyBuf &) = delete;
  DcfKeyBuf &operator=(const DcfKeyBuf &) = delete;
};

#endif

void CheckCorrectness() {
  static bool checked = false;
  if (checked) return;
  uint8_t seeds[kLambda * 2];
  for (int i = 0; i < kLambda * 2; ++i) seeds[i] = i * 17 + 3;
  seeds[15] &= 0x7f;
  seeds[31] &= 0x7f;
  uint8_t beta[kLambda];
  for (auto &value : beta) value = 0xff;
  beta[15] &= 0x7f;
  for (int test = 0; test < 10; ++test) {
    uint8_t alpha_bytes[kInBytes];
    uint8_t query_bytes[kInBytes];
    for (int i = 0; i < kInBytes; ++i) {
      alpha_bytes[i] = (kCheckedAlpha[test] & (kDomainSize - 1)) >> (8 * i);
      query_bytes[i] = (kCheckedQuery[test] & (kDomainSize - 1)) >> (8 * i);
    }
    CheckedBuffer cws(kCheckedCwLen * kInBits);
    CheckedBuffer last(kLambda);
    CheckedBuffer gen(kCheckedGenBytes);
    CheckedBuffer eval(kCheckedEvalBytes);
    CheckedKey key{cws.data(), last.data()};
    memcpy(gen.data(), seeds, sizeof(seeds));
    CheckedGen(key, Bits{alpha_bytes, kInBits}, beta, gen.data());
    uint8_t outputs[2][kLambda];
    for (int party = 0; party < 2; ++party) {
      memcpy(eval.data(), seeds + party * kLambda, kLambda);
      CheckedEval(eval.data(), party, key, Bits{query_bytes, kInBits});
      memcpy(outputs[party], eval.data(), kLambda);
      eval.Check();
    }
    CheckShares(kCheckedAlpha[test] & (kDomainSize - 1),
                kCheckedQuery[test] & (kDomainSize - 1), outputs[0],
                outputs[1], beta);
    cws.Check();
    last.Check();
    gen.Check();
  }
  checked = true;
  fprintf(stderr, "historical CPU reconstruction, MSB, and canary checks passed\n");
}

void InitPrg() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis;
  // prg_init expects BLOCK_NUM * 16 bytes of key material.
  // DPF uses BLOCK_NUM=2 (32 bytes), DCF uses BLOCK_NUM=4 (64 bytes).
  uint8_t keys[FSS070_PRG_BLOCK_NUM * 16];
  for (auto &k : keys) k = dis(gen);
  prg_init(keys, sizeof(keys));
  CheckCorrectness();
  if (getenv("FSS070_CHECK_ONLY")) exit(EXIT_SUCCESS);
}

}  // namespace

#ifdef FSS070_BENCH_DPF
// --- DPF benchmarks ---

static void BM_DpfGen(benchmark::State &state) {
  InitPrg();

  uint32_t alpha_int = kAlphaVal;
  uint8_t alpha_bytes[kInBytes];
  memcpy(alpha_bytes, &alpha_int, kInBytes);
  Bits alpha_bits = {alpha_bytes, kInBits};

  __uint128_t beta_int = 7;
  uint8_t *beta = reinterpret_cast<uint8_t *>(&beta_int);
  PointFunc pf = {alpha_bits, beta};

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint8_t> dis;
  uint8_t s0s[kLambda * 2];
  for (auto &b : s0s) b = dis(rng);

  uint8_t *sbuf = static_cast<uint8_t *>(malloc(kLambda * 6));

  for (auto _ : state) {
    DpfKeyBuf kb;
    memcpy(sbuf, s0s, kLambda * 2);
    dpf_gen(kb.key, pf, sbuf);
    benchmark::DoNotOptimize(kb.key.cws);
  }

  free(sbuf);
}
BENCHMARK(BM_DpfGen)->Name(BENCH_NAME_PREFIX "/DPF/Gen");

static void BM_DpfEval(benchmark::State &state) {
  InitPrg();

  uint32_t alpha_int = kAlphaVal;
  uint8_t alpha_bytes[kInBytes];
  memcpy(alpha_bytes, &alpha_int, kInBytes);
  Bits alpha_bits = {alpha_bytes, kInBits};

  __uint128_t beta_int = 7;
  uint8_t *beta = reinterpret_cast<uint8_t *>(&beta_int);
  PointFunc pf = {alpha_bits, beta};

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint8_t> dis;
  uint8_t s0s[kLambda * 2];
  for (auto &b : s0s) b = dis(rng);

  uint8_t *sbuf = static_cast<uint8_t *>(malloc(kLambda * 6));
  DpfKeyBuf kb;
  memcpy(sbuf, s0s, kLambda * 2);
  dpf_gen(kb.key, pf, sbuf);

  uint32_t x = 0;
  uint8_t x_bytes[kInBytes];
  for (auto _ : state) {
    memcpy(x_bytes, &x, kInBytes);
    Bits x_bits = {x_bytes, kInBits};
    memcpy(sbuf, s0s, kLambda);
    dpf_eval(sbuf, 0, kb.key, x_bits);
    benchmark::DoNotOptimize(sbuf);
    x = (x + 1) & (kDomainSize - 1);
  }

  free(sbuf);
}
BENCHMARK(BM_DpfEval)->Name(BENCH_NAME_PREFIX "/DPF/Eval");


#else
// --- DCF benchmarks ---

static void BM_DcfGen(benchmark::State &state) {
  InitPrg();

  uint32_t alpha_int = kAlphaVal;
  uint8_t alpha_bytes[kInBytes];
  memcpy(alpha_bytes, &alpha_int, kInBytes);
  Bits alpha_bits = {alpha_bytes, kInBits};

  __uint128_t beta_int = 7;
  uint8_t *beta = reinterpret_cast<uint8_t *>(&beta_int);
  CmpFunc cf = {alpha_bits, beta, kLtAlpha};

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint8_t> dis;
  uint8_t s0s[kLambda * 2];
  for (auto &b : s0s) b = dis(rng);

  uint8_t *sbuf = static_cast<uint8_t *>(malloc(kLambda * 10));

  for (auto _ : state) {
    DcfKeyBuf kb;
    memcpy(sbuf, s0s, kLambda * 2);
    dcf_gen(kb.key, cf, sbuf);
    benchmark::DoNotOptimize(kb.key.cws);
  }

  free(sbuf);
}
BENCHMARK(BM_DcfGen)->Name(BENCH_NAME_PREFIX "/DCF/Gen");

static void BM_DcfEval(benchmark::State &state) {
  InitPrg();

  uint32_t alpha_int = kAlphaVal;
  uint8_t alpha_bytes[kInBytes];
  memcpy(alpha_bytes, &alpha_int, kInBytes);
  Bits alpha_bits = {alpha_bytes, kInBits};

  __uint128_t beta_int = 7;
  uint8_t *beta = reinterpret_cast<uint8_t *>(&beta_int);
  CmpFunc cf = {alpha_bits, beta, kLtAlpha};

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint8_t> dis;
  uint8_t s0s[kLambda * 2];
  for (auto &b : s0s) b = dis(rng);

  uint8_t *sbuf = static_cast<uint8_t *>(malloc(kLambda * 10));
  DcfKeyBuf kb;
  memcpy(sbuf, s0s, kLambda * 2);
  dcf_gen(kb.key, cf, sbuf);

  uint32_t x = 0;
  uint8_t x_bytes[kInBytes];
  // dcf_eval needs 6 * kLambda buffer
  uint8_t *eval_sbuf = static_cast<uint8_t *>(malloc(kLambda * 6));
  for (auto _ : state) {
    memcpy(x_bytes, &x, kInBytes);
    Bits x_bits = {x_bytes, kInBits};
    memcpy(eval_sbuf, s0s, kLambda);
    dcf_eval(eval_sbuf, 0, kb.key, x_bits);
    benchmark::DoNotOptimize(eval_sbuf);
    x = (x + 1) & (kDomainSize - 1);
  }

  free(sbuf);
  free(eval_sbuf);
}
BENCHMARK(BM_DcfEval)->Name(BENCH_NAME_PREFIX "/DCF/Eval");


#endif
