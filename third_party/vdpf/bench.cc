// Benchmark adapter for the Servan-Schreiber / Langowski VDPF reference
// implementation (https://github.com/sachaservan/vdpf), the code release of
// eprint 2021/580 "Lightweight, Maliciously Secure Verifiable Function Secret
// Sharing". Timing boundaries follow upstream src/test.c usage: the MMO hash
// objects are stateful (AES-CTR), so every evaluated call re-initializes them,
// and that re-initialization is inside the timed region. Proof verification is
// an equality check between the two parties' proof blocks and stays outside
// timing.
#include <benchmark/benchmark.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <vector>

// Upstream headers are C-only (typedef struct Hash hash round-trips that C++
// rejects), so declare the needed API directly and keep the types opaque.
extern "C" {
typedef struct Hash Hash;
extern EVP_CIPHER_CTX *getDPFContext(uint8_t *key);
extern void destroyContext(EVP_CIPHER_CTX *ctx);
extern void genVDPF(EVP_CIPHER_CTX *ctx, Hash *hash, int size, uint64_t index, unsigned char *k0, unsigned char *k1);
extern void batchEvalVDPF(EVP_CIPHER_CTX *ctx, Hash *mmo_hash1, Hash *mmo_hash2, int size, bool b,
    unsigned char *k, uint64_t *in, uint64_t inl, uint8_t *out, uint8_t *pi);
extern void fullDomainVDPF(EVP_CIPHER_CTX *ctx, Hash *mmo_hash1, Hash *mmo_hash2, int size, bool b,
    unsigned char *k, uint8_t *out, uint8_t *proof);
extern Hash *initMMOHash(uint8_t *seed, uint64_t outblocks);
extern void destroyMMOHash(Hash *hash);
}

using uint128_t = unsigned __int128;
constexpr int kFieldSize = 2;  // FIELDSIZE: shares are in F2

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
constexpr int kInBits = FSS_BENCH_DOMAIN_BITS;
static_assert(kInBits >= 1 && kInBits <= 30);

constexpr uint64_t kAlpha = 42;
constexpr size_t kOutBlocks = 4;
// INDEX_LASTCW expands to 18 * size + 18 with the runtime size; spell it out.
constexpr size_t kKeyBytes = 18 * kInBits + 18 + 16 + 16 * kOutBlocks;

static const uint8_t kPrgKey[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const uint8_t kHashKey1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
static const uint8_t kHashKey2[16] = {0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static void GenKeys(unsigned char *vk0, unsigned char *vk1) {
  EVP_CIPHER_CTX *ctx = getDPFContext(const_cast<uint8_t *>(kPrgKey));
  Hash *mmo_hash1 = initMMOHash(const_cast<uint8_t *>(kHashKey1), kOutBlocks);
  genVDPF(ctx, mmo_hash1, kInBits, kAlpha, vk0, vk1);
  destroyMMOHash(mmo_hash1);
  destroyContext(ctx);
}

// Both parties evaluate the same points; reconstruction must give 1 at alpha
// and 0 elsewhere (field F2, FIELDSIZE 2) and the proofs must match.
static void CheckPointEval(unsigned char *vk0, unsigned char *vk1) {
  EVP_CIPHER_CTX *ctx = getDPFContext(const_cast<uint8_t *>(kPrgKey));
  const uint64_t points[] = {0, kAlpha > 0 ? kAlpha - 1 : 0, kAlpha, kAlpha + 1,
      (1ULL << kInBits) - 1};
  uint128_t pi0[kOutBlocks], pi1[kOutBlocks];
  for (uint64_t x : points) {
    uint128_t share0 = 0, share1 = 0;
    uint64_t in = x;
    Hash *h1 = initMMOHash(const_cast<uint8_t *>(kHashKey1), kOutBlocks);
    Hash *h2 = initMMOHash(const_cast<uint8_t *>(kHashKey2), kOutBlocks);
    batchEvalVDPF(ctx, h1, h2, kInBits, false, vk0, &in, 1, (uint8_t *)&share0, (uint8_t *)&pi0[0]);
    destroyMMOHash(h1);
    destroyMMOHash(h2);
    h1 = initMMOHash(const_cast<uint8_t *>(kHashKey1), kOutBlocks);
    h2 = initMMOHash(const_cast<uint8_t *>(kHashKey2), kOutBlocks);
    batchEvalVDPF(ctx, h1, h2, kInBits, true, vk1, &in, 1, (uint8_t *)&share1, (uint8_t *)&pi1[0]);
    destroyMMOHash(h1);
    destroyMMOHash(h2);

    if ((int)((share0 + share1) % kFieldSize) != (x == kAlpha ? 1 : 0)) {
      fprintf(stderr, "servan vdpf reconstruction mismatch at input %llu\n", (unsigned long long)x);
      exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < 2; ++i) {
      if (pi0[i] != pi1[i]) {
        fprintf(stderr, "servan vdpf proof rejected on honest shares at input %llu\n", (unsigned long long)x);
        exit(EXIT_FAILURE);
      }
    }
  }
  destroyContext(ctx);
}

static void BM_ServanVdpfGen(benchmark::State &state) {
  EVP_CIPHER_CTX *ctx = getDPFContext(const_cast<uint8_t *>(kPrgKey));
  std::vector<unsigned char> vk0(kKeyBytes), vk1(kKeyBytes);
  GenKeys(vk0.data(), vk1.data());
  CheckPointEval(vk0.data(), vk1.data());
  for (auto _ : state) {
    Hash *mmo_hash1 = initMMOHash(const_cast<uint8_t *>(kHashKey1), kOutBlocks);
    genVDPF(ctx, mmo_hash1, kInBits, kAlpha, vk0.data(), vk1.data());
    destroyMMOHash(mmo_hash1);
    benchmark::DoNotOptimize(vk0.data());
    benchmark::DoNotOptimize(vk1.data());
  }
  destroyContext(ctx);
}

static void BM_ServanVdpfEval(benchmark::State &state) {
  EVP_CIPHER_CTX *ctx = getDPFContext(const_cast<uint8_t *>(kPrgKey));
  std::vector<unsigned char> vk0(kKeyBytes), vk1(kKeyBytes);
  GenKeys(vk0.data(), vk1.data());
  CheckPointEval(vk0.data(), vk1.data());
  uint64_t x = 0;
  for (auto _ : state) {
    uint64_t in = x;
    uint128_t share = 0;
    uint128_t pi[kOutBlocks];
    Hash *h1 = initMMOHash(const_cast<uint8_t *>(kHashKey1), kOutBlocks);
    Hash *h2 = initMMOHash(const_cast<uint8_t *>(kHashKey2), kOutBlocks);
    batchEvalVDPF(ctx, h1, h2, kInBits, false, vk0.data(), &in, 1, (uint8_t *)&share, (uint8_t *)&pi[0]);
    destroyMMOHash(h1);
    destroyMMOHash(h2);
    benchmark::DoNotOptimize(share);
    x = (x + 1) & ((1ULL << kInBits) - 1);
  }
  destroyContext(ctx);
}

static void BM_ServanVdpfEvalAll(benchmark::State &state) {
  EVP_CIPHER_CTX *ctx = getDPFContext(const_cast<uint8_t *>(kPrgKey));
  constexpr size_t n = size_t{1} << kInBits;
  std::vector<unsigned char> vk0(kKeyBytes), vk1(kKeyBytes);
  GenKeys(vk0.data(), vk1.data());
  CheckPointEval(vk0.data(), vk1.data());
  std::vector<uint128_t> shares(n);
  uint128_t pi[kOutBlocks];
  for (auto _ : state) {
    Hash *h1 = initMMOHash(const_cast<uint8_t *>(kHashKey1), kOutBlocks);
    Hash *h2 = initMMOHash(const_cast<uint8_t *>(kHashKey2), kOutBlocks);
    fullDomainVDPF(ctx, h1, h2, kInBits, false, vk0.data(), (uint8_t *)shares.data(), (uint8_t *)&pi[0]);
    destroyMMOHash(h1);
    destroyMMOHash(h2);
    benchmark::DoNotOptimize(shares.data());
    benchmark::DoNotOptimize(pi);
  }
  destroyContext(ctx);
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_ServanVdpfGen)->Name("servan_vdpf/CPU/VDPF/Gen");
BENCHMARK(BM_ServanVdpfEval)->Name("servan_vdpf/CPU/VDPF/Eval");
BENCHMARK(BM_ServanVdpfEvalAll)->Name("servan_vdpf/CPU/VDPF/EvalAll");
