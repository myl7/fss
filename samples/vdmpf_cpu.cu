// Sample: VDMPF on CPU
//
// Shows how to use Gen/BatchEval for the verifiable distributed multi-point
// function, which realizes t point functions at once with a cuckoo-hash
// packing. Uses AES-128 MMO PRG, SHA-256 hashes, and an AES-Feistel PRP for
// the cuckoo hashing.
#include <stdio.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <openssl/rand.h>

#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/hash/sha256.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prp/aes128_feistel.cuh>

// 8-bit input domain, at most 32 points, bucket address space of 8 bits
constexpr int kInBits = 8;
constexpr int kMaxPoints = 32;
constexpr int kBucketBits = 8;
constexpr int kT = 30;  // number of points to pack
using In = uint8_t;
using Group = fss::group::Bytes;

using VdmpfPrg = fss::prg::Aes128Mmo<2>;
using Vdmpf = fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, Group, VdmpfPrg, fss::hash::Sha256, fss::hash::Sha256,
    fss::prp::Aes128Feistel, In>;

// Compare two int4 values
static bool Equal(int4 a, int4 b) {
  return memcmp(&a, &b, sizeof(int4)) == 0;
}

// Draw a fresh uniform seed block from OpenSSL's cryptographic RNG.
static bool RandomSeed(int4 &seed) {
  return RAND_bytes(reinterpret_cast<unsigned char *>(&seed), sizeof(seed)) == 1;
}

int main() {
  printf("=== VDMPF Sample ===\n");

  // Create AES cipher contexts (mul=2)
  unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const unsigned char *keys[2] = {key0, key1};
  auto ctxs = VdmpfPrg::CreateCtxs(keys);

  VdmpfPrg prg(ctxs);
  fss::hash::Sha256 xor_hash({0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)});
  fss::hash::Sha256 hash_(
      {static_cast<int>(0x0fedcba9u), static_cast<int>(0x87654321u), static_cast<int>(0x2468ace0u), 0x13572468});
  fss::prp::Aes128Feistel prp;
  Vdmpf vdmpf{prg, xor_hash, hash_, prp};

  // Secret inputs: t points alpha_i with payloads beta_i
  In alphas[kT] = {10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58,
      60, 62, 64, 66, 68};
  int4 betas[kT];
  for (int i = 0; i < kT; ++i) {
    betas[i] = {(i + 1) * 11, 0, 0, 0};
  }

  // The sample owns retries. m is the cuckoo table size.
  constexpr int m = Vdmpf::kBucketCount;
  Vdmpf::Cw shared_cws[size_t{m} * (kBucketBits + 1)];
  int m_rt = 0;
  __uint128_t b_size_rt = 0;
  int4 sigma;
  cuda::std::array<int4, size_t{m} * 2> s0s;
  auto cws = cuda::std::span<Vdmpf::Cw>(shared_cws);
  int4 shared_cs[size_t{m} * 4];
  auto cs = cuda::std::span<int4>(shared_cs);
  constexpr int kMaxAttempts = 128;
  int status = 1;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    // Placement failure changes only sigma; inner VDPF failure changes seeds.
    if ((attempt == 0 || status == 1) && !RandomSeed(sigma)) {
      fprintf(stderr, "seed generation failed\n");
      VdmpfPrg::FreeCtxs(ctxs);
      return 1;
    }
    if (attempt == 0 || status == 2) {
      for (int i = 0; i < m; ++i) {
        if (!RandomSeed(s0s[i]) || !RandomSeed(s0s[size_t{m} + i])) {
          fprintf(stderr, "seed generation failed\n");
          VdmpfPrg::FreeCtxs(ctxs);
          return 1;
        }
      }
    }
    status = vdmpf.Gen(cws, cs, m_rt, b_size_rt, sigma,
        cuda::std::span<const int4>(s0s),
        cuda::std::span<const In>(alphas), cuda::std::span<const int4>(betas), kT, 1000);
    if (status == 0) break;
  }
  if (status != 0) {
    fprintf(stderr, "vdmpf gen failed after %d attempts (status %d)\n", kMaxAttempts, status);
    VdmpfPrg::FreeCtxs(ctxs);
    return 1;
  }
  auto seeds0 = cuda::std::span<const int4, m>(s0s.data(), m);
  auto seeds1 = cuda::std::span<const int4, m>(s0s.data() + m, m);
  printf("  Gen done, table size m=%d, runtime bucket count m_rt=%d\n", m, m_rt);
  // Evaluation takes only b_size_rt: it bounds the bucket index of every candidate.

  // Batch evaluation at the points: y0+y1 == beta_i
  std::vector<In> xs(alphas, alphas + kT);
  std::vector<int4> ys0(kT), ys1(kT);
  cuda::std::array<int4, 4> pi0, pi1;
  vdmpf.BatchEval(false, sigma, b_size_rt, seeds0, cws, cs, cuda::std::span<const In>(xs), cuda::std::span<int4>(ys0), pi0);
  vdmpf.BatchEval(true, sigma, b_size_rt, seeds1, cws, cs, cuda::std::span<const In>(xs), cuda::std::span<int4>(ys1), pi1);

  bool proofs_match = Vdmpf::Verify(pi0, pi1);
  int mismatches = 0;
  for (int i = 0; i < kT; ++i) {
    int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
    if (!Equal(sum, betas[i])) {
      ++mismatches;
    }
  }
  printf("  BatchEval at points: mismatches: %d, proof match: %s\n", mismatches, proofs_match ? "yes" : "NO");
  bool ok = mismatches == 0 && proofs_match;

  // Batch evaluation away from the points: y0+y1 == 0
  In non_alphas[kT] = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57,
      59, 61, 63, 65, 67, 69};
  xs.assign(non_alphas, non_alphas + kT);
  vdmpf.BatchEval(false, sigma, b_size_rt, seeds0, cws, cs, cuda::std::span<const In>(xs), cuda::std::span<int4>(ys0), pi0);
  vdmpf.BatchEval(true, sigma, b_size_rt, seeds1, cws, cs, cuda::std::span<const In>(xs), cuda::std::span<int4>(ys1), pi1);

  int4 zero = {0, 0, 0, 0};
  mismatches = 0;
  for (int i = 0; i < kT; ++i) {
    int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
    if (!Equal(sum, zero)) {
      ++mismatches;
    }
  }
  proofs_match = Vdmpf::Verify(pi0, pi1);
  printf("  BatchEval off points: mismatches: %d, proof match: %s\n", mismatches, proofs_match ? "yes" : "NO");

  VdmpfPrg::FreeCtxs(ctxs);
  return (ok && mismatches == 0 && proofs_match) ? 0 : 1;
}
