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

// Deterministic LSB-cleared seed for seed index i
static int4 MakeSeed(int i) {
  return {0x11111111 + i, 0x22222222 + 2 * i, 0x33333333 + 3 * i, static_cast<int>(0x44444440u + 4 * i) & ~1};
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

  // Key generation. Gen() retries internally with perturbed sigma and seeds on
  // the rare failure cases, and writes the effective values back into sigma
  // and s0s. m is the cuckoo table size.
  constexpr int m = Vdmpf::m;
  Vdmpf::Key k0, k1;
  int4 sigma = {static_cast<int>(0xdeadbeefu), 0x12345678, static_cast<int>(0xabcdef01u),
      static_cast<int>(0x98765432u) & ~1};
  cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
  for (int i = 0; i < m; ++i) {
    s0s[i] = {MakeSeed(2 * i), MakeSeed(2 * i + 1)};
  }
  if (vdmpf.Gen(k0, k1, sigma, cuda::std::span<cuda::std::array<int4, 2>, m>(s0s), cuda::std::span<const In>(alphas, kT),
      cuda::std::span<const int4>(betas, kT), kT, 1000, true) != 0) {
    fprintf(stderr, "vdmpf gen failed after retries\n");
    VdmpfPrg::FreeCtxs(ctxs);
    return 1;
  }
  printf("  Gen done, table size m=%d\n", m);

  // Batch evaluation at the points: y0+y1 == beta_i
  std::vector<In> xs(alphas, alphas + kT);
  std::vector<int4> ys0(kT), ys1(kT);
  cuda::std::array<int4, 4> pi0, pi1;
  vdmpf.BatchEval(false, k0, std::span<const In>(xs), std::span<int4>(ys0), pi0);
  vdmpf.BatchEval(true, k1, std::span<const In>(xs), std::span<int4>(ys1), pi1);

  int mismatches = 0;
  for (int i = 0; i < kT; ++i) {
    int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
    if (!Equal(sum, betas[i])) {
      ++mismatches;
    }
  }
  printf("  BatchEval at points: mismatches: %d\n", mismatches);

  // Batch evaluation away from the points: y0+y1 == 0
  In non_alphas[kT] = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57,
      59, 61, 63, 65, 67, 69};
  xs.assign(non_alphas, non_alphas + kT);
  vdmpf.BatchEval(false, k0, std::span<const In>(xs), std::span<int4>(ys0), pi0);
  vdmpf.BatchEval(true, k1, std::span<const In>(xs), std::span<int4>(ys1), pi1);

  int4 zero = {0, 0, 0, 0};
  mismatches = 0;
  for (int i = 0; i < kT; ++i) {
    int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
    if (!Equal(sum, zero)) {
      ++mismatches;
    }
  }
  printf("  BatchEval off points: mismatches: %d\n", mismatches);

  VdmpfPrg::FreeCtxs(ctxs);
  return 0;
}
