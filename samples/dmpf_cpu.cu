// Sample: DMPF on CPU
//
// Shows how to use Gen/BatchEval for the distributed multi-point function,
// which realizes t point functions at once with a cuckoo-hash packing.
// Uses AES-128 MMO PRG, SHA-256 hashes, and an AES-Feistel PRP for
// the cuckoo hashing.

#include <stdio.h>
#include <string.h>
// #include <span>
#include <vector>
#include <cuda_runtime.h>

#include <fss/dmpf.cuh>
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

using DmpfPrg = fss::prg::Aes128Mmo<2>;
using Dmpf = fss::Dmpf<kInBits, kMaxPoints, kBucketBits, Group, DmpfPrg, fss::prp::Aes128Feistel, In>;

static bool Equal(int4 a, int4 b) {
  return memcmp(&a, &b, sizeof(int4)) == 0;
}

// Deterministic LSB-cleared seed for retry round r and seed index i
static int4 MakeSeed(int r, int i) {
  return {
      0x11111111 + r + i,
      0x22222222 + r + 2 * i,
      0x33333333 + r + 3 * i,
      static_cast<int>(0x44444440u + r + 4 * i) & ~1,
  };
}

int main() {
  printf("=== DMPF Sample ===\n");

  // Create AES cipher contexts (mul=2)
  unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const unsigned char *keys[2] = {key0, key1};
  auto ctxs = DmpfPrg::CreateCtxs(keys);

  DmpfPrg prg(ctxs);
  fss::hash::Sha256 xor_hash({0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)});
  fss::hash::Sha256 hash_(
      {static_cast<int>(0x0fedcba9u), static_cast<int>(0x87654321u), static_cast<int>(0x2468ace0u), 0x13572468});
  fss::prp::Aes128Feistel prp;
  Dmpf dmpf{prg, prp};

  // Secret inputs: t points alpha_i with payloads beta_i
  In alphas[kT] = {10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58,
      60, 62, 64, 66, 68};
  int4 betas[kT];
  for (int i = 0; i < kT; ++i) {
    betas[i] = {(i + 1) * 11, 0, 0, 0};
  }

  // Key generation. Gen() returns 1 when the sampled seeds hit a bad case;
  // the dealer retries with fresh seeds then. m is the cuckoo table size.
  constexpr int m = Dmpf::m;
  Dmpf::Key k0, k1;
  int ret;
  int r = 0;
  do {
    int4 sigma = {static_cast<int>(0xdeadbeefu) + r, 0x12345678 + r, static_cast<int>(0xabcdef01u + r),
        static_cast<int>(0x98765432u + r) & ~1};
    cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = {MakeSeed(r, 2 * i), MakeSeed(r, 2 * i + 1)};
    }
    ret = dmpf.Gen(k0, k1, sigma, cuda::std::span<const cuda::std::array<int4, 2>, m>(s0s),
        std::span<const In>(alphas, kT), std::span<const int4>(betas, kT), kT);
    ++r;
  } while (ret != 0);
  printf("  Gen retried %d time(s), table size m=%d\n", r - 1, m);

  // Batch evaluation at the points: y0+y1 == beta_i.
  std::vector<In> xs(alphas, alphas + kT);
  std::vector<int4> ys0(kT), ys1(kT);

  dmpf.BatchEval(false, k0, std::span<const In>(xs), std::span<int4>(ys0));
  dmpf.BatchEval(true, k1, std::span<const In>(xs), std::span<int4>(ys1));

  int mismatches = 0;
  for (int i = 0; i < kT; ++i) {
    int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
    if (!Equal(sum, betas[i])) {
      ++mismatches;
    }
  }
  printf("  BatchEval off points: mismatches: %d\n", mismatches);

  // Batch evaluation away from the points: y0+y1 == 0
  In non_alphas[kT] = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57,
      59, 61, 63, 65, 67, 69};
  xs.assign(non_alphas, non_alphas + kT);
  dmpf.BatchEval(false, k0, std::span<const In>(xs), std::span<int4>(ys0));
  dmpf.BatchEval(true, k1, std::span<const In>(xs), std::span<int4>(ys1));

  int4 zero = {0, 0, 0, 0};
  mismatches = 0;
  for (int i = 0; i < kT; ++i) {
    int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
    if (!Equal(sum, zero)) {
      ++mismatches;
    }
  }
  printf("  BatchEval off points: mismatches: %d\n", mismatches);

  DmpfPrg::FreeCtxs(ctxs);

  return 0;
}