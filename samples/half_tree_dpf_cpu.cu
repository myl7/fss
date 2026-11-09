// Sample: Half-Tree DPF on CPU
//
// Shows how to use Gen/Eval/EvalAll for the Half-Tree DPF variant, which
// uses a CCR hash and a separate instance seed S.
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <openssl/rand.h>

#include <fss/half_tree_dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/hash/aes128_ccr.cuh>

// 8-bit input domain for a small example
constexpr int kInBits = 8;
using In = uint8_t;
using Group = fss::group::Bytes;

// The CCR construction wraps the single-key AES-MMO backend.
using HtDpfPrg = fss::prg::Aes128Mmo<1>;
using Hash = fss::hash::Aes128Ccr<HtDpfPrg>;
using HtDpf = fss::HalfTreeDpf<kInBits, Group, Hash, In>;

// Compare two int4 values
static bool Equal(int4 a, int4 b) {
  return memcmp(&a, &b, sizeof(int4)) == 0;
}

// Reconstruct: convert Eval outputs to group elements, add, convert back
static int4 Reconstruct(int4 y0, int4 y1) {
  return (Group::From(y0) + Group::From(y1)).Into();
}

int main() {
  printf("=== Half-Tree DPF Sample ===\n");

  // Create the AES context for the CCR hash.
  unsigned char key[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const unsigned char *keys[1] = {key};
  auto ctxs = HtDpfPrg::CreateCtxs(keys);

  HtDpfPrg prg(ctxs);
  Hash hash(prg);
  HtDpf dpf{hash, {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)}};

  // Secret inputs: alpha (point), beta (payload)
  In alpha = 42;
  int4 beta = {7, 0, 0, 0};

  // Gen sets the party control bit; initial seeds are independent random blocks.
  int4 seeds[2];
  if (RAND_bytes(reinterpret_cast<unsigned char *>(seeds), sizeof(seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    HtDpfPrg::FreeCtxs(ctxs);
    return 1;
  }

  // Key generation (done by a trusted dealer)
  int4 cws[kInBits + 1];
  In tr = 0;
  dpf.Gen(cuda::std::span<int4>(cws, kInBits + 1), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha, beta);

  // Evaluation
  int4 zero = {0, 0, 0, 0};
  int4 y0 = dpf.Eval(false, seeds[0], cuda::std::span<const int4>(cws, kInBits + 1), tr, alpha);
  int4 y1 = dpf.Eval(true, seeds[1], cuda::std::span<const int4>(cws, kInBits + 1), tr, alpha);
  int4 sum = Reconstruct(y0, y1);
  const bool eq_beta = Equal(sum, beta);
  printf("  Eval(x=%d == alpha): y0+y1 == beta? %s\n", alpha, eq_beta ? "yes" : "NO");

  In x = 100;
  y0 = dpf.Eval(false, seeds[0], cuda::std::span<const int4>(cws, kInBits + 1), tr, x);
  y1 = dpf.Eval(true, seeds[1], cuda::std::span<const int4>(cws, kInBits + 1), tr, x);
  sum = Reconstruct(y0, y1);
  const bool eq_zero = Equal(sum, zero);
  printf("  Eval(x=%d != alpha): y0+y1 == 0?    %s\n", x, eq_zero ? "yes" : "NO");

  // Full-domain evaluation over all 2^8 inputs
  int4 ys0[1 << kInBits];
  int4 ys1[1 << kInBits];
  dpf.EvalAll(false, seeds[0], cuda::std::span<const int4>(cws, kInBits + 1), tr, ys0);
  dpf.EvalAll(true, seeds[1], cuda::std::span<const int4>(cws, kInBits + 1), tr, ys1);

  int mismatches = 0;
  for (int i = 0; i < (1 << kInBits); ++i) {
    int4 expected = (i == alpha) ? beta : zero;
    if (!Equal(Reconstruct(ys0[i], ys1[i]), expected)) {
      ++mismatches;
    }
  }
  printf("  EvalAll: mismatches over 2^%d inputs: %d\n", kInBits, mismatches);

  HtDpfPrg::FreeCtxs(ctxs);
  return (eq_beta & eq_zero & (mismatches == 0)) ? 0 : 1;
}
