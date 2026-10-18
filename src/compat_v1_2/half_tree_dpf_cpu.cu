// Sample: Half-Tree DPF on CPU
//
// Shows how to use Gen/Eval/EvalAll for the Half-Tree DPF variant, which
// needs a mul=1 PRG and a separate hash key for key stretching.
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#include <fss/half_tree_dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/aes128_mmo.cuh>

// 8-bit input domain for a small example
constexpr int kInBits = 8;
using In = uint8_t;
using Group = fss::group::Bytes;

// Half-Tree DPF uses a mul=1 PRG (one AES key)
using HtDpfPrg = fss::prg::Aes128Mmo<1>;
using HtDpf = fss::HalfTreeDpf<kInBits, Group, HtDpfPrg, In>;

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

  // Create the AES cipher context (mul=1) and the hash key
  unsigned char key[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const unsigned char *keys[1] = {key};
  auto ctxs = HtDpfPrg::CreateCtxs(keys);

  HtDpfPrg prg(ctxs);
  HtDpf dpf{prg, {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)}};

  // Secret inputs: alpha (point), beta (payload)
  In alpha = 42;
  int4 beta = {7, 0, 0, 0};

  // Random seeds (LSB of .w must be 0)
  int4 seeds[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
  };

  // Key generation (done by a trusted dealer)
  HtDpf::Cw cws[kInBits];
  int4 ocw;
  dpf.Gen(cws, ocw, seeds, alpha, beta);

  // Evaluation
  int4 zero = {0, 0, 0, 0};
  int4 y0 = dpf.Eval(false, seeds[0], cws, ocw, alpha);
  int4 y1 = dpf.Eval(true, seeds[1], cws, ocw, alpha);
  int4 sum = Reconstruct(y0, y1);
  printf("  Eval(x=%d == alpha): y0+y1 == beta? %s\n", alpha, Equal(sum, beta) ? "yes" : "NO");

  In x = 100;
  y0 = dpf.Eval(false, seeds[0], cws, ocw, x);
  y1 = dpf.Eval(true, seeds[1], cws, ocw, x);
  sum = Reconstruct(y0, y1);
  printf("  Eval(x=%d != alpha): y0+y1 == 0?    %s\n", x, Equal(sum, zero) ? "yes" : "NO");

  // Full-domain evaluation over all 2^8 inputs
  int4 ys0[1 << kInBits];
  int4 ys1[1 << kInBits];
  dpf.EvalAll(false, seeds[0], cws, ocw, ys0);
  dpf.EvalAll(true, seeds[1], cws, ocw, ys1);

  int mismatches = 0;
  for (int i = 0; i < (1 << kInBits); ++i) {
    int4 expected = (i == alpha) ? beta : zero;
    if (!Equal(Reconstruct(ys0[i], ys1[i]), expected)) {
      ++mismatches;
    }
  }
  printf("  EvalAll: mismatches over 2^%d inputs: %d\n", kInBits, mismatches);

  HtDpfPrg::FreeCtxs(ctxs);
  return 0;
}
