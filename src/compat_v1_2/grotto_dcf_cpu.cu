// Sample: Grotto DCF on CPU
//
// Shows how to use Gen/Preprocess/Eval for the Grotto DCF variant, whose
// output is a bool share (x >= alpha in the reconstructed value), plus the
// full-domain EvalAll. The Preprocess path builds a parity segment tree
// that answers later Eval() queries in O(log N).
#include <stdio.h>
#include <cuda_runtime.h>
#include <memory>

#include <fss/grotto_dcf.cuh>
#include <fss/prg/aes128_mmo.cuh>

// 8-bit input domain for a small example
constexpr int kInBits = 8;
using In = uint8_t;

// Grotto DCF uses a mul=2 PRG (the inner DPF)
using GdcfPrg = fss::prg::Aes128Mmo<2>;
using Gdcf = fss::GrottoDcf<kInBits, GdcfPrg, In>;

int main() {
  printf("=== Grotto DCF Sample ===\n");

  // Create AES cipher contexts (mul=2)
  unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const unsigned char *keys[2] = {key0, key1};
  auto ctxs = GdcfPrg::CreateCtxs(keys);

  GdcfPrg prg(ctxs);
  Gdcf dcf{prg};

  // Secret input: alpha (threshold)
  In alpha = 42;

  // Random seeds (LSB of .w must be 0)
  int4 seeds[2] = {
      {0x11111111, 0x22222222, 0x33333333, 0x44444440},
      {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
  };

  // Key generation (done by a trusted dealer)
  Gdcf::Cw cws[kInBits + 1];
  dcf.Gen(cws, seeds, alpha);

  // Preprocess: each party expands its tree and builds a parity segment tree.
  // pt.p must hold 2*N-1 bools, pt.b is the party index.
  constexpr size_t N = size_t{1} << kInBits;
  auto p0 = std::make_unique<bool[]>(2 * N - 1);
  auto p1 = std::make_unique<bool[]>(2 * N - 1);
  Gdcf::ParityTree pt0{p0.get(), false};
  Gdcf::ParityTree pt1{p1.get(), true};
  dcf.Preprocess(pt0, seeds[0], cws);
  dcf.Preprocess(pt1, seeds[1], cws);

  // Evaluation: the parties XOR their bool shares; the result is x >= alpha
  bool y0 = Gdcf::Eval(pt0, alpha);
  bool y1 = Gdcf::Eval(pt1, alpha);
  printf("  Eval(x=%d == alpha): y0^y1 == 1? %s\n", alpha, (y0 ^ y1) ? "yes" : "NO");

  In x = 10;
  y0 = Gdcf::Eval(pt0, x);
  y1 = Gdcf::Eval(pt1, x);
  printf("  Eval(x=%d  < alpha): y0^y1 == 0? %s\n", x, !(y0 ^ y1) ? "yes" : "NO");

  // Full-domain evaluation over all 2^8 inputs
  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);
  dcf.EvalAll(false, seeds[0], cws, ys0.get());
  dcf.EvalAll(true, seeds[1], cws, ys1.get());

  int mismatches = 0;
  for (size_t i = 0; i < N; ++i) {
    bool expected = (alpha <= i);
    if ((ys0[i] ^ ys1[i]) != expected) {
      ++mismatches;
    }
  }
  printf("  EvalAll: mismatches over 2^%d inputs: %d\n", kInBits, mismatches);

  GdcfPrg::FreeCtxs(ctxs);
  return 0;
}
