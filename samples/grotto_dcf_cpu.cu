// Sample: Grotto DCF on CPU
//
// Shows how to use Gen/Preprocess/Eval for the Grotto DCF variant, whose
// output is a bool share (x >= alpha in the reconstructed value), plus the
// full-domain EvalAll. The Preprocess path builds a parity segment tree
// that answers later Eval() queries in O(log N).
#include <stdio.h>
#include <cuda_runtime.h>
#include <memory>
#include <openssl/rand.h>

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

  // Gen sets the party control bit; initial seeds are independent random blocks.
  int4 seeds[2];
  if (RAND_bytes(reinterpret_cast<unsigned char *>(seeds), sizeof(seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    GdcfPrg::FreeCtxs(ctxs);
    return 1;
  }

  // Key generation (done by a trusted dealer)
  int4 cws[kInBits + 1];
  In tr = 0;
  dcf.Gen(cuda::std::span<int4>(cws, kInBits + 1), tr,
      cuda::std::span<const int4, 2>(seeds, 2), alpha);

  // Preprocess: each party expands its tree and builds a parity segment tree.
  // The result owns its 2*N-1 parity bits and answers Eval() queries in O(log N).
  constexpr size_t N = size_t{1} << kInBits;
  auto tree0 = dcf.PreprocessOwned(false, seeds[0], cuda::std::span<const int4>(cws, kInBits + 1), tr);
  auto tree1 = dcf.PreprocessOwned(true, seeds[1], cuda::std::span<const int4>(cws, kInBits + 1), tr);

  // Evaluation: the parties XOR their bool shares; the result is x >= alpha
  bool y0 = Gdcf::Eval(tree0.View(), alpha);
  bool y1 = Gdcf::Eval(tree1.View(), alpha);
  const bool eq_at = y0 ^ y1;
  printf("  Eval(x=%d == alpha): y0^y1 == 1? %s\n", alpha, eq_at ? "yes" : "NO");

  In x = 10;
  y0 = Gdcf::Eval(tree0.View(), x);
  y1 = Gdcf::Eval(tree1.View(), x);
  const bool eq_below = !(y0 ^ y1);
  printf("  Eval(x=%d  < alpha): y0^y1 == 0? %s\n", x, eq_below ? "yes" : "NO");

  // Full-domain evaluation over all 2^8 inputs
  auto ys0 = std::make_unique<bool[]>(N);
  auto ys1 = std::make_unique<bool[]>(N);
  Gdcf::EvalAll(tree0.View(), ys0.get());
  Gdcf::EvalAll(tree1.View(), ys1.get());

  int mismatches = 0;
  for (size_t i = 0; i < N; ++i) {
    bool expected = (alpha <= i);
    if ((ys0[i] ^ ys1[i]) != expected) {
      ++mismatches;
    }
  }
  printf("  EvalAll: mismatches over 2^%d inputs: %d\n", kInBits, mismatches);

  GdcfPrg::FreeCtxs(ctxs);
  return (eq_at & eq_below & (mismatches == 0)) ? 0 : 1;
}
