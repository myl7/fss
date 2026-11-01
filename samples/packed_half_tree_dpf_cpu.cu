// Sample: Packed Half-Tree DPF on CPU
//
// Shows how to use Gen/Eval/EvalAll for the packed-output DPF variant
// (experimental), which uses a CCR hash and instance seed S. Each 16B
// output block packs 128/out_bits consecutive outputs, so EvalAll returns
// 2^(in_bits-log2(128/out_bits)) blocks instead of 2^in_bits elements.
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <openssl/rand.h>

#include <fss/packed_half_tree_dpf.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/util.cuh>
#include <fss/hash/aes128_ccr.cuh>

// 8-bit input domain, 1-byte outputs: 16 lanes per block, 16 blocks in total
constexpr int kInBits = 8;
constexpr int kOutBits = 8;
using In = uint8_t;

using Prg = fss::prg::Aes128Mmo<1>;
using Hash = fss::hash::Aes128Ccr<Prg>;
using Dpf = fss::PackedHalfTreeDpf<kInBits, kOutBits, Hash, In>;

int main() {
  printf("=== Packed Half-Tree DPF Sample (experimental) ===\n");

  // Create the AES context for the CCR hash.
  unsigned char key[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const unsigned char *keys[1] = {key};
  auto ctxs = Prg::CreateCtxs(keys);

  Prg prg(ctxs);
  Hash hash(prg);
  Dpf dpf{hash, {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)}};

  // Secret inputs: alpha (point), beta (payload, must fit in out_bits bits)
  In alpha = 42;
  uint64_t beta = 0x5A;

  // Gen sets the party control bit; initial seeds are independent random blocks.
  int4 seeds[2];
  if (RAND_bytes(reinterpret_cast<unsigned char *>(seeds), sizeof(seeds)) != 1) {
    fprintf(stderr, "seed generation failed\n");
    Prg::FreeCtxs(ctxs);
    return 1;
  }

  // Key generation (done by a trusted dealer)
  Dpf::Cw cws[Dpf::kDepth + 1];
  dpf.Gen(cws, seeds, alpha, beta);

  // Evaluation: Eval returns the block containing x, Extract reads x's lane
  int4 diff = fss::util::Xor(dpf.Eval(false, seeds[0], cws, alpha), dpf.Eval(true, seeds[1], cws, alpha));
  printf("  Eval(x=%d == alpha): reconstruct == beta? %s\n", alpha,
      Dpf::Extract(diff, alpha) == beta ? "yes" : "NO");

  In x = 100;  // Different block from alpha
  diff = fss::util::Xor(dpf.Eval(false, seeds[0], cws, x), dpf.Eval(true, seeds[1], cws, x));
  printf("  Eval(x=%d != alpha): reconstruct == 0?    %s\n", x,
      Dpf::Extract(diff, x) == 0 ? "yes" : "NO");

  // Full-domain evaluation: 2^8 inputs in 2^4 blocks
  int4 ys0[1 << Dpf::kDepth];
  int4 ys1[1 << Dpf::kDepth];
  dpf.EvalAll(false, seeds[0], cws, ys0);
  dpf.EvalAll(true, seeds[1], cws, ys1);

  int mismatches = 0;
  for (int i = 0; i < (1 << kInBits); ++i) {
    int4 d = fss::util::Xor(ys0[i / Dpf::kLanes], ys1[i / Dpf::kLanes]);
    uint64_t expected = (i == alpha) ? beta : 0;
    if (Dpf::Extract(d, static_cast<In>(i)) != expected) {
      ++mismatches;
    }
  }
  printf("  EvalAll: mismatches over 2^%d inputs in 2^%d blocks: %d\n", kInBits, Dpf::kDepth, mismatches);

  Prg::FreeCtxs(ctxs);
  return 0;
}
