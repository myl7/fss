// Sample: VDPF on CPU
//
// Shows how to use Gen/Eval for the verifiable distributed point function
// (VDPF), plus Prove/Verify that let the parties check their evaluations
// agree with the shared key. Uses AES-128 MMO PRG and SHA-256 hashes.
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#include <fss/vdpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/hash/sha256.cuh>
#include <fss/prg/aes128_mmo.cuh>

// 8-bit input domain for a small example
constexpr int kInBits = 8;
using In = uint8_t;
using Group = fss::group::Bytes;

// VDPF uses a mul=2 PRG (the inner DPF) and SHA-256 for the hashes
using VdpfPrg = fss::prg::Aes128Mmo<2>;
using Vdpf = fss::Vdpf<kInBits, Group, VdpfPrg, fss::hash::Sha256, fss::hash::Sha256, In>;

// Compare two int4 values
static bool Equal(int4 a, int4 b) {
  return memcmp(&a, &b, sizeof(int4)) == 0;
}

// Reconstruct: convert Eval outputs to group elements, add, convert back
static int4 Reconstruct(int4 y0, int4 y1) {
  return (Group::From(y0) + Group::From(y1)).Into();
}

// Deterministic LSB-cleared seeds for retry round r
static void MakeSeeds(int r, int4 seeds[2]) {
  seeds[0] = {0x11111111 + r, 0x22222222 + r, 0x33333333 + r, 0x44444440 + r};
  seeds[1] = {0x55555555 + r, 0x66666666 + r, 0x77777777 + r, static_cast<int>(0x88888880u) + r};
}

int main() {
  printf("=== VDPF Sample ===\n");

  // Create AES cipher contexts (mul=2)
  unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const unsigned char *keys[2] = {key0, key1};
  auto ctxs = VdpfPrg::CreateCtxs(keys);

  VdpfPrg prg(ctxs);
  fss::hash::Sha256 xor_hash(
      {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, static_cast<int>(0x2468ace0u)});
  fss::hash::Sha256 hash_(
      {static_cast<int>(0x0fedcba9u), static_cast<int>(0x87654321u), static_cast<int>(0x2468ace0u),
          0x13572468});
  Vdpf vdpf{prg, xor_hash, hash_};

  // Secret inputs: alpha (point), beta (payload)
  In alpha = 42;
  int4 beta = {7, 0, 0, 0};

  // Key generation. Gen() returns 1 when the sampled seeds hit a bad case;
  // the dealer retries with fresh seeds then.
  Vdpf::Cw cws[kInBits];
  cuda::std::array<int4, 4> cs;
  int4 ocw;
  int4 seeds[2];
  int ret;
  int r = 0;
  do {
    MakeSeeds(r, seeds);
    ret = vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(seeds, 2), alpha, beta);
    ++r;
  } while (ret != 0);
  printf("  Gen retried %d time(s)\n", r - 1);

  // Evaluation with proof. Each party also gets pi_tilde at every point.
  int4 y0, y1;
  auto pi_tilde0 =
      vdpf.Eval(false, seeds[0], cuda::std::span<const Vdpf::Cw>(cws, kInBits),
          cuda::std::span<const int4, 4>(cs), ocw, alpha, y0);
  auto pi_tilde1 =
      vdpf.Eval(true, seeds[1], cuda::std::span<const Vdpf::Cw>(cws, kInBits),
          cuda::std::span<const int4, 4>(cs), ocw, alpha, y1);
  int4 sum = Reconstruct(y0, y1);
  printf("  Eval(x=%d == alpha): y0+y1 == beta? %s\n", alpha, Equal(sum, beta) ? "yes" : "NO");

  // Prove: combine the per-point pi_tildes of each party into a single proof.
  // With one evaluation the proof just aggregates that point's pi_tilde.
  cuda::std::array<int4, 4> pi0, pi1;
  vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(&pi_tilde0, 1),
      cuda::std::span<const int4, 4>(cs), pi0);
  vdpf.Prove(cuda::std::span<const cuda::std::array<int4, 4>>(&pi_tilde1, 1),
      cuda::std::span<const int4, 4>(cs), pi1);

  // Verify: the two parties' proofs match iff they evaluated the same key
  printf("  Verify(pi0, pi1) == true?           %s\n",
      Vdpf::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)) ? "yes" : "NO");

  // Full-domain evaluation over all 2^8 inputs
  int4 ys0[1 << kInBits];
  int4 ys1[1 << kInBits];
  vdpf.EvalAll(false, seeds[0], cuda::std::span<const Vdpf::Cw>(cws, kInBits),
      cuda::std::span<const int4, 4>(cs), ocw, cuda::std::span<int4>(ys0), pi0);
  vdpf.EvalAll(true, seeds[1], cuda::std::span<const Vdpf::Cw>(cws, kInBits),
      cuda::std::span<const int4, 4>(cs), ocw, cuda::std::span<int4>(ys1), pi1);

  int4 zero = {0, 0, 0, 0};
  int mismatches = 0;
  for (int x = 0; x < (1 << kInBits); ++x) {
    int4 expected = (x == alpha) ? beta : zero;
    if (!Equal(Reconstruct(ys0[x], ys1[x]), expected)) {
      ++mismatches;
    }
  }
  printf("  EvalAll: mismatches over 2^%d inputs: %d\n", kInBits, mismatches);
  printf("  EvalAll: Verify(pi0, pi1) == true?  %s\n",
      Vdpf::Verify(cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)) ? "yes" : "NO");

  VdpfPrg::FreeCtxs(ctxs);
  return 0;
}
