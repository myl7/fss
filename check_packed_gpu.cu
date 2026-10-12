// Ad-hoc correctness check: PackedHalfTreeDpf device-side Eval (naive GPU EvalAll,
// one thread per output block) vs CPU EvalAll, both parties, plus XOR reconstruction.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/prg/chacha.cuh>

__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

#define CHECK(x) \
  do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "cuda error at %d: %s\n", __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

template <int in_bits, int out_bits>
__global__ void PackedEvalAllKernel(bool b, int4 s0,
    const typename fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint>::Cw *cws, int4 fcw, int4 *ys,
    fss::prg::ChaCha<1> prg, int4 hash_key, int num_blocks) {
  using Packed = fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint>;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= num_blocks) return;
  Packed dpf{prg, hash_key};
  ys[j] = dpf.Eval(b, s0, cws, fcw, static_cast<uint>(j) * Packed::kLanes);
}

template <int in_bits, int out_bits>
bool CheckPacked(int trials) {
  using Packed = fss::PackedHalfTreeDpf<in_bits, out_bits, fss::prg::ChaCha<1>, uint>;
  constexpr int kN = 1 << in_bits;
  constexpr int kNumBlocks = 1 << Packed::kDepth;
  const int4 kHashKey = {0x0fedcba9, static_cast<int>(0x87654321u), 0x13579bdf, 0x2468ace0};

  fss::prg::ChaCha<1> prg(kNonce);
  Packed dpf{prg, kHashKey};

  typename Packed::Cw *h_cws = new typename Packed::Cw[Packed::kDepth];
  int4 h_fcw;
  typename Packed::Cw *d_cws;
  int4 *d_fcw, *d_ys;
  CHECK(cudaMalloc(&d_cws, sizeof(typename Packed::Cw) * Packed::kDepth));
  CHECK(cudaMalloc(&d_fcw, sizeof(int4)));
  CHECK(cudaMalloc(&d_ys, sizeof(int4) * kNumBlocks));

  constexpr int kBs = 256;
  constexpr int kBlocks = (kNumBlocks + kBs - 1) / kBs;

  bool ok = true;
  for (int trial = 0; trial < trials; ++trial) {
    int4 s0s[2];
    uint alpha;
    uint64_t beta;
    s0s[0] = {rand(), rand(), rand(), rand()};
    s0s[1] = {rand(), rand(), rand(), rand()};
    alpha = static_cast<uint>(rand()) & (kN - 1);
    beta = static_cast<uint64_t>(rand()) & Packed::kBetaMax;

    dpf.Gen(h_cws, h_fcw, s0s, alpha, beta);
    CHECK(cudaMemcpy(d_cws, h_cws, sizeof(typename Packed::Cw) * Packed::kDepth, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_fcw, &h_fcw, sizeof(int4), cudaMemcpyHostToDevice));

    int4 *cpu[2], *gpu[2];
    for (int b = 0; b < 2; ++b) {
      cpu[b] = new int4[kNumBlocks];
      gpu[b] = new int4[kNumBlocks];

      dpf.EvalAll(b, s0s[b], h_cws, h_fcw, cpu[b]);
      PackedEvalAllKernel<in_bits, out_bits>
          <<<kBlocks, kBs>>>(b, s0s[b], d_cws, h_fcw, d_ys, prg, kHashKey, kNumBlocks);
      CHECK(cudaMemcpy(gpu[b], d_ys, sizeof(int4) * kNumBlocks, cudaMemcpyDeviceToHost));

      long mism = 0;
      for (int j = 0; j < kNumBlocks; ++j)
        if (cpu[b][j].x != gpu[b][j].x || cpu[b][j].y != gpu[b][j].y || cpu[b][j].z != gpu[b][j].z ||
            cpu[b][j].w != gpu[b][j].w)
          ++mism;
      printf("in_bits=%d out_bits=%d trial %d party %d: CPU==GPU %s (%ld mismatches of %d blocks)\n", in_bits, out_bits,
          trial, b, mism == 0 ? "PASS" : "FAIL", mism, kNumBlocks);
      if (mism) ok = false;
    }

    int bad = 0;
    for (int x = 0; x < kN; ++x) {
      int4 sum = {gpu[0][x / Packed::kLanes].x ^ gpu[1][x / Packed::kLanes].x,
          gpu[0][x / Packed::kLanes].y ^ gpu[1][x / Packed::kLanes].y,
          gpu[0][x / Packed::kLanes].z ^ gpu[1][x / Packed::kLanes].z,
          gpu[0][x / Packed::kLanes].w ^ gpu[1][x / Packed::kLanes].w};
      uint64_t lane = Packed::Extract(sum, static_cast<uint>(x));
      bool correct = static_cast<uint>(x) == alpha ? lane == beta : lane == 0;
      if (!correct) ++bad;
    }
    printf("in_bits=%d out_bits=%d trial %d FSS property (y0^y1 lane==beta at alpha else 0): %s (%d bad of %d)\n",
        in_bits, out_bits, trial, bad == 0 ? "PASS" : "FAIL", bad, kN);
    if (bad) ok = false;

    delete[] cpu[0];
    delete[] cpu[1];
    delete[] gpu[0];
    delete[] gpu[1];
  }

  cudaFree(d_cws);
  cudaFree(d_fcw);
  cudaFree(d_ys);
  delete[] h_cws;
  return ok;
}

int main() {
  srand(42);
  bool ok = true;
  ok &= CheckPacked<16, 8>(3);
  ok &= CheckPacked<16, 1>(3);
  ok &= CheckPacked<20, 64>(3);
  printf(ok ? "ALL PASS\n" : "FAILURES\n");
  return ok ? 0 : 1;
}
