// Ad-hoc correctness check: GPU full-domain eval vs CPU EvalAll, both parties.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/eval_all_gpu.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>

constexpr int kInBits = 20;
constexpr int kN = 1 << kInBits;
using Group = fss::group::Uint<uint64_t>;
using HtDpf = fss::HalfTreeDpf<kInBits, Group, fss::prg::ChaCha<1>, uint>;
using Dpf = fss::Dpf<kInBits, Group, fss::prg::ChaCha<2>, uint>;

__constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};
const int4 kHalfTreeHashKey = {0x12345678, static_cast<int>(0x9abcdef0u), 0x0fedcba9, static_cast<int>(0x87654321u)};

#define CHECK(x) \
  do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "cuda error at %d: %s\n", __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

int main() {
  int4 s0s[2];
  uint alpha;
  int4 beta;
  srand(42);
  s0s[0] = {rand(), rand(), rand(), rand() & ~1};
  s0s[1] = {rand(), rand(), rand(), rand() & ~1};
  alpha = rand();
  beta = {rand(), rand(), rand(), rand() & ~1};

  fss::prg::ChaCha<1> prg1(kNonce);
  HtDpf ht{prg1, kHalfTreeHashKey};
  fss::prg::ChaCha<2> prg2(kNonce);
  Dpf dpf{prg2};

  HtDpf::Cw *h_ht_cws = new HtDpf::Cw[kInBits];
  int4 h_ht_ocw;
  Dpf::Cw *h_dpf_cws = new Dpf::Cw[kInBits + 1];
  ht.Gen(h_ht_cws, h_ht_ocw, s0s, alpha, beta);
  dpf.Gen(h_dpf_cws, s0s, alpha, beta);

  // Device copies
  HtDpf::Cw *d_ht_cws;
  Dpf::Cw *d_dpf_cws;
  int4 *d_ht_ocw;
  int4 *d_ht_ys, *d_dpf_ys;
  CHECK(cudaMalloc(&d_ht_cws, sizeof(HtDpf::Cw) * kInBits));
  CHECK(cudaMalloc(&d_dpf_cws, sizeof(Dpf::Cw) * (kInBits + 1)));
  CHECK(cudaMalloc(&d_ht_ocw, sizeof(int4)));
  CHECK(cudaMalloc(&d_ht_ys, sizeof(int4) * kN));
  CHECK(cudaMalloc(&d_dpf_ys, sizeof(int4) * kN));
  CHECK(cudaMemcpy(d_ht_cws, h_ht_cws, sizeof(HtDpf::Cw) * kInBits, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_dpf_cws, h_dpf_cws, sizeof(Dpf::Cw) * (kInBits + 1), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_ht_ocw, &h_ht_ocw, sizeof(int4), cudaMemcpyHostToDevice));

  bool ok = true;
  for (int b = 0; b < 2; ++b) {
    int4 *cpu = new int4[kN];
    int4 *gpu = new int4[kN];

    // HalfTree
    ht.EvalAll(b, s0s[b], h_ht_cws, h_ht_ocw, cpu);
    fss::gpu::HalfTreeDpfEvalAllGpu<16>(b, s0s[b], d_ht_cws, h_ht_ocw, d_ht_ys, ht);
    CHECK(cudaMemcpy(gpu, d_ht_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));
    long mism = 0;
    for (int i = 0; i < kN; ++i)
      if (cpu[i].x != gpu[i].x || cpu[i].y != gpu[i].y || cpu[i].z != gpu[i].z || cpu[i].w != gpu[i].w) ++mism;
    printf("HalfTree party %d: CPU==GPU %s (%ld mismatches of %d)\n", b, mism == 0 ? "PASS" : "FAIL", mism, kN);
    if (mism) ok = false;

    // DPF
    dpf.EvalAll(b, s0s[b], h_dpf_cws, cpu);
    fss::gpu::DpfEvalAllGpu<16>(b, s0s[b], d_dpf_cws, d_dpf_ys, dpf);
    CHECK(cudaMemcpy(gpu, d_dpf_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));
    mism = 0;
    for (int i = 0; i < kN; ++i)
      if (cpu[i].x != gpu[i].x || cpu[i].y != gpu[i].y || cpu[i].z != gpu[i].z || cpu[i].w != gpu[i].w) ++mism;
    printf("Dpf party %d: CPU==GPU %s (%ld mismatches of %d)\n", b, mism == 0 ? "PASS" : "FAIL", mism, kN);
    if (mism) ok = false;

    delete[] cpu;
    delete[] gpu;
  }

  // FSS property via GPU outputs: y0 + y1 = beta at alpha, else 0 (Uint group = wrapping add)
  {
    int4 *g0 = new int4[kN];
    int4 *g1 = new int4[kN];
    fss::gpu::HalfTreeDpfEvalAllGpu<16>(false, s0s[0], d_ht_cws, h_ht_ocw, d_ht_ys, ht);
    CHECK(cudaMemcpy(g0, d_ht_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));
    fss::gpu::HalfTreeDpfEvalAllGpu<16>(true, s0s[1], d_ht_cws, h_ht_ocw, d_ht_ys, ht);
    CHECK(cudaMemcpy(g1, d_ht_ys, sizeof(int4) * kN, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < kN; ++i) {
      auto s0 = Group::From(g0[i]);
      auto s1 = Group::From(g1[i]);
      auto sum = s0 + s1;
      bool at_alpha = (uint)i == (alpha & (kN - 1));
      bool correct = at_alpha ? (sum.Into().x == beta.x && sum.Into().y == beta.y) : (sum.Into().x == 0 && sum.Into().y == 0);
      if (!correct) ++bad;
    }
    printf("HalfTree FSS property (y0+y1==beta at alpha else 0): %s (%d bad of %d)\n", bad == 0 ? "PASS" : "FAIL", bad, kN);
    if (bad) ok = false;
    delete[] g0;
    delete[] g1;
  }

  printf(ok ? "ALL PASS\n" : "FAILURES\n");
  return ok ? 0 : 1;
}
