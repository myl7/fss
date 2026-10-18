// SPDX-License-Identifier: Apache-2.0
// Compile this unchanged source with v1.2.0 and candidate installed headers.
// The digest includes named initialized fields, never C++ padding.
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include <fss/hash/blake3.cuh>
#include <cstdlib>

namespace {

struct Digest {
  uint64_t state = UINT64_C(14695981039346656037);
  void Word(uint32_t word) {
    for (int shift = 0; shift < 32; shift += 8) {
      state ^= (word >> shift) & 255;
      state *= UINT64_C(1099511628211);
    }
  }
  void Block(int4 block) {
    Word(static_cast<uint32_t>(block.x));
    Word(static_cast<uint32_t>(block.y));
    Word(static_cast<uint32_t>(block.z));
    Word(static_cast<uint32_t>(block.w));
  }
  template <typename Cw>
  void Correction(const Cw &cw) {
    Block(cw.s);
    if constexpr (requires { cw.v; }) Block(cw.v);
    if constexpr (requires { cw.tr; }) Word(cw.tr);
    if constexpr (requires { cw.extra; }) Word(cw.extra);
  }
  template <size_t count>
  void Proof(const cuda::std::array<int4, count> &proof) {
    for (auto block : proof) Block(block);
  }
  void Print(const char *name) const { std::printf("%s %016llx\n", name, static_cast<unsigned long long>(state)); }
};

constexpr int kBits = 8;
constexpr int kCount = 1 << kBits;
constexpr uint16_t kAlpha = 107;
constexpr int4 kBeta = {604, 20, 0, 0};
constexpr int4 kHashKey = {1, 2, 3, 4};
constexpr int4 kSeeds[2] = {{12345, 87, 6543, 42}, {8765, 4321, 111, 222}};
constexpr int4 kIv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
int kNonce[2] = {123, 456};

int4 Seed(int round, int bucket, int party) {
  return {round * 12345 + bucket * 87 + party * 765,
      round * 23 + bucket * 197 + party * 43, round * 31 + bucket * 79 + party * 29,
      (round * 17 + bucket * 53 + party * 19) & ~1};
}

template <typename Group>
void DpfAndDcf(const char *dpf_name, const char *dcf_lt_name, const char *dcf_gt_name) {
  using Dpf = fss::Dpf<kBits, Group, fss::prg::ChaCha<2>, uint16_t, 0>;
  Dpf dpf{fss::prg::ChaCha<2>{kNonce}};
  typename Dpf::Cw cws[kBits + 1];
  dpf.Gen(cws, kSeeds, kAlpha, kBeta);
  Digest digest;
  for (const auto &cw : cws) digest.Correction(cw);
  int4 ys[kCount];
  for (bool party : {false, true}) {
    dpf.EvalAll(party, kSeeds[party], cws, ys);
    for (int x = 0; x < kCount; ++x) {
      digest.Block(dpf.Eval(party, kSeeds[party], cws, x));
      digest.Block(ys[x]);
    }
  }
  digest.Print(dpf_name);

  auto dcf_case = [&]<fss::DcfPred pred>(const char *name) {
    using Dcf = fss::Dcf<kBits, Group, fss::prg::ChaCha<4>, uint16_t, pred, 0>;
    Dcf dcf{fss::prg::ChaCha<4>{kNonce}};
    typename Dcf::Cw words[kBits + 1];
    dcf.Gen(words, kSeeds, kAlpha, kBeta);
    Digest d;
    for (const auto &cw : words) d.Correction(cw);
    for (bool party : {false, true}) {
      dcf.EvalAll(party, kSeeds[party], words, ys);
      for (int x = 0; x < kCount; ++x) {
        d.Block(dcf.Eval(party, kSeeds[party], words, x));
        d.Block(ys[x]);
      }
    }
    d.Print(name);
  };
  dcf_case.template operator()<fss::DcfPred::kLt>(dcf_lt_name);
  dcf_case.template operator()<fss::DcfPred::kGt>(dcf_gt_name);
}

template <typename Group>
void HalfTree(const char *name) {
  using Scheme = fss::HalfTreeDpf<kBits, Group, fss::prg::ChaCha<1>, uint16_t, 0>;
  Scheme scheme{fss::prg::ChaCha<1>{kNonce}, kHashKey};
  typename Scheme::Cw cws[kBits];
  int4 ocw;
  scheme.Gen(cws, ocw, kSeeds, kAlpha, kBeta);
  Digest digest;
  for (const auto &cw : cws) digest.Correction(cw);
  digest.Block(ocw);
  int4 ys[kCount];
  for (bool party : {false, true}) {
    scheme.EvalAll(party, kSeeds[party], cws, ocw, ys);
    for (int x = 0; x < kCount; ++x) {
      digest.Block(scheme.Eval(party, kSeeds[party], cws, ocw, x));
      digest.Block(ys[x]);
    }
  }
  digest.Print(name);
}

void Grotto() {
  using Scheme = fss::GrottoDcf<kBits, fss::prg::ChaCha<2>, uint16_t, 0>;
  Scheme scheme{fss::prg::ChaCha<2>{kNonce}};
  Scheme::Cw cws[kBits + 1];
  scheme.Gen(cws, kSeeds, kAlpha);
  Digest digest;
  for (const auto &cw : cws) digest.Correction(cw);
  bool storage[2 * kCount - 1], ys[kCount];
  for (bool party : {false, true}) {
    Scheme::ParityTree tree{storage, party};
    scheme.Preprocess(tree, kSeeds[party], cws);
    scheme.EvalAll(party, kSeeds[party], cws, ys);
    for (bool bit : storage) digest.Word(bit);
    for (int x = 0; x < kCount; ++x) {
      digest.Word(scheme.Eval(tree, x));
      digest.Word(ys[x]);
    }
  }
  digest.Print("grotto");
}

void Vdpf() {
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdpf<kBits, fss::group::Bytes, fss::prg::ChaCha<2>, Hash, Hash, uint16_t, 0>;
  Scheme scheme{fss::prg::ChaCha<2>{kNonce}, Hash{kIv}, Hash{kIv}};
  Scheme::Cw cws[kBits];
  cuda::std::array<int4, 4> cs;
  int4 ocw;
  int4 seeds[2];
  int result = 1;
  for (int round = 0; round < 1000 && result != 0; ++round) {
    seeds[0] = Seed(round, 0, 0);
    seeds[1] = Seed(round, 0, 1);
    result = scheme.Gen(cws, cs, ocw, seeds, kAlpha, kBeta);
  }
  if (result != 0) std::abort();
  Digest digest;
  for (const auto &cw : cws) digest.Correction(cw);
  digest.Proof(cs);
  digest.Block(ocw);
  int4 ys[kCount];
  std::vector<cuda::std::array<int4, 4>> point_proofs(kCount);
  for (bool party : {false, true}) {
    cuda::std::array<int4, 4> pi;
    scheme.EvalAll(party, seeds[party], cws, cs, ocw, ys, pi);
    digest.Proof(pi);
    for (int x = 0; x < kCount; ++x) {
      int4 y;
      point_proofs[x] = scheme.Eval(party, seeds[party], cws, cs, ocw, x, y);
      digest.Block(y);
      digest.Block(ys[x]);
      digest.Proof(point_proofs[x]);
    }
    scheme.Prove(point_proofs, cs, pi);
    digest.Proof(pi);
  }
  digest.Print("vdpf-chain");
}

void Vdmpf() {
  using Hash = fss::hash::Blake3;
  using Scheme = fss::Vdmpf<kBits, 30, 6, fss::group::Bytes, fss::prg::ChaCha<2>, Hash, Hash,
      fss::prp::Aes128Feistel, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<2>{kNonce}, Hash{kIv}, Hash{kIv}, {}};
  Scheme::Key k0, k1;
  uint16_t alphas[30];
  int4 betas[30];
  for (int i = 0; i < 30; ++i) {
    alphas[i] = i * 3;
    betas[i] = {i * 11, 0, 0, 0};
  }
  cuda::std::array<cuda::std::array<int4, 2>, Scheme::m> seeds;
  int result = 1;
  for (int round = 0; round < 1000 && result != 0; ++round) {
    int4 sigma = Seed(round, 3, 0);
    for (int i = 0; i < Scheme::m; ++i) seeds[i] = {Seed(round, i, 0), Seed(round, i, 1)};
    result = scheme.Gen(k0, k1, sigma, cuda::std::span<const cuda::std::array<int4, 2>, Scheme::m>(seeds),
        std::span<const uint16_t>(alphas), std::span<const int4>(betas), 30);
  }
  if (result != 0) std::abort();
  Digest digest;
  std::vector<uint16_t> xs(kCount);
  for (int i = 0; i < kCount; ++i) xs[i] = i;
  xs.push_back(kAlpha);
  std::vector<int4> ys(xs.size());
  for (const auto *key : {&k0, &k1}) {
    digest.Block(key->sigma);
    digest.Word(key->m_rt);
    digest.Word(key->b_size_rt);
    for (const typename Scheme::BucketKey &bucket : key->bks) {
      for (const auto &cw : bucket.cws) digest.Correction(cw);
      digest.Proof(bucket.cs);
      digest.Block(bucket.ocw);
      digest.Block(bucket.s0);
    }
    cuda::std::array<int4, 4> pi;
    scheme.BatchEval(key == &k1, *key, std::span<const uint16_t>(xs), std::span<int4>(ys), pi);
    for (auto y : ys) digest.Block(y);
    digest.Proof(pi);
  }
  digest.Print("vdmpf-chain");
}

}  // namespace

int main() {
  DpfAndDcf<fss::group::Bytes>("dpf-bytes", "dcf-lt-bytes", "dcf-gt-bytes");
  DpfAndDcf<fss::group::Uint<uint64_t>>("dpf-uint64", "dcf-lt-uint64", "dcf-gt-uint64");
  HalfTree<fss::group::Bytes>("half-tree-bytes");
  HalfTree<fss::group::Uint<uint64_t>>("half-tree-uint64");
  Grotto();
  Vdpf();
  Vdmpf();
}
