// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <fss/hash/aes128_ccr.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <vector>

namespace fss_test {

struct PrgCallCounts {
  int full = 0;
  int half = 0;
  int right = 0;
};

template <typename Backend, int mul, bool has_half>
class CountedPrg {
public:
  __host__ __device__ CountedPrg(Backend backend, PrgCallCounts &counts)
      : backend_(backend), counts_(&counts) {}

  __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
    ++counts_->full;
    return backend_.Gen(seed);
  }

  __host__ __device__ cuda::std::array<int4, mul / 2> GenHalf(int4 seed, bool right)
    requires(has_half) {
    ++counts_->half;
    counts_->right += static_cast<int>(right);
    return backend_.GenHalf(seed, right);
  }

private:
  Backend backend_;
  PrgCallCounts *counts_;
};

template <int mul>
struct GenHalfAesFixture {
  uint8_t keys[mul][16] = {};
  uint32_t te0[256];
  uint8_t sbox[256];

  GenHalfAesFixture() {
    for (int j = 0; j < mul; ++j) {
      for (int i = 0; i < 16; ++i) keys[j][i] = static_cast<uint8_t>(17 * j + i);
    }
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
  }

  fss::prg::Aes128Soft<mul> Backend() {
    return fss::prg::Aes128Soft<mul>(keys, te0, sbox);
  }
};

template <int mul>
struct GenHalfDeviceAesFixture {
  GenHalfAesFixture<mul> host;
  uint32_t *te0 = nullptr;
  uint8_t *sbox = nullptr;

  ~GenHalfDeviceAesFixture() {
    if (sbox) cudaFree(sbox);
    if (te0) cudaFree(te0);
  }

  cudaError_t Allocate() {
    auto status = cudaMallocManaged(&te0, 256 * sizeof(uint32_t));
    if (status != cudaSuccess) return status;
    status = cudaMallocManaged(&sbox, 256);
    if (status != cudaSuccess) return status;
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
    return cudaSuccess;
  }

  fss::prg::Aes128Soft<mul> Backend() {
    return fss::prg::Aes128Soft<mul>(host.keys, te0, sbox);
  }
};

struct EvalResult {
  int4 y;
  cuda::std::array<int4, 4> pi_tilde;
};

// Owned tables keep this test fixture self-contained when captured by a kernel.
class TestCcrHash {
public:
  __host__ __device__ TestCcrHash(int nonce0 = 0x12345678, int nonce1 = 0x13572468) {
    for (int i = 0; i < 16; ++i) {
      unsigned word = static_cast<unsigned>(i < 8 ? nonce0 : nonce1);
      key_[0][i] = static_cast<uint8_t>(word >> ((i % 4) * 8));
    }
    fss::prg::aes_detail::InitTe0(te0_);
    fss::prg::aes_detail::InitSbox(sbox_);
  }

  __host__ __device__ int4 Hash(int4 block) {
    fss::prg::Aes128Soft<1> aes(key_, te0_, sbox_);
    fss::hash::Aes128Ccr hash(aes);
    return hash.Hash(block);
  }

private:
  uint8_t key_[1][16];
  uint32_t te0_[256];
  uint8_t sbox_[256];
};

template <typename Backend>
class CcrAdapter : public fss::hash::Aes128Ccr<Backend> {
public:
  __host__ __device__ explicit CcrAdapter(Backend backend) : fss::hash::Aes128Ccr<Backend>(backend) {}
};

template <typename Backend, bool = CcrHashable<Backend>>
struct CcrBackendType {
  using type = CcrAdapter<Backend>;
};

template <typename Backend>
struct CcrBackendType<Backend, true> {
  using type = Backend;
};

template <typename Backend>
using CcrBackend = typename CcrBackendType<Backend>::type;

template <typename Backend>
__host__ __device__ CcrBackend<Backend> Ccr(Backend backend) {
  if constexpr (CcrHashable<Backend>) {
    return backend;
  } else {
    return CcrAdapter<Backend>(backend);
  }
}


template <typename Buffer>
__host__ __device__ auto Data(Buffer &buffer) {
  if constexpr (requires { buffer.data(); }) return buffer.data();
  else return buffer;
}

template <typename Scheme, typename Seeds>
__host__ __device__ void AssemblePair(typename Scheme::Key &k0, typename Scheme::Key &k1, Seeds &seeds) {
  auto data = Data(seeds);
  k0.s0 = data[0];
  k1.s0 = data[1];
  constexpr size_t count = std::extent_v<decltype(k0.cws)>;
  for (size_t i = 0; i < count; ++i) k1.cws[i] = k0.cws[i];
  if constexpr (requires { k0.cs; }) {
    for (int i = 0; i < 4; ++i) k1.cs[i] = k0.cs[i];
  }
}

// Fixture assembly performs one Core Gen attempt and preserves its status.
template <typename Scheme, typename Seeds, typename In, typename Beta = int4>
__host__ __device__ decltype(auto) GenPair(Scheme &scheme, typename Scheme::Key &k0,
    typename Scheme::Key &k1, Seeds &&seeds, In alpha, Beta beta) {
  if constexpr (requires { k0.cs; }) {
    int ret = scheme.Gen(k0.cws, cuda::std::span<int4, 4>(k0.cs, 4),
        cuda::std::span<const int4, 2>(Data(seeds), 2), alpha, beta);
    if (ret == 0) AssemblePair<Scheme>(k0, k1, seeds);
    return ret;
  } else {
    scheme.Gen(k0.cws, Data(seeds), alpha, beta);
    AssemblePair<Scheme>(k0, k1, seeds);
  }
}

template <typename Scheme, typename Seeds, typename In>
__host__ __device__ void GenPair(Scheme &scheme, typename Scheme::Key &k0,
    typename Scheme::Key &k1, Seeds &&seeds, In alpha) {
  scheme.Gen(k0.cws, Data(seeds), alpha);
  AssemblePair<Scheme>(k0, k1, seeds);
}

template <typename Scheme, typename Seeds, typename Alphas, typename Betas>
int GenMulti(Scheme &scheme, typename Scheme::Key &k0, typename Scheme::Key &k1,
    int4 &sigma, Seeds &&seeds, Alphas &&alphas, Betas &&betas, int count,
    int max_evictions = 1000) {
  constexpr size_t kWords = std::extent_v<decltype(k0.bks[0].cws)>;
  std::vector<typename Scheme::Cw> cw_storage(size_t{Scheme::kBucketCount} * kWords);
  auto cws = cuda::std::span<typename Scheme::Cw>(cw_storage.data(), cw_storage.size());
  std::vector<int4> checks;
  auto seed_view = cuda::std::span<const int4>(Data(seeds), size_t{Scheme::kBucketCount} * 2);
  auto alpha_view = cuda::std::span<const std::remove_cvref_t<decltype(Data(alphas)[0])>>(Data(alphas), count);
  auto beta_view = cuda::std::span<const int4>(Data(betas), count);
  int ret;
  if constexpr (requires { k0.bks[0].cs; }) {
    checks.resize(size_t{Scheme::kBucketCount} * 4);
    auto cs = cuda::std::span<int4>(checks.data(), checks.size());
    ret = scheme.Gen(cws, cs, k0.m_rt, k0.b_size_rt, sigma,
        seed_view, alpha_view, beta_view, count, max_evictions);
  } else {
    ret = scheme.Gen(cws, k0.m_rt, k0.b_size_rt, sigma,
        seed_view, alpha_view, beta_view, count, max_evictions);
  }
  if (ret == 0) {
    k0.sigma = k1.sigma = sigma;
    k1.m_rt = k0.m_rt;
    k1.b_size_rt = k0.b_size_rt;
    for (int i = 0; i < Scheme::kBucketCount; ++i) {
      k0.bks[i].s0 = seed_view[i];
      k1.bks[i].s0 = seed_view[size_t{Scheme::kBucketCount} + i];
      for (size_t j = 0; j < std::extent_v<decltype(k0.bks[i].cws)>; ++j) {
        k0.bks[i].cws[j] = k1.bks[i].cws[j] = cw_storage[static_cast<size_t>(i) * kWords + j];
      }
      if constexpr (requires { k0.bks[i].cs; }) {
        for (int j = 0; j < 4; ++j) k0.bks[i].cs[j] = k1.bks[i].cs[j] = checks[static_cast<size_t>(i) * 4 + j];
      }
    }
  }
  return ret;
}

template <typename Scheme, typename Key, typename In>
__host__ __device__ auto EvalKey(Scheme &scheme, bool party, const Key &key, In x) {
  if constexpr (requires { key.cs; }) {
    EvalResult result;
    result.pi_tilde = scheme.Eval(party, key.s0, key.cws, key.cs, x, result.y);
    return result;
  } else {
    return scheme.Eval(party, key.s0, key.cws, x);
  }
}

template <typename Scheme>
struct MultiRows {
  cuda::std::array<int4, Scheme::kBucketCount> s0s;
  std::vector<typename Scheme::Cw> cw_storage;
  std::vector<int4> checks;
  cuda::std::span<const typename Scheme::Cw> cws;
  cuda::std::span<const int4> cs;

  explicit MultiRows(const typename Scheme::Key &key) {
    constexpr size_t kWords = std::extent_v<decltype(key.bks[0].cws)>;
    cw_storage.resize(size_t{Scheme::kBucketCount} * kWords);
    if constexpr (requires { key.bks[0].cs; }) checks.resize(size_t{Scheme::kBucketCount} * 4);
    for (int i = 0; i < Scheme::kBucketCount; ++i) {
      s0s[i] = key.bks[i].s0;
      for (size_t j = 0; j < kWords; ++j) cw_storage[static_cast<size_t>(i) * kWords + j] = key.bks[i].cws[j];
      if constexpr (requires { key.bks[i].cs; }) {
        for (int j = 0; j < 4; ++j) checks[static_cast<size_t>(i) * 4 + j] = key.bks[i].cs[j];
      }
    }
    cws = {cw_storage.data(), cw_storage.size()};
    cs = {checks.data(), checks.size()};
  }
};

template <int par_depth = 0, typename Scheme, typename Key, typename Output, typename... Proof>
void EvalAllKey(Scheme &scheme, bool party, const Key &key, Output &&output, Proof &...proof) {
  if constexpr (requires { key.bks; }) {
    MultiRows<Scheme> parts(key);
    scheme.EvalAll(party, key.sigma, key.b_size_rt, parts.s0s, parts.cws, output, par_depth);
  } else if constexpr (requires { key.cs; }) {
    scheme.EvalAll(party, key.s0, key.cws, key.cs, output, proof..., par_depth);
  } else {
    scheme.EvalAll(party, key.s0, key.cws, Data(output), par_depth);
  }
}

template <int par_depth = 0, typename Scheme, typename Key, typename Output, typename Proof>
void EvalAllMerkleKey(Scheme &scheme, bool party, const Key &key, Output &&output, Proof &proof) {
  scheme.EvalAllMerkle(party, key.s0, key.cws, key.cs, output, proof, par_depth);
}

template <int par_depth = 0, typename Scheme, typename Key, typename Output, typename Proof>
void EvalRangeKey(Scheme &scheme, bool party, size_t left, size_t right, const Key &key,
    Output &&output, Proof &proof) {
  scheme.EvalRange(party, left, right, key.s0, key.cws, key.cs, output, proof, par_depth);
}

template <typename Scheme, typename Key, typename Xs, typename Ys, typename... Proof>
void BatchEvalKey(Scheme &scheme, bool party, const Key &key, Xs &&xs, Ys &&ys, Proof &...proof) {
  MultiRows<Scheme> parts(key);
  if constexpr (requires { key.bks[0].cs; }) {
    scheme.BatchEval(party, key.sigma, key.b_size_rt,
        parts.s0s, parts.cws, parts.cs, xs, ys, proof...);
  } else {
    scheme.BatchEval(party, key.sigma, key.b_size_rt, parts.s0s, parts.cws, xs, ys);
  }
}

template <typename Scheme, typename Key, typename Xs, typename Ys, typename Proof>
void BatchEvalMerkleKey(Scheme &scheme, bool party, const Key &key, Xs &&xs, Ys &&ys, Proof &proof) {
  MultiRows<Scheme> parts(key);
  scheme.BatchEvalMerkle(party, key.sigma, key.b_size_rt,
      parts.s0s, parts.cws, parts.cs, xs, ys, proof);
}

template <int par_depth = 0, typename Scheme, typename Key>
auto PreprocessKey(Scheme &scheme, bool party, const Key &key) {
  return scheme.PreprocessOwned(party, key.s0, key.cws, par_depth);
}

template <int width, typename In>
int4 PackBetaReference(In x, uint64_t beta) {
  int4 block{};
  unsigned int words[4]{};
  size_t lane = static_cast<size_t>(x) % (128 / width);
  for (int bit = 0; bit < width; ++bit) {
    if ((beta >> bit) & 1) {
      size_t at = lane * width + bit;
      words[at / 32] |= 1u << (at % 32);
    }
  }
  block.x = static_cast<int>(words[0]);
  block.y = static_cast<int>(words[1]);
  block.z = static_cast<int>(words[2]);
  block.w = static_cast<int>(words[3]);
  return block;
}

}  // namespace fss_test
