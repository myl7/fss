// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <fss/hash/aes128_ccr.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <type_traits>
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


// nvcc reports #20011-D and #20014-D here because host tests pass std::vector to the
// __host__ __device__ GenPair helpers, which instantiates Data() with a host-only data().
// No device code passes a host container, so the warning is a known false alarm.
// A new #20011-D or #20014-D elsewhere is not this one.
// If this one changes, check whether device code started passing a host container.
template <typename Buffer>
__host__ __device__ auto Data(Buffer &buffer) {
  if constexpr (requires { buffer.data(); }) return buffer.data();
  else return buffer;
}

// Single-point key fixture in the component layout: contiguous strided
// correction words plus packed control bits. Schemes whose Cw carries no
// right-control bit (PackedHalfTreeDpf, HalfTreeGrottoDcf) leave tr zero.
// cs exists in every fixture and only the verifiable schemes write it.
template <typename Scheme, typename In = typename Scheme::Input>
struct ComponentKey {
  static constexpr size_t kCwWords = static_cast<size_t>(Scheme::kCwCount);
  int4 cws[kCwWords];
  // Value-correction components, written only by schemes whose Cw carries a
  // group-value word (Dcf); its right control lives in packed tr instead.
  int4 vws[kCwWords];
  In tr = 0;
  int4 s0;
  int4 cs[4];
};

// True when the scheme's correction words carry a separate group-value word.
template <typename Scheme>
inline constexpr bool kHasValueWord = requires { std::declval<typename Scheme::Cw &>().v; };

// True when the scheme's correction words carry a packed control component,
// so its component APIs take a control argument. Dpf, Dcf, Vdpf, and GrottoDcf
// carry the right-control bit in Cw::tr, HalfTreeDpf carries its final-level
// control in Cw::extra, and PackedHalfTreeDpf and HalfTreeGrottoDcf carry only
// the seed component.
template <typename Scheme>
inline constexpr bool kHasControlBit = requires { std::declval<typename Scheme::Cw &>().tr; } ||
    requires { std::declval<typename Scheme::Cw &>().extra; };

// True for the verifiable single-point scheme (Vdpf), whose Gen and Eval
// exchange the shared correction seed cs.
template <typename Scheme>
inline constexpr bool kIsVerifiable = requires { &Scheme::Verify; };

template <typename Scheme, typename In>
__host__ __device__ cuda::std::span<int4> KeyCws(ComponentKey<Scheme, In> &key) {
  return cuda::std::span<int4>(key.cws, ComponentKey<Scheme, In>::kCwWords);
}

template <typename Scheme, typename In>
__host__ __device__ cuda::std::span<const int4> KeyCws(const ComponentKey<Scheme, In> &key) {
  return cuda::std::span<const int4>(key.cws, ComponentKey<Scheme, In>::kCwWords);
}

template <typename Scheme, typename In>
__host__ __device__ cuda::std::span<int4> KeyVws(ComponentKey<Scheme, In> &key) {
  return cuda::std::span<int4>(key.vws, ComponentKey<Scheme, In>::kCwWords);
}

template <typename Scheme, typename In>
__host__ __device__ cuda::std::span<const int4> KeyVws(const ComponentKey<Scheme, In> &key) {
  return cuda::std::span<const int4>(key.vws, ComponentKey<Scheme, In>::kCwWords);
}

template <typename Scheme, typename In, typename Seeds>
__host__ __device__ void AssemblePair(ComponentKey<Scheme, In> &k0,
    ComponentKey<Scheme, In> &k1, Seeds &seeds) {
  auto data = Data(seeds);
  k0.s0 = data[0];
  k1.s0 = data[1];
  k1.tr = k0.tr;
  for (size_t i = 0; i < ComponentKey<Scheme, In>::kCwWords; ++i) {
    k1.cws[i] = k0.cws[i];
    k1.vws[i] = k0.vws[i];
  }
  for (int i = 0; i < 4; ++i) k1.cs[i] = k0.cs[i];
}

// Fixture assembly performs one Core Gen attempt and preserves its status.
template <typename Scheme, typename Seeds, typename In, typename Beta = int4>
__host__ __device__ decltype(auto) GenPair(Scheme &scheme, ComponentKey<Scheme, In> &k0,
    ComponentKey<Scheme, In> &k1, Seeds &&seeds, auto alpha, Beta beta) {
  const cuda::std::span<const int4, 2> seed_view(Data(seeds), 2);
  if constexpr (kIsVerifiable<Scheme>) {
    int ret = scheme.Gen(KeyCws(k0), k0.tr, cuda::std::span<int4, 4>(k0.cs, 4),
        seed_view, alpha, beta);
    if (ret == 0) AssemblePair(k0, k1, seeds);
    return ret;
  } else if constexpr (!kHasControlBit<Scheme> && !kHasValueWord<Scheme>) {
    // The packed scheme carries no control component: the payload is the only
    // extra argument, whatever integer type the caller spelled it with.
    scheme.Gen(KeyCws(k0), seed_view, alpha, static_cast<uint64_t>(beta));
    AssemblePair(k0, k1, seeds);
  } else if constexpr (kHasValueWord<Scheme>) {
    scheme.Gen(KeyCws(k0), KeyVws(k0), k0.tr, seed_view, alpha, beta);
    AssemblePair(k0, k1, seeds);
  } else {
    scheme.Gen(KeyCws(k0), k0.tr, seed_view, alpha, beta);
    AssemblePair(k0, k1, seeds);
  }
}

template <typename Scheme, typename Seeds, typename In>
__host__ __device__ void GenPair(Scheme &scheme, ComponentKey<Scheme, In> &k0,
    ComponentKey<Scheme, In> &k1, Seeds &&seeds, auto alpha) {
  const cuda::std::span<const int4, 2> seed_view(Data(seeds), 2);
  if constexpr (kHasControlBit<Scheme>) {
    scheme.Gen(KeyCws(k0), k0.tr, seed_view, alpha);
  } else {
    scheme.Gen(KeyCws(k0), seed_view, alpha);
  }
  AssemblePair(k0, k1, seeds);
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

template <typename Scheme, typename In>
__host__ __device__ auto EvalKey(Scheme &scheme, bool party, const ComponentKey<Scheme, In> &key, auto x) {
  if constexpr (kIsVerifiable<Scheme>) {
    EvalResult result;
    result.pi_tilde = scheme.Eval(party, key.s0, KeyCws(key), key.tr,
        cuda::std::span<const int4, 4>(key.cs, 4), x, result.y);
    return result;
  } else if constexpr (kHasValueWord<Scheme>) {
    return scheme.Eval(party, key.s0, KeyCws(key), KeyVws(key), key.tr, x);
  } else if constexpr (kHasControlBit<Scheme>) {
    return scheme.Eval(party, key.s0, KeyCws(key), key.tr, x);
  } else {
    return scheme.Eval(party, key.s0, KeyCws(key), x);
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
  } else if constexpr (kIsVerifiable<Scheme>) {
    scheme.EvalAll(party, key.s0, KeyCws(key), key.tr,
        cuda::std::span<const int4, 4>(key.cs, 4), output, proof..., par_depth);
  } else if constexpr (kHasValueWord<Scheme>) {
    scheme.EvalAll(party, key.s0, KeyCws(key), KeyVws(key), key.tr, Data(output), par_depth);
  } else if constexpr (kHasControlBit<Scheme>) {
    scheme.EvalAll(party, key.s0, KeyCws(key), key.tr, Data(output), par_depth);
  } else {
    scheme.EvalAll(party, key.s0, KeyCws(key), Data(output), par_depth);
  }
}

template <int par_depth = 0, typename Scheme, typename In, typename Output, typename Proof>
void EvalAllMerkleKey(Scheme &scheme, bool party, const ComponentKey<Scheme, In> &key,
    Output &&output, Proof &proof) {
  scheme.EvalAllMerkle(party, key.s0, KeyCws(key), key.tr,
      cuda::std::span<const int4, 4>(key.cs, 4), output, proof, par_depth);
}

template <int par_depth = 0, typename Scheme, typename In, typename Output, typename Proof>
void EvalRangeKey(Scheme &scheme, bool party, auto left, auto right,
    const ComponentKey<Scheme, In> &key, Output &&output, Proof &proof) {
  scheme.EvalRange(party, left, right, key.s0, KeyCws(key), key.tr,
      cuda::std::span<const int4, 4>(key.cs, 4), output, proof, par_depth);
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

template <int par_depth = 0, typename Scheme, typename In>
auto PreprocessKey(Scheme &scheme, bool party, const ComponentKey<Scheme, In> &key) {
  if constexpr (kHasControlBit<Scheme>) {
    return scheme.PreprocessOwned(party, key.s0, KeyCws(key), key.tr, par_depth);
  } else {
    return scheme.PreprocessOwned(party, key.s0, KeyCws(key), par_depth);
  }
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
