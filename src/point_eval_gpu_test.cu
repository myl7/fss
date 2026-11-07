#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda/std/span>
#include <fss/group/bytes.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/dcf.cuh>
#include <fss/dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/vdpf.cuh>
#include "core_api_test_helpers.cuh"

namespace {

template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) EXPECT_EQ(cudaFree(data_), cudaSuccess);
  }
  cudaError_t Allocate(size_t count) {
    return cudaMalloc(&data_, count * sizeof(T));
  }
  cudaError_t Upload(const T *source, size_t count) {
    return cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice);
  }
  T *data() const {
    return data_;
  }

private:
  T *data_ = nullptr;
};

void ExpectInt4(int4 actual, int4 expected, int key) {
  EXPECT_EQ(actual.x, expected.x) << "key " << key;
  EXPECT_EQ(actual.y, expected.y) << "key " << key;
  EXPECT_EQ(actual.z, expected.z) << "key " << key;
  EXPECT_EQ(actual.w, expected.w) << "key " << key;
}

// Alphas keep the top input bit set so wide-In kernels run their widened bit
// extraction, not just the low 32 bits.
template <int bits, typename In>
std::vector<In> MakeAlphas(int keys) {
  std::vector<In> alphas(keys);
  for (int k = 0; k < keys; ++k) {
    alphas[k] = static_cast<In>(static_cast<In>(1) << (bits - 1)) + static_cast<In>(42 + k);
  }
  return alphas;
}

// Key 0 queries its own alpha; the rest spread over high and low halves.
template <int bits, typename In>
std::vector<In> MakePoints(const std::vector<In> &alphas) {
  std::vector<In> xs(alphas.size());
  for (size_t k = 0; k < alphas.size(); ++k) {
    In high = static_cast<In>(static_cast<In>(k % 2) << (bits - 1));
    xs[k] = k == 0 ? alphas[0] : static_cast<In>(high + static_cast<In>(3 * k + 1));
  }
  return xs;
}

template <int bits, typename In, int keys = 8, typename Query = In>
void CheckDpfPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Group = fss::group::Bytes;
  using Scheme = fss::Dpf<bits, Group, Prg, In>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<int4> cws(static_cast<size_t>(bits + 1) * keys);
  std::vector<In> trs(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    // Split form: layer i of key k lands at cws[i * keys + k], the terminal at
    // cws[bits * keys + k], and the packed right-control bits in trs[k].
    scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(bits) * keys + 1), trs[k],
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta, keys);
  }
  DeviceBuffer<int4> device_cws, device_seeds, device_ys;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::DpfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), static_cast<size_t>(bits + 1) * keys),
        cuda::std::span<const In>(device_tr.data(), keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(),
        keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], fss_test::EvalKey(scheme, party != 0, party_key, xs[k]), k);
      if (party == 0) first[k] = actual[k];
      else {
        int4 sum = (Group::From(first[k]) + Group::From(actual[k])).Into();
        ExpectInt4(sum, xs[k] == alphas[k] ? beta : int4{0, 0, 0, 0}, k);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

template <int bits, typename In, int keys = 8, typename Query = In>
void CheckHalfTreePoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss_test::TestCcrHash;
  using Group = fss::group::Bytes;
  using Scheme = fss::HalfTreeDpf<bits, Group, fss_test::CcrBackend<Prg>, In>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  const int4 hash_key = {11, 22, 33, 44};
  Scheme scheme{fss_test::Ccr(Prg(0x12345678, static_cast<int>(0x9abcdef0u))), hash_key};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<int4> cws(static_cast<size_t>(bits + 1) * keys);
  std::vector<In> trs(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    // Split form: layer i of key k lands at cws[i * keys + k], the terminal at
    // cws[bits * keys + k], and the packed control bits in trs[k].
    scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(bits) * keys + 1), trs[k],
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta, keys);
  }
  DeviceBuffer<int4> device_cws, device_seeds, device_ys;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::HalfTreeDpfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), static_cast<size_t>(bits + 1) * keys),
        cuda::std::span<const In>(device_tr.data(), keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(),
        keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], fss_test::EvalKey(scheme, party != 0, party_key, xs[k]), k);
      if (party == 0) first[k] = actual[k];
      else {
        int4 sum = (Group::From(first[k]) + Group::From(actual[k])).Into();
        ExpectInt4(sum, xs[k] == alphas[k] ? beta : int4{0, 0, 0, 0}, k);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Boundary points per key: alpha, its neighbors, alpha with the last tree bit flipped (the other
// block of alpha's lane), alpha with the top bit flipped, and a point in the other half.
template <int bits, typename In, int depth>
std::vector<In> MakeBoundaryPoints(const std::vector<In> &alphas) {
  const __uint128_t mask = bits == 128 ? ~static_cast<__uint128_t>(0) : (static_cast<__uint128_t>(1) << bits) - 1;
  std::vector<In> xs(alphas.size());
  for (size_t k = 0; k < alphas.size(); ++k) {
    const __uint128_t a = alphas[k];
    const __uint128_t candidates[6] = {a, a - 1, a + 1, a ^ (static_cast<__uint128_t>(1) << (bits - depth)),
        a ^ (static_cast<__uint128_t>(1) << (bits - 1)), static_cast<__uint128_t>(3 * k + 1)};
    xs[k] = static_cast<In>(candidates[k % 6] & mask);
  }
  return xs;
}

// Packed HalfTree DPF point evaluation against the host Eval and the two-party reconstruction.
template <int bits, typename In, int out_bits, int keys = 37, typename Query = In>
void CheckPackedPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::PackedHalfTreeDpf<bits, out_bits, fss_test::CcrBackend<Prg>, In>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  constexpr int kDepth = Scheme::kDepth;
  Scheme scheme{fss_test::Ccr(Prg(0x12345678, static_cast<int>(0x9abcdef0u))), {11, 22, 33, 44}};
  const uint64_t beta = out_bits == 1 ? 1 : (out_bits == 64 ? ~uint64_t{0} : (uint64_t{1} << out_bits) - 1) - 2;
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakeBoundaryPoints<bits, In, kDepth>(alphas);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<int4> cws(static_cast<size_t>(kDepth + 1) * keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    // Split form: layer i of key k lands at cws[i * keys + k] and the final
    // correction word at cws[kDepth * keys + k].
    scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(kDepth) * keys + 1),
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta, keys);
  }
  DeviceBuffer<int4> device_cws, device_seeds, device_ys;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::PackedHalfTreeDpfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), static_cast<size_t>(kDepth + 1) * keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(), keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], fss_test::EvalKey(scheme, party != 0, party_key, xs[k]), k);
      if (party == 0) {
        first[k] = actual[k];
      } else {
        const bool same_block = (xs[k] >> (bits - kDepth)) == (alphas[k] >> (bits - kDepth));
        const int4 expected = same_block ? fss_test::PackBetaReference<128 / Scheme::kLanes>(alphas[k], beta) : int4{0, 0, 0, 0};
        ExpectInt4(fss::util::Xor(first[k], actual[k]), expected, k);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// HalfTree Grotto DCF point evaluation against the host Eval and the two-party reconstruction.
template <int bits, typename In, int keys = 37, typename Query = In>
void CheckHalfTreeGrottoPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss_test::TestCcrHash;
  using Scheme = fss::HalfTreeGrottoDcf<bits, fss_test::CcrBackend<Prg>, In>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  constexpr int kDepth = Scheme::kDepth;
  Scheme scheme{fss_test::Ccr(Prg(0x12345678, static_cast<int>(0x9abcdef0u))), {11, 22, 33, 44}};
  // Every other run of six keys flips alpha's last tree bit, so each boundary point meets both
  // leaf parity sides.
  std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  for (int k = 0; k < keys; ++k) {
    if ((k / 6) % 2 == 1) alphas[k] = static_cast<In>(alphas[k] ^ (static_cast<In>(1) << (bits - kDepth)));
  }
  const std::vector<In> xs = MakeBoundaryPoints<bits, In, kDepth>(alphas);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<int4> cws(static_cast<size_t>(kDepth + 1) * keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k]);
    // Split form: layer i of key k lands at cws[i * keys + k] and the final
    // correction word at cws[kDepth * keys + k].
    scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(kDepth) * keys + 1),
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], keys);
  }
  DeviceBuffer<int4> device_cws, device_seeds;
  DeviceBuffer<bool> device_ys;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<uint8_t> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::HalfTreeGrottoDcfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), static_cast<size_t>(kDepth + 1) * keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(), keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      EXPECT_EQ(actual[k] != 0, fss_test::EvalKey(scheme, party != 0, party_key, xs[k]));
      if (party == 0) {
        first[k] = actual[k];
      } else {
        EXPECT_EQ((first[k] != 0) != (actual[k] != 0), alphas[k] <= xs[k]);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Grotto DCF single-path evaluation against the host Eval and the two-party reconstruction.
template <int bits, typename In, int keys = 37, typename Query = In>
void CheckGrottoPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::GrottoDcf<bits, Prg, In>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakeBoundaryPoints<bits, In, bits>(alphas);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<int4> cws(static_cast<size_t>(bits + 1) * keys);
  std::vector<In> trs(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k]);
    // Split form: layer i of key k lands at cws[i * keys + k], the terminal at
    // cws[bits * keys + k], and the packed right-control bits in trs[k].
    scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(bits) * keys + 1), trs[k],
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], keys);
  }
  DeviceBuffer<int4> device_cws, device_seeds;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<bool> device_ys;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<uint8_t> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::GrottoDcfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), static_cast<size_t>(bits + 1) * keys),
        cuda::std::span<const In>(device_tr.data(), keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(),
        keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      EXPECT_EQ(actual[k] != 0, fss_test::EvalKey(scheme, party != 0, party_key, xs[k]));
      if (party == 0) {
        first[k] = actual[k];
      } else {
        EXPECT_EQ((first[k] != 0) != (actual[k] != 0), alphas[k] <= xs[k]);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

template <int bits, typename In, int keys = 8, fss::DcfPred pred = fss::DcfPred::kLt, typename Query = In>
void CheckDcfPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<4>;
  using Group = fss::group::Bytes;
  using Scheme = fss::Dcf<bits, Group, Prg, In, pred>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<int4> cws(static_cast<size_t>(bits + 1) * keys), vws(static_cast<size_t>(bits + 1) * keys);
  std::vector<In> trs(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    // Split form: layer i of key k lands at cws[i * keys + k] and vws[i * keys + k],
    // the terminal v in vws's tail row, and the right-control bits in trs[k].
    scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(bits) * keys + 1),
        cuda::std::span<int4>(vws.data() + k, static_cast<size_t>(bits) * keys + 1), trs[k],
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta, keys);
  }
  DeviceBuffer<int4> device_cws, device_vws, device_seeds, device_ys;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_vws.Allocate(vws.size()), cudaSuccess);
  ASSERT_EQ(device_vws.Upload(vws.data(), vws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::DcfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), static_cast<size_t>(bits + 1) * keys),
        cuda::std::span<const int4>(device_vws.data(), static_cast<size_t>(bits + 1) * keys),
        cuda::std::span<const In>(device_tr.data(), keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(),
        keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], fss_test::EvalKey(scheme, party != 0, party_key, xs[k]), k);
      if (party == 0) first[k] = actual[k];
      else {
        const bool hit = pred == fss::DcfPred::kLt ? xs[k] < alphas[k] : xs[k] > alphas[k];
        int4 sum = (Group::From(first[k]) + Group::From(actual[k])).Into();
        ExpectInt4(sum, hit ? beta : int4{0, 0, 0, 0}, k);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

template <int bits, typename In, int keys = 8, typename Query = In>
void CheckVdpfPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Blake3 = fss::hash::Blake3;
  using Group = fss::group::Bytes;
  using Scheme = fss::Vdpf<bits, Group, Prg, Blake3, Blake3, In>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  constexpr int kStride = bits + 1;
  // Blake3 copies the IV by value at construction.
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Blake3 hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash, hash};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  // Split form: layer i of key k lands at cws[i * keys + k], the terminal at
  // cws[bits * keys + k], and the packed right-control bits in trs[k]. cs stays
  // per-key.
  std::vector<int4> cws(static_cast<size_t>(kStride) * keys);
  std::vector<In> trs(keys);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  std::vector<cuda::std::array<int4, 4>> cs(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    ASSERT_EQ(fss_test::GenPair(scheme, keys0[k], keys1[k], cuda::std::span<int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta), 0);
    ASSERT_EQ(scheme.Gen(cuda::std::span<int4>(cws.data() + k, static_cast<size_t>(bits) * keys + 1), trs[k],
        cuda::std::span<int4, 4>(keys0[k].cs, 4),
        cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta, keys), 0);
    for (int i = 0; i < 4; ++i) cs[k][i] = keys0[k].cs[i];
  }
  DeviceBuffer<int4> device_cws;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<int4> device_seeds, device_ys;
  DeviceBuffer<cuda::std::array<int4, 4>> device_pi;
  DeviceBuffer<cuda::std::array<int4, 4>> device_cs;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_cs.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_cs.Upload(cs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_pi.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  std::vector<int4> actual(keys), first(keys), gpu_pi(2 * 4 * keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::detail::VdpfEvalPointGpu(party != 0,
        cuda::std::span<const int4>(device_seeds.data(), keys),
        cuda::std::span<const int4>(device_cws.data(), cws.size()),
        cuda::std::span<const In>(device_tr.data(), keys),
        cuda::std::span<const cuda::std::array<int4, 4>>(device_cs.data(), keys),
        cuda::std::span<const Query>(device_xs.data(), keys), device_ys.data(),
        cuda::std::span<cuda::std::array<int4, 4>>(device_pi.data(), keys), keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(gpu_pi.data() + party * 4 * keys, device_pi.data(), 4 * keys * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      auto expected = fss_test::EvalKey(scheme, party != 0, party_key, xs[k]);
      ExpectInt4(actual[k], expected.y, k);
      for (int j = 0; j < 4; ++j) {
        ExpectInt4(gpu_pi[party * 4 * keys + 4 * k + j], expected.pi_tilde[j], k);
      }
      if (party == 0) first[k] = actual[k];
      else {
        int4 sum = (Group::From(first[k]) + Group::From(actual[k])).Into();
        ExpectInt4(sum, xs[k] == alphas[k] ? beta : int4{0, 0, 0, 0}, k);
        EXPECT_TRUE(Scheme::Verify(cuda::std::span<const int4, 4>(gpu_pi.data() + 4 * k, 4),
            cuda::std::span<const int4, 4>(gpu_pi.data() + 4 * keys + 4 * k, 4)))
            << "key " << k;
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PointEvalGpu, DpfBytes) {
  CheckDpfPoint<10, uint>();
}
TEST(PointEvalGpu, DpfBoundary32) {
  CheckDpfPoint<32, uint>();
}
TEST(PointEvalGpu, DpfUint64) {
  CheckDpfPoint<40, uint64_t>();
}
TEST(PointEvalGpu, DpfUint128) {
  CheckDpfPoint<64, __uint128_t>();
  CheckDpfPoint<65, __uint128_t>();
  CheckDpfPoint<128, __uint128_t>();
}
TEST(PointEvalGpu, HalfTreeBytes) {
  CheckHalfTreePoint<10, uint>();
}
TEST(PointEvalGpu, HalfTreeUint128) {
  CheckHalfTreePoint<64, __uint128_t>();
}
TEST(PointEvalGpu, PackedHalfTree1Bit) {
  CheckPackedPoint<20, uint32_t, 1>();
}
TEST(PointEvalGpu, PackedHalfTree8Bit) {
  CheckPackedPoint<20, uint32_t, 8>();
}
TEST(PointEvalGpu, PackedHalfTree64BitUint64) {
  CheckPackedPoint<64, uint64_t, 64>();
}
TEST(PointEvalGpu, PackedHalfTree1BitUint128) {
  CheckPackedPoint<128, __uint128_t, 1>();
}
TEST(PointEvalGpu, PackedHalfTreeUint32QueriesForNarrowInputs) {
  CheckPackedPoint<16, uint16_t, 1, 37, uint32_t>();
}
TEST(PointEvalGpu, PackedHalfTreeManyBlocks) {
  CheckPackedPoint<20, uint32_t, 1, 1000>();
}
TEST(PointEvalGpu, PackedHalfTreeDepth1Uint8) {
  CheckPackedPoint<8, uint8_t, 1>();
}
TEST(PointEvalGpu, PackedHalfTree10Bits) {
  CheckPackedPoint<10, uint32_t, 1, 8, uint32_t>();
}

TEST(PointEvalGpu, GrottoDcf) {
  CheckGrottoPoint<20, uint32_t>();
}
TEST(PointEvalGpu, GrottoDcfUint8) {
  CheckGrottoPoint<8, uint8_t>();
}
TEST(PointEvalGpu, GrottoDcfUint64) {
  CheckGrottoPoint<64, uint64_t>();
}
TEST(PointEvalGpu, GrottoDcfUint128) {
  CheckGrottoPoint<128, __uint128_t>();
}
TEST(PointEvalGpu, GrottoDcfUint32QueriesForNarrowInputs) {
  CheckGrottoPoint<16, uint16_t, 37, uint32_t>();
}
TEST(PointEvalGpu, GrottoDcfManyBlocks) {
  CheckGrottoPoint<20, uint32_t, 1000>();
}
TEST(PointEvalGpu, GrottoDcf10Bits) {
  CheckGrottoPoint<10, uint32_t, 8, uint32_t>();
}

TEST(PointEvalGpu, HalfTreeGrottoDcf) {
  CheckHalfTreeGrottoPoint<20, uint32_t>();
}
TEST(PointEvalGpu, HalfTreeGrottoDcfDepth1Uint8) {
  CheckHalfTreeGrottoPoint<8, uint8_t>();
}
TEST(PointEvalGpu, HalfTreeGrottoDcfUint128) {
  CheckHalfTreeGrottoPoint<128, __uint128_t>();
}
TEST(PointEvalGpu, HalfTreeGrottoDcfUint32QueriesForNarrowInputs) {
  CheckHalfTreeGrottoPoint<16, uint16_t, 37, uint32_t>();
}
TEST(PointEvalGpu, HalfTreeGrottoDcfManyBlocks) {
  CheckHalfTreeGrottoPoint<20, uint32_t, 1000>();
}
TEST(PointEvalGpu, HalfTreeGrottoDcf10Bits) {
  CheckHalfTreeGrottoPoint<10, uint32_t, 8, uint32_t>();
}

TEST(PointEvalGpu, DcfBytes) {
  CheckDcfPoint<10, uint>();
}
TEST(PointEvalGpu, DcfUint64) {
  CheckDcfPoint<40, uint64_t>();
}
TEST(PointEvalGpu, DcfGtBytes) {
  CheckDcfPoint<10, uint, 8, fss::DcfPred::kGt>();
}
TEST(PointEvalGpu, DcfGtUint64) {
  CheckDcfPoint<40, uint64_t, 8, fss::DcfPred::kGt>();
}
TEST(PointEvalGpu, VdpfBytes) {
  CheckVdpfPoint<10, uint>();
}
TEST(PointEvalGpu, VdpfUint128) {
  CheckVdpfPoint<64, __uint128_t>();
  CheckVdpfPoint<65, __uint128_t>();
  CheckVdpfPoint<128, __uint128_t>();
}

TEST(PointEvalGpu, Uint32QueriesForNarrowInputs) {
  CheckDpfPoint<16, uint16_t, 8, uint32_t>();
  CheckHalfTreePoint<16, uint16_t, 8, uint32_t>();
  CheckDcfPoint<16, uint16_t, 8, fss::DcfPred::kLt, uint32_t>();
  CheckVdpfPoint<16, uint16_t, 8, uint32_t>();
}


}  // namespace

// The span checks run on the host before any CUDA call, so these cases need no
// device and never launch a kernel.
TEST(PointEvalGpu, RejectsInvalidSpansBeforeLaunch) {
  using BytesGroup = fss::group::Bytes;
  using Prg = fss::prg::ChaCha<2>;
  using CcrHash = fss_test::CcrBackend<fss_test::TestCcrHash>;
  const int4 seed = {1, 2, 3, 4};
  const int4 hash_key = {5, 6, 7, 8};
  int4 ys = {};
  bool bool_ys = false;
  cuda::std::array<int4, 4> pi{};
  const int4 cws_storage[9] = {};
  const int4 seeds_storage[1] = {seed};
  const uint8_t tr = 0;
  const uint8_t xs8 = 0;
  const uint16_t xs16 = 0;

  {
    fss::Dpf<8, BytesGroup, Prg, uint8_t> scheme{Prg(1, 2)};
    EXPECT_THROW(fss::gpu::detail::DpfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const uint8_t>(&tr, 1),
                     cuda::std::span<const uint8_t>(&xs8, 1), &ys, 1, scheme),
        std::invalid_argument);
    EXPECT_THROW(fss::gpu::detail::DpfEvalPointGpu(false, cuda::std::span<const int4>(),
                     cuda::std::span<const int4>(cws_storage, 9),
                     cuda::std::span<const uint8_t>(&tr, 1),
                     cuda::std::span<const uint8_t>(&xs8, 1), &ys, 1, scheme),
        std::invalid_argument);
  }
  {
    using DcfPrg = fss::prg::ChaCha<4>;
    fss::Dcf<8, BytesGroup, DcfPrg, uint8_t> scheme{DcfPrg(1, 2)};
    EXPECT_THROW(fss::gpu::detail::DcfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 9),
                     cuda::std::span<const uint8_t>(&tr, 1),
                     cuda::std::span<const uint8_t>(&xs8, 1), &ys, 1, scheme),
        std::invalid_argument);
  }
  {
    fss::GrottoDcf<8, Prg, uint8_t> scheme{Prg(1, 2)};
    EXPECT_THROW(fss::gpu::detail::GrottoDcfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const uint8_t>(&tr, 1),
                     cuda::std::span<const uint8_t>(&xs8, 1), &bool_ys, 1, scheme),
        std::invalid_argument);
  }
  {
    fss::HalfTreeDpf<8, BytesGroup, CcrHash, uint8_t> scheme{
        fss_test::Ccr(fss_test::TestCcrHash(1, 2)), hash_key};
    EXPECT_THROW(fss::gpu::detail::HalfTreeDpfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const uint8_t>(&tr, 1),
                     cuda::std::span<const uint8_t>(&xs8, 1), &ys, 1, scheme),
        std::invalid_argument);
  }
  {
    fss::PackedHalfTreeDpf<10, 1, CcrHash, uint16_t> scheme{
        fss_test::Ccr(fss_test::TestCcrHash(1, 2)), hash_key};
    EXPECT_THROW(fss::gpu::detail::PackedHalfTreeDpfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const uint16_t>(&xs16, 1), &ys, 1, scheme),
        std::invalid_argument);
  }
  {
    fss::HalfTreeGrottoDcf<8, CcrHash, uint8_t> scheme{
        fss_test::Ccr(fss_test::TestCcrHash(1, 2)), hash_key};
    EXPECT_THROW(fss::gpu::detail::HalfTreeGrottoDcfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const uint8_t>(&xs8, 1), &bool_ys, 1, scheme),
        std::invalid_argument);
  }
  {
    const int4 iv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    const cuda::std::span<const int4, 2> iv_span(iv, 2);
    fss::Vdpf<8, BytesGroup, Prg, fss::hash::Blake3, fss::hash::Blake3, uint16_t> scheme{
        Prg(1, 2), fss::hash::Blake3(iv_span), fss::hash::Blake3(iv_span)};
    EXPECT_THROW(fss::gpu::detail::VdpfEvalPointGpu(false,
                     cuda::std::span<const int4>(seeds_storage, 1),
                     cuda::std::span<const int4>(cws_storage, 1),
                     cuda::std::span<const uint16_t>(&xs16, 1),
                     cuda::std::span<const cuda::std::array<int4, 4>>(&pi, 1),
                     cuda::std::span<const uint16_t>(&xs16, 1), &ys,
                     cuda::std::span<cuda::std::array<int4, 4>>(&pi, 1), 1, scheme),
        std::invalid_argument);
  }
}

// Zero keys write no output, so all seven point launchers accept empty spans
// and a null ys as a no-op, while a negative key count is still rejected. The
// checks run before any CUDA call, so no device is needed.
TEST(PointEvalGpu, ZeroKeysWithEmptySpansIsNoOp) {
  using BytesGroup = fss::group::Bytes;
  using Prg = fss::prg::ChaCha<2>;
  using CcrHash = fss_test::CcrBackend<fss_test::TestCcrHash>;
  const int4 hash_key = {5, 6, 7, 8};

  {
    fss::Dpf<8, BytesGroup, Prg, uint8_t> scheme{Prg(1, 2)};
    EXPECT_NO_THROW(fss::gpu::detail::DpfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint8_t>(),
        cuda::std::span<const uint8_t>(), nullptr, 0, scheme));
    EXPECT_THROW(fss::gpu::detail::DpfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint8_t>(),
        cuda::std::span<const uint8_t>(), nullptr, -1, scheme), std::invalid_argument);
  }
  {
    using DcfPrg = fss::prg::ChaCha<4>;
    fss::Dcf<8, BytesGroup, DcfPrg, uint8_t> scheme{DcfPrg(1, 2)};
    EXPECT_NO_THROW(fss::gpu::detail::DcfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
        cuda::std::span<const uint8_t>(), cuda::std::span<const uint8_t>(), nullptr, 0,
        scheme));
    EXPECT_THROW(fss::gpu::detail::DcfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
        cuda::std::span<const uint8_t>(), cuda::std::span<const uint8_t>(), nullptr, -1,
        scheme), std::invalid_argument);
  }
  {
    fss::GrottoDcf<8, Prg, uint8_t> scheme{Prg(1, 2)};
    EXPECT_NO_THROW(fss::gpu::detail::GrottoDcfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint8_t>(),
        cuda::std::span<const uint8_t>(), nullptr, 0, scheme));
    EXPECT_THROW(fss::gpu::detail::GrottoDcfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint8_t>(),
        cuda::std::span<const uint8_t>(), nullptr, -1, scheme), std::invalid_argument);
  }
  {
    fss::HalfTreeDpf<8, BytesGroup, CcrHash, uint8_t> scheme{
        fss_test::Ccr(fss_test::TestCcrHash(1, 2)), hash_key};
    EXPECT_NO_THROW(fss::gpu::detail::HalfTreeDpfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint8_t>(),
        cuda::std::span<const uint8_t>(), nullptr, 0, scheme));
    EXPECT_THROW(fss::gpu::detail::HalfTreeDpfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint8_t>(),
        cuda::std::span<const uint8_t>(), nullptr, -1, scheme), std::invalid_argument);
  }
  {
    fss::PackedHalfTreeDpf<10, 1, CcrHash, uint16_t> scheme{
        fss_test::Ccr(fss_test::TestCcrHash(1, 2)), hash_key};
    EXPECT_NO_THROW(fss::gpu::detail::PackedHalfTreeDpfEvalPointGpu(false,
        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
        cuda::std::span<const uint16_t>(), nullptr, 0, scheme));
    EXPECT_THROW(fss::gpu::detail::PackedHalfTreeDpfEvalPointGpu(false,
        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
        cuda::std::span<const uint16_t>(), nullptr, -1, scheme), std::invalid_argument);
  }
  {
    fss::HalfTreeGrottoDcf<8, CcrHash, uint8_t> scheme{
        fss_test::Ccr(fss_test::TestCcrHash(1, 2)), hash_key};
    EXPECT_NO_THROW(fss::gpu::detail::HalfTreeGrottoDcfEvalPointGpu(false,
        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
        cuda::std::span<const uint8_t>(), nullptr, 0, scheme));
    EXPECT_THROW(fss::gpu::detail::HalfTreeGrottoDcfEvalPointGpu(false,
        cuda::std::span<const int4>(), cuda::std::span<const int4>(),
        cuda::std::span<const uint8_t>(), nullptr, -1, scheme), std::invalid_argument);
  }
  {
    const int4 iv[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    const cuda::std::span<const int4, 2> iv_span(iv, 2);
    fss::Vdpf<8, BytesGroup, Prg, fss::hash::Blake3, fss::hash::Blake3, uint16_t> scheme{
        Prg(1, 2), fss::hash::Blake3(iv_span), fss::hash::Blake3(iv_span)};
    EXPECT_NO_THROW(fss::gpu::detail::VdpfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint16_t>(),
        cuda::std::span<const cuda::std::array<int4, 4>>(),
        cuda::std::span<const uint16_t>(), nullptr,
        cuda::std::span<cuda::std::array<int4, 4>>(), 0, scheme));
    EXPECT_THROW(fss::gpu::detail::VdpfEvalPointGpu(false, cuda::std::span<const int4>(),
        cuda::std::span<const int4>(), cuda::std::span<const uint16_t>(),
        cuda::std::span<const cuda::std::array<int4, 4>>(),
        cuda::std::span<const uint16_t>(), nullptr,
        cuda::std::span<cuda::std::array<int4, 4>>(), -1, scheme), std::invalid_argument);
  }
}
