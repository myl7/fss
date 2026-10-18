#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda/std/span>
#include <fss/group/bytes.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/point_eval_gpu.cuh>
#include <fss/prg/chacha.cuh>
#include <thread>
#include <limits>

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

template <int bits, typename In, int keys = 8, int par_depth = -1, typename Query = In, bool released = false>
void CheckDpfPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Group = fss::group::Bytes;
  using Scheme = fss::Dpf<bits, Group, Prg, In, par_depth>;
  using Cw = typename Scheme::Cw;
  using Key = typename Scheme::Key;
  constexpr int kStride = bits + 1;
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Cw> cws(keys * kStride);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    scheme.Gen(keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    for (int i = 0; i < kStride; ++i) cws[k * kStride + i] = keys0[k].cws[i];
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_cw_s, device_out_cw, device_seeds, device_ys;
  DeviceBuffer<Query> device_xs;
  DeviceBuffer<fss::gpu::ExtBits<bits>> device_extra;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_cw_s.Allocate(bits * keys), cudaSuccess);
  ASSERT_EQ(device_extra.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_out_cw.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  if constexpr (released) {
    fss::gpu::DpfRelayoutGpu<bits, Group, Prg, In>(
        device_cws.data(), keys, device_cw_s.data(), device_extra.data(), device_out_cw.data(), stream);
  } else {
    fss::gpu::DpfRepackGpu<bits, Group, Prg, In, par_depth>(
        device_cws.data(), keys, device_cw_s.data(), device_extra.data(), device_out_cw.data(), stream);
  }
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::DpfEvalPointGpu(party != 0, device_seeds.data(), device_cw_s.data(), device_extra.data(),
        device_out_cw.data(), device_xs.data(), device_ys.data(), keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], scheme.Eval(party != 0, party_key, xs[k]), k);
      if (party == 0) first[k] = actual[k];
      else {
        int4 sum = (Group::From(first[k]) + Group::From(actual[k])).Into();
        ExpectInt4(sum, xs[k] == alphas[k] ? beta : int4{0, 0, 0, 0}, k);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

template <int bits, typename In, int keys = 8, int par_depth = -1, typename Query = In, bool released = false>
void CheckHalfTreePoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<1>;
  using Group = fss::group::Bytes;
  using Scheme = fss::HalfTreeDpf<bits, Group, Prg, In, par_depth>;
  using Cw = typename Scheme::Cw;
  using Key = typename Scheme::Key;
  constexpr int kStride = bits;
  const int4 hash_key = {11, 22, 33, 44};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash_key};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Cw> cws(keys * kStride);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys), ocws(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    scheme.Gen(keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    for (int i = 0; i < kStride; ++i) cws[k * kStride + i] = keys0[k].cws[i];
    ocws[k] = keys0[k].ocw;
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_cw_s, device_ocws, device_seeds, device_ys;
  DeviceBuffer<uint32_t> device_extra;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_cw_s.Allocate(bits * keys), cudaSuccess);
  ASSERT_EQ(device_extra.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ocws.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ocws.Upload(ocws.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  if constexpr (released) {
    fss::gpu::HalfTreeDpfRelayoutGpu<bits, Group, Prg, In>(
        device_cws.data(), keys, device_cw_s.data(), device_extra.data(), stream);
  } else {
    fss::gpu::HalfTreeDpfRepackGpu<bits, Group, Prg, In, par_depth>(
        device_cws.data(), keys, device_cw_s.data(), device_extra.data(), stream);
  }
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::HalfTreeDpfEvalPointGpu(party != 0, device_seeds.data(), device_cw_s.data(), device_extra.data(),
        device_ocws.data(), device_xs.data(), device_ys.data(), keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], scheme.Eval(party != 0, party_key, xs[k]), k);
      if (party == 0) first[k] = actual[k];
      else {
        int4 sum = (Group::From(first[k]) + Group::From(actual[k])).Into();
        ExpectInt4(sum, xs[k] == alphas[k] ? beta : int4{0, 0, 0, 0}, k);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

template <int bits, typename In, int keys = 8, fss::DcfPred pred = fss::DcfPred::kLt, int par_depth = -1,
    typename Query = In, bool released = false>
void CheckDcfPoint() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<4>;
  using Group = fss::group::Bytes;
  using Scheme = fss::Dcf<bits, Group, Prg, In, pred, par_depth>;
  using Cw = typename Scheme::Cw;
  using Key = typename Scheme::Key;
  constexpr int kStride = bits + 1;
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Cw> cws(keys * kStride);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    scheme.Gen(keys0[k], keys1[k], cuda::std::span<const int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta);
    for (int i = 0; i < kStride; ++i) cws[k * kStride + i] = keys0[k].cws[i];
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_cw_s, device_cw_v, device_out_cw, device_seeds, device_ys;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_cw_s.Allocate(bits * keys), cudaSuccess);
  ASSERT_EQ(device_cw_v.Allocate(bits * keys), cudaSuccess);
  ASSERT_EQ(device_out_cw.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  if constexpr (released) {
    fss::gpu::DcfRelayoutGpu<bits, Group, Prg, In>(
        device_cws.data(), keys, device_cw_s.data(), device_cw_v.data(), device_out_cw.data(), stream);
  } else {
    fss::gpu::DcfRepackGpu<bits, Group, Prg, In, pred, par_depth>(
        device_cws.data(), keys, device_cw_s.data(), device_cw_v.data(), device_out_cw.data(), stream);
  }
  std::vector<int4> actual(keys), first(keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::DcfEvalPointGpu(party != 0, device_seeds.data(), device_cw_s.data(), device_cw_v.data(),
        device_out_cw.data(), device_xs.data(), device_ys.data(), keys, scheme, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      ExpectInt4(actual[k], scheme.Eval(party != 0, party_key, xs[k]), k);
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

template <int bits, typename In, int keys = 8, int par_depth = -1, typename Query = In, bool released = false>
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
  using Scheme = fss::Vdpf<bits, Group, Prg, Blake3, Blake3, In, par_depth>;
  using Cw = typename Scheme::Cw;
  using Key = typename Scheme::Key;
  constexpr int kStride = bits;
  // Blake3 copies the IV by value at construction.
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Blake3 hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash, hash};
  const int4 beta = {7, 19, 23, 42};
  const std::vector<In> alphas = MakeAlphas<bits, In>(keys);
  const std::vector<In> xs = MakePoints<bits, In>(alphas);
  std::vector<Cw> cws(keys * kStride);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys), ocws(keys);
  std::vector<cuda::std::array<int4, 4>> cs(keys);
  for (int k = 0; k < keys; ++k) {
    seeds[2 * k] = {11 + k, 22, 33, 44};
    seeds[2 * k + 1] = {55 + k, 66, 77, 88};
    ASSERT_EQ(scheme.Gen(keys0[k], keys1[k], cuda::std::span<int4, 2>(seeds.data() + 2 * k, 2), alphas[k], beta), 0);
    for (int i = 0; i < kStride; ++i) cws[k * kStride + i] = keys0[k].cws[i];
    cs[k] = keys0[k].cs;
    ocws[k] = keys0[k].ocw;
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_cw_s, device_ocws, device_seeds, device_ys, device_pi;
  DeviceBuffer<cuda::std::array<int4, 4>> device_cs;
  DeviceBuffer<fss::gpu::ExtBits<bits>> device_extra;
  DeviceBuffer<Query> device_xs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_cw_s.Allocate(bits * keys), cudaSuccess);
  ASSERT_EQ(device_extra.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_cs.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_cs.Upload(cs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_ocws.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ocws.Upload(ocws.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_pi.Allocate(4 * keys), cudaSuccess);
  ASSERT_EQ(device_xs.Allocate(keys), cudaSuccess);
  const std::vector<Query> query_buffer(xs.begin(), xs.end());
  ASSERT_EQ(device_xs.Upload(query_buffer.data(), keys), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  if constexpr (released) {
    fss::gpu::VdpfRelayoutGpu<bits, Group, Prg, Blake3, Blake3, In>(
        device_cws.data(), keys, device_cw_s.data(), device_extra.data(), stream);
  } else {
    fss::gpu::VdpfRepackGpu<bits, Group, Prg, Blake3, Blake3, In, par_depth>(
        device_cws.data(), keys, device_cw_s.data(), device_extra.data(), stream);
  }
  std::vector<int4> actual(keys), first(keys), gpu_pi(2 * 4 * keys);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int k = 0; k < keys; ++k) party_seeds[k] = seeds[2 * k + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    fss::gpu::VdpfEvalPointGpu(party != 0, device_seeds.data(), device_cw_s.data(), device_extra.data(),
        device_cs.data(), device_ocws.data(), device_xs.data(), device_ys.data(), device_pi.data(), keys, scheme,
        stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), keys * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(gpu_pi.data() + party * 4 * keys, device_pi.data(), 4 * keys * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int k = 0; k < keys; ++k) {
      SCOPED_TRACE(::testing::Message() << "key " << k);
      const Key &party_key = party == 0 ? keys0[k] : keys1[k];
      auto expected = scheme.Eval(party != 0, party_key, xs[k]);
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

TEST(PointEvalGpu, ReleasedUint32QueriesForNarrowInputs) {
  CheckDpfPoint<16, uint16_t, 8, -1, uint32_t, true>();
  CheckHalfTreePoint<16, uint16_t, 8, -1, uint32_t, true>();
  CheckDcfPoint<16, uint16_t, 8, fss::DcfPred::kLt, -1, uint32_t, true>();
  CheckVdpfPoint<16, uint16_t, 8, -1, uint32_t, true>();
}

TEST(PointEvalGpu, ConfiguredDepthsAndPreparation) {
  CheckDpfPoint<10, uint, 8, 0>();
  CheckDpfPoint<10, uint, 8, 2>();
  CheckHalfTreePoint<10, uint, 8, 0>();
  CheckHalfTreePoint<10, uint, 8, 2>();
  CheckDcfPoint<10, uint, 8, fss::DcfPred::kLt, 0>();
  CheckDcfPoint<10, uint, 8, fss::DcfPred::kGt, 2>();
  CheckVdpfPoint<10, uint, 8, 0>();
  CheckVdpfPoint<10, uint, 8, 2>();
}

TEST(GpuWorkspace, CapacityReuseAndAllocationFailure) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  fss::gpu::detail::GpuWorkspace workspace(device);
  auto *first = workspace.Ensure<int4>(0, 64);
  EXPECT_EQ(workspace.Ensure<int4>(0, 32), first);
  EXPECT_THROW(workspace.Ensure<unsigned char>(0, std::numeric_limits<size_t>::max() / 2), std::runtime_error);
  cudaGetLastError();
  EXPECT_EQ(workspace.Ensure<int4>(0, 64), first);
  EXPECT_EQ(workspace.Release(), cudaSuccess);
}

TEST(GpuWorkspace, ThreadExitReleasesPendingScratch) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(fss::gpu::ReleaseGpuWorkspaces(), cudaSuccess);
  for (int i = 0; i < 3; ++i) {
    unsigned char *scratch = nullptr;
    std::thread([device, &scratch] {
      ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
      static const int kTag = 0;
      auto &workspace = fss::gpu::detail::GetGpuWorkspaceCache().Get(&kTag);
      scratch = workspace.Ensure<unsigned char>(0, 8 * 1024 * 1024);
      cudaPointerAttributes attributes{};
      ASSERT_EQ(cudaPointerGetAttributes(&attributes, scratch), cudaSuccess);
      ASSERT_EQ(attributes.type, cudaMemoryTypeDevice);
      ASSERT_EQ(cudaMemsetAsync(scratch, 0x42, 8 * 1024 * 1024), cudaSuccess);
    }).join();
    ASSERT_NE(scratch, nullptr);
    // Inspect this context's allocation, because device-wide free memory can
    // change when another process or test allocates on the same GPU.
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, scratch);
    cudaGetLastError();
    if (status == cudaSuccess) {
      EXPECT_EQ(attributes.type, cudaMemoryTypeUnregistered);
    } else {
      EXPECT_EQ(status, cudaErrorInvalidValue);
    }
  }
}

TEST(GpuWorkspace, DeviceOwnershipAndReleaseRestoresDevice) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) GTEST_SKIP() << "requires two CUDA devices";
  int original = 0;
  ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);
  static const int kTag = 0;
  unsigned char *first = nullptr;
  for (int device : {0, 1, 0}) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    auto *scratch = fss::gpu::detail::GetGpuWorkspaceCache().Get(&kTag).Ensure<unsigned char>(0, 1024);
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, scratch), cudaSuccess);
    EXPECT_EQ(attributes.device, device);
    if (device == 0) {
      if (first != nullptr) EXPECT_EQ(scratch, first);
      first = scratch;
    }
    ASSERT_EQ(cudaMemsetAsync(scratch, device, 1024), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(fss::gpu::ReleaseGpuWorkspaces(), cudaSuccess);
  int current = 0;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  ASSERT_EQ(cudaSetDevice(original), cudaSuccess);
}

}  // namespace
