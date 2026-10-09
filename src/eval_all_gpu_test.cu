#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <fss/eval_all_gpu.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>

namespace {

// A named wrapper keeps the 128-bit modulus out of nvcc kernel stubs.
struct Uint127Group {
  using Impl = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
  Impl value;
  __host__ __device__ Uint127Group operator+(Uint127Group rhs) const { return {value + rhs.value}; }
  __host__ __device__ Uint127Group operator-() const { return {-value}; }
  __host__ __device__ static Uint127Group From(int4 input) { return {Impl::From(input)}; }
  __host__ __device__ int4 Into() const { return value.Into(); }
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) EXPECT_EQ(cudaFree(data_), cudaSuccess);
  }
  cudaError_t Allocate(size_t count) { return cudaMalloc(&data_, count * sizeof(T)); }
  cudaError_t Upload(const T *source, size_t count) {
    return cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice);
  }
  T *data() const { return data_; }
 private:
  T *data_ = nullptr;
};

template <int bits, typename Group, bool half_tree>
struct SchemeType;
template <int bits, typename Group>
struct SchemeType<bits, Group, false> {
  using Prg = fss::prg::ChaCha<2>;
  using Type = fss::Dpf<bits, Group, Prg, uint>;
};
template <int bits, typename Group>
struct SchemeType<bits, Group, true> {
  using Prg = fss::prg::ChaCha<1>;
  using Type = fss::HalfTreeDpf<bits, Group, Prg, uint>;
};

void ExpectOutput(int4 actual, int4 expected, size_t input) {
  EXPECT_EQ(actual.x, expected.x) << "input " << input;
  EXPECT_EQ(actual.y, expected.y) << "input " << input;
  EXPECT_EQ(actual.z, expected.z) << "input " << input;
  EXPECT_EQ(actual.w, expected.w) << "input " << input;
}

template <int bits, int z, int threads, int keys, typename Group, bool half_tree>
void CheckFullEvaluation() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Types = SchemeType<bits, Group, half_tree>;
  using Scheme = typename Types::Type;
  using Prg = typename Types::Prg;
  using Cw = typename Scheme::Cw;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = half_tree ? bits : bits + 1;
  constexpr int b1 = z - (threads == 128 ? 7 : 8);
  const int nonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};
  const int4 hash_key = {11, 22, 33, 44};
  const int4 beta = {7, 19, 23, 42};
  DeviceBuffer<int> device_nonce;
  ASSERT_EQ(device_nonce.Allocate(2), cudaSuccess);
  ASSERT_EQ(device_nonce.Upload(nonce, 2), cudaSuccess);
  auto make_scheme = [&](Prg prg) {
    if constexpr (half_tree) return Scheme{prg, hash_key};
    else return Scheme{prg};
  };
  Scheme cpu = make_scheme(Prg(nonce));
  Scheme gpu = make_scheme(Prg(device_nonce.data()));
  std::vector<Cw> cws(keys * stride);
  std::vector<int4> seeds(2 * keys), ocws(keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    if constexpr (half_tree) {
      cpu.Gen(cws.data() + key * stride, ocws[key], seeds.data() + 2 * key, 42 + key, beta);
    } else {
      cpu.Gen(cws.data() + key * stride, seeds.data() + 2 * key, 42 + key, beta);
    }
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_ocws, device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_ocws.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ocws.Upload(ocws.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * n), cudaSuccess);
  std::vector<int4> actual(keys * n), first(keys * n);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1 && half_tree) {
      fss::gpu::HalfTreeDpfEvalAllGpuBatch<z, b1, threads>(party, device_seeds.data(), device_cws.data(),
          device_ocws.data(), keys, device_outputs.data(), gpu);
    } else if constexpr (keys > 1) {
      fss::gpu::DpfEvalAllGpuBatch<z, b1, threads>(party, device_seeds.data(), device_cws.data(), keys,
          device_outputs.data(), gpu);
    } else if constexpr (half_tree) {
      fss::gpu::HalfTreeDpfEvalAllGpu<z, b1, threads>(party, party_seeds[0], device_cws.data(), ocws[0],
          device_outputs.data(), gpu);
    } else {
      fss::gpu::DpfEvalAllGpu<z, b1, threads>(party, party_seeds[0], device_cws.data(), device_outputs.data(), gpu);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      for (size_t x = 0; x < n; ++x) {
        int4 expected;
        if constexpr (half_tree) {
          expected = cpu.Eval(party, party_seeds[key], cws.data() + key * stride, ocws[key], x);
        } else {
          expected = cpu.Eval(party, party_seeds[key], cws.data() + key * stride, x);
        }
        ExpectOutput(actual[key * n + x], expected, x);
        if (party == 0) first[key * n + x] = actual[key * n + x];
        else {
          int4 sum = (Group::From(first[key * n + x]) + Group::From(actual[key * n + x])).Into();
          ExpectOutput(sum, Group::From(x == 42 + key ? beta : int4{0, 0, 0, 0}).Into(), x);
        }
      }
    }
  }
}

TEST(EvalAllGpu, DpfLeafFrontierBytes) {
  CheckFullEvaluation<8, 8, 128, 1, fss::group::Bytes, false>();
}
TEST(EvalAllGpu, DpfLeafFrontierUint) {
  CheckFullEvaluation<8, 8, 128, 1, Uint127Group, false>();
}
TEST(EvalAllGpu, HalfTreeParentFrontierBytes) {
  CheckFullEvaluation<8, 7, 128, 1, fss::group::Bytes, true>();
}
TEST(EvalAllGpu, HalfTreeParentFrontierUint) {
  CheckFullEvaluation<8, 7, 128, 1, Uint127Group, true>();
}
TEST(EvalAllGpu, DpfBatchBytes) {
  CheckFullEvaluation<8, 8, 128, 3, fss::group::Bytes, false>();
}
TEST(EvalAllGpu, DpfBatchUint) {
  CheckFullEvaluation<8, 8, 128, 3, Uint127Group, false>();
}
TEST(EvalAllGpu, HalfTreeBatchBytes) {
  CheckFullEvaluation<8, 7, 128, 3, fss::group::Bytes, true>();
}
TEST(EvalAllGpu, HalfTreeBatchUint) {
  CheckFullEvaluation<8, 7, 128, 3, Uint127Group, true>();
}
TEST(EvalAllGpu, DpfSubtreesUint) {
  CheckFullEvaluation<14, 12, 256, 1, Uint127Group, false>();
}
TEST(EvalAllGpu, HalfTreeSubtreesUint) {
  CheckFullEvaluation<14, 12, 256, 1, Uint127Group, true>();
}

}  // namespace
