#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <fss/eval_all_gpu.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/hash/blake3.cuh>
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

template <int bits, int z, int threads, int keys, typename Group, bool half_tree, bool large = false>
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
  using Key = typename Scheme::Key;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = half_tree ? bits : bits + 1;
  constexpr int b1 = z - (threads == 128 ? 7 : 8);
  const int4 hash_key = {11, 22, 33, 44};
  const int4 beta = {7, 19, 23, 42};
  auto make_scheme = [&](Prg prg) {
    if constexpr (half_tree) return Scheme{prg, hash_key};
    else return Scheme{prg};
  };
  Scheme scheme = make_scheme(Prg(0x12345678, static_cast<int>(0x9abcdef0u)));
  std::vector<Cw> cws(keys * stride);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), ocws(keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    scheme.Gen(keys0[key], keys1[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), 42 + key, beta);
    for (int i = 0; i < stride; ++i) cws[key * stride + i] = keys0[key].cws[i];
    if constexpr (half_tree) ocws[key] = keys0[key].ocw;
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_ocws, device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_ocws.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_ocws.Upload(ocws.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * n), cudaSuccess);
  std::vector<int4> actual(keys * n), first(keys * n), host_ys(large ? n : 0);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1 && half_tree) {
      fss::gpu::HalfTreeDpfEvalAllGpuBatch<z, b1, threads>(party, device_seeds.data(), device_cws.data(),
          device_ocws.data(), keys, device_outputs.data(), scheme);
    } else if constexpr (keys > 1) {
      fss::gpu::DpfEvalAllGpuBatch<z, b1, threads>(party, device_seeds.data(), device_cws.data(), keys,
          device_outputs.data(), scheme);
    } else if constexpr (half_tree) {
      fss::gpu::HalfTreeDpfEvalAllGpu<z, b1, threads>(party, party_seeds[0], device_cws.data(), ocws[0],
          device_outputs.data(), scheme);
    } else {
      fss::gpu::DpfEvalAllGpu<z, b1, threads>(party, party_seeds[0], device_cws.data(), device_outputs.data(), scheme);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      const Key &party_key = party == 0 ? keys0[key] : keys1[key];
      if constexpr (large) scheme.EvalAll(static_cast<bool>(party), party_key, cuda::std::span<int4>(host_ys.data(), n));
      for (size_t x = 0; x < n; ++x) {
        int4 expected;
        if constexpr (large) expected = host_ys[x];
        else expected = scheme.Eval(party, party_key, x);
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
TEST(EvalAllGpu, DpfAbove24Bits) {
  CheckFullEvaluation<25, 21, 128, 1, Uint127Group, false, true>();
}

template <int bits, int z, int threads, typename In = uint>
void CheckVdpfGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Blake3 = fss::hash::Blake3;
  using Group = Uint127Group;
  using Scheme = fss::Vdpf<bits, Group, Prg, Blake3, Blake3, In>;
  using Cw = typename Scheme::Cw;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits;
  constexpr int b1 = z - (threads == 128 ? 7 : 8);
  // Blake3 copies the IV by value at construction.
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Blake3 hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash, hash};
  const In alpha = 42;
  const int4 beta = {7, 19, 23, 42};
  int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  typename Scheme::Key k0, k1;
  ASSERT_EQ(scheme.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), alpha, beta), 0);
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_ys;
  DeviceBuffer<cuda::std::array<int4, 4>> device_pi;
  ASSERT_EQ(device_cws.Allocate(stride), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(k0.cws, stride), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(n), cudaSuccess);
  ASSERT_EQ(device_pi.Allocate(1), cudaSuccess);

  std::vector<int4> host_ys(n), actual(n), first(n);
  cuda::std::array<int4, 4> pi_cpu, gpu_pi[2];
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    const typename Scheme::Key &key = party == 0 ? k0 : k1;
    scheme.EvalAllMerkle(party, key, cuda::std::span<int4>(host_ys.data(), n), pi_cpu);
    fss::gpu::VdpfEvalAllMerkleGpu<z, b1, threads>(
        party, key.s0, device_cws.data(), key.cs, key.ocw, device_ys.data(), device_pi.data(), scheme);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), n * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&gpu_pi[party], device_pi.data(), sizeof(cuda::std::array<int4, 4>), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t x = 0; x < n; ++x) {
      ExpectOutput(actual[x], host_ys[x], x);
      if (party == 0) first[x] = actual[x];
      else {
        int4 sum = (Group::From(first[x]) + Group::From(actual[x])).Into();
        ExpectOutput(sum, Group::From(x == alpha ? beta : int4{0, 0, 0, 0}).Into(), x);
      }
    }
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(gpu_pi[party][i].x, pi_cpu[i].x) << "pi word " << i;
      EXPECT_EQ(gpu_pi[party][i].y, pi_cpu[i].y) << "pi word " << i;
      EXPECT_EQ(gpu_pi[party][i].z, pi_cpu[i].z) << "pi word " << i;
      EXPECT_EQ(gpu_pi[party][i].w, pi_cpu[i].w) << "pi word " << i;
    }
  }
  EXPECT_TRUE(Scheme::Verify(cuda::std::span<const int4, 4>(gpu_pi[0]), cuda::std::span<const int4, 4>(gpu_pi[1])));
}

template <int bits, int out_bits, int z, int b1, int threads, int keys = 1>
void CheckPackedGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<1>;
  using Scheme = fss::PackedHalfTreeDpf<bits, out_bits, Prg, uint>;
  using Cw = typename Scheme::Cw;
  using Key = typename Scheme::Key;
  constexpr int kDepth = Scheme::kDepth;
  constexpr int kLanes = Scheme::kLanes;
  constexpr size_t n = size_t{1} << bits;
  constexpr size_t blocks = size_t{1} << kDepth;
  const int4 hash_key = {11, 22, 33, 44};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash_key};
  const uint64_t beta = 1;
  std::vector<Cw> cws(keys * kDepth);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> fcws(keys), seeds(2 * keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    scheme.Gen(keys0[key], keys1[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), 42 + key, beta);
    for (int i = 0; i < kDepth; ++i) cws[key * kDepth + i] = keys0[key].cws[i];
    fcws[key] = keys0[key].fcw;
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_fcws, device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_fcws.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_fcws.Upload(fcws.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * blocks), cudaSuccess);
  std::vector<int4> actual(keys * blocks), first(keys * blocks), host_ys(blocks);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1) {
      fss::gpu::PackedHalfTreeDpfEvalAllGpuBatch<z, b1, threads>(
          party, device_seeds.data(), device_cws.data(), device_fcws.data(), keys, device_outputs.data(), scheme);
    } else {
      fss::gpu::PackedHalfTreeDpfEvalAllGpu<z, b1, threads>(
          party, party == 0 ? keys0[0].s0 : keys1[0].s0, device_cws.data(), keys0[0].fcw, device_outputs.data(), scheme);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      const Key &party_key = party == 0 ? keys0[key] : keys1[key];
      scheme.EvalAll(party, party_key, cuda::std::span<int4>(host_ys.data(), blocks));
      for (size_t j = 0; j < blocks; ++j) {
        ExpectOutput(actual[key * blocks + j], host_ys[j], j);
        if (party == 0) first[key * blocks + j] = actual[key * blocks + j];
      }
      if (party == 1) {
        for (size_t x = 0; x < n; ++x) {
          int4 sum = fss::util::Xor(first[key * blocks + x / kLanes], actual[key * blocks + x / kLanes]);
          EXPECT_EQ(Scheme::Extract(sum, static_cast<uint>(x)), x == size_t{42 + key} ? beta : 0) << "input " << x;
        }
      }
    }
  }
}

template <int bits, int z, int threads, int keys, typename Group, fss::DcfPred pred = fss::DcfPred::kLt,
    int alpha = 42, typename In = uint, int par_depth = -1, bool large = false>
void CheckDcfGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Scheme = fss::Dcf<bits, Group, fss::prg::ChaCha<4>, In, pred, par_depth>;
  using Cw = typename Scheme::Cw;
  using Key = typename Scheme::Key;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits + 1;
  constexpr int b1 = z - (threads == 128 ? 7 : 8);
  Scheme scheme{fss::prg::ChaCha<4>(0x12345678, static_cast<int>(0x9abcdef0u))};
  const int4 beta = {7, 19, 23, 42};
  std::vector<Cw> cws(keys * stride);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    scheme.Gen(keys0[key], keys1[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2),
        static_cast<In>(alpha + key), beta);
    for (int i = 0; i < stride; ++i) cws[key * stride + i] = keys0[key].cws[i];
  }
  DeviceBuffer<Cw> device_cws;
  DeviceBuffer<int4> device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * n), cudaSuccess);
  std::vector<int4> actual(keys * n), first(keys * n), host_ys(large ? n : 0);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1) {
      fss::gpu::DcfEvalAllGpuBatch<z, b1, threads>(
          party, device_seeds.data(), device_cws.data(), keys, device_outputs.data(), scheme);
    } else {
      fss::gpu::DcfEvalAllGpu<z, b1, threads>(
          party, party_seeds[0], device_cws.data(), device_outputs.data(), scheme);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      const Key &party_key = party == 0 ? keys0[key] : keys1[key];
      if constexpr (large) scheme.EvalAll(static_cast<bool>(party), party_key, cuda::std::span<int4>(host_ys.data(), n));
      for (size_t x = 0; x < n; ++x) {
        int4 expected;
        if constexpr (large) expected = host_ys[x];
        else expected = scheme.Eval(party, party_key, x);
        ExpectOutput(actual[key * n + x], expected, x);
        if (party == 0) first[key * n + x] = actual[key * n + x];
        else {
          const bool hit = pred == fss::DcfPred::kLt ? x < size_t{alpha + key} : x > size_t{alpha + key};
          int4 sum = (Group::From(first[key * n + x]) + Group::From(actual[key * n + x])).Into();
          ExpectOutput(sum, Group::From(hit ? beta : int4{0, 0, 0, 0}).Into(), x);
        }
      }
    }
  }
}

template <int bits, int z, int threads>
void CheckGrottoGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::GrottoDcf<bits, Prg, uint>;
  using Cw = typename Scheme::Cw;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits + 1;
  constexpr int b1 = z - (threads == 128 ? 7 : 8);
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint alpha = 42;
  typename Scheme::Key k0, k1;
  scheme.Gen(k0, k1, cuda::std::span<const int4, 2>(seeds), alpha);
  DeviceBuffer<Cw> device_cws;
  ASSERT_EQ(device_cws.Allocate(stride), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(k0.cws, stride), cudaSuccess);

  auto host_ys = std::make_unique<bool[]>(n);
  auto first = std::make_unique<bool[]>(n);
  DeviceBuffer<bool> device_p, device_ys;
  ASSERT_EQ(device_p.Allocate(2 * n - 1), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(n), cudaSuccess);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    const typename Scheme::Key &key = party == 0 ? k0 : k1;
    auto host_tree = scheme.Preprocess(static_cast<bool>(party), key);
    fss::gpu::GrottoDcfPreprocessGpu<z, b1, threads>(
        static_cast<bool>(party), key.s0, device_cws.data(), device_p.data(), scheme);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto actual_p = std::make_unique<bool[]>(2 * n - 1);
    ASSERT_EQ(cudaMemcpy(actual_p.get(), device_p.data(), (2 * n - 1) * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < 2 * n - 1; ++i) EXPECT_EQ(actual_p[i], host_tree.p[i]) << "node " << i;

    scheme.EvalAll(static_cast<bool>(party), key, cuda::std::span<bool>(host_ys.get(), n));
    fss::gpu::GrottoDcfEvalAllGpu<z, b1, threads>(
        static_cast<bool>(party), key.s0, device_cws.data(), device_ys.data(), scheme);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto actual_ys = std::make_unique<bool[]>(n);
    ASSERT_EQ(cudaMemcpy(actual_ys.get(), device_ys.data(), n * sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t x = 0; x < n; ++x) {
      EXPECT_EQ(actual_ys[x], host_ys[x]) << "input " << x;
      if (party == 0) {
        first[x] = actual_ys[x];
      } else {
        EXPECT_EQ(static_cast<bool>(first[x] ^ actual_ys[x]), x >= alpha) << "input " << x;
      }
    }
  }
}

TEST(EvalAllGpu, VdpfSmall) {
  CheckVdpfGpu<12, 12, 256>();
}
TEST(EvalAllGpu, VdpfSubtrees) {
  CheckVdpfGpu<14, 12, 256>();
}
TEST(EvalAllGpu, VdpfWideInUint128) {
  CheckVdpfGpu<10, 10, 256, __uint128_t>();
}

TEST(EvalAllGpu, PackedSmall) {
  CheckPackedGpu<12, 8, 8, 1, 128>();
}
TEST(EvalAllGpu, PackedOneBit) {
  CheckPackedGpu<12, 1, 5, 0, 32>();
}
TEST(EvalAllGpu, PackedBatch) {
  CheckPackedGpu<12, 8, 8, 1, 128, 3>();
}

TEST(EvalAllGpu, DcfLeafFrontierBytes) {
  CheckDcfGpu<8, 8, 128, 1, fss::group::Bytes>();
}
TEST(EvalAllGpu, DcfLeafFrontierUint) {
  CheckDcfGpu<8, 8, 128, 1, Uint127Group>();
}
TEST(EvalAllGpu, DcfSubtreesUint) {
  CheckDcfGpu<14, 12, 256, 1, Uint127Group>();
}
TEST(EvalAllGpu, DcfBatchUint) {
  CheckDcfGpu<8, 8, 128, 3, Uint127Group>();
}

TEST(EvalAllGpu, DcfGtLeafFrontierBytes) {
  CheckDcfGpu<8, 8, 128, 1, fss::group::Bytes, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtLeafFrontierUint) {
  CheckDcfGpu<8, 8, 128, 1, Uint127Group, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtSubtreesUint) {
  CheckDcfGpu<14, 12, 256, 1, Uint127Group, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtBatchUint) {
  CheckDcfGpu<8, 8, 128, 3, Uint127Group, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtAlphaZero) {
  CheckDcfGpu<8, 8, 128, 1, Uint127Group, fss::DcfPred::kGt, 0>();
}
TEST(EvalAllGpu, DcfGtAlphaMax) {
  CheckDcfGpu<8, 8, 128, 1, Uint127Group, fss::DcfPred::kGt, 255>();
}
TEST(EvalAllGpu, DcfAbove24Bits) {
  CheckDcfGpu<25, 21, 128, 1, Uint127Group, fss::DcfPred::kLt, 42, uint, -1, true>();
}
TEST(EvalAllGpu, DcfGtAbove24Bits) {
  CheckDcfGpu<25, 21, 128, 1, Uint127Group, fss::DcfPred::kGt, 42, uint, -1, true>();
}
TEST(EvalAllGpu, DcfWideInUint16) {
  CheckDcfGpu<8, 8, 128, 1, Uint127Group, fss::DcfPred::kLt, 42, uint16_t>();
}
TEST(EvalAllGpu, DcfCustomParDepth) {
  CheckDcfGpu<8, 8, 128, 1, Uint127Group, fss::DcfPred::kLt, 42, uint, 2>();
}

TEST(EvalAllGpu, GrottoSmall) {
  CheckGrottoGpu<8, 8, 128>();
}
TEST(EvalAllGpu, GrottoSubtrees) {
  CheckGrottoGpu<14, 12, 256>();
}

}  // namespace
