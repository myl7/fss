#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prg/chacha.cuh>
#include "core_api_test_helpers.cuh"
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/vdpf.cuh>

namespace {

using Uint128Group = fss::group::Uint128<>;

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

template <int bits, typename Group, bool half_tree, typename In = uint>
struct SchemeType;
template <int bits, typename Group, typename In>
struct SchemeType<bits, Group, false, In> {
  using Prg = fss::prg::ChaCha<2>;
  using Type = fss::Dpf<bits, Group, Prg, In>;
};
template <int bits, typename Group, typename In>
struct SchemeType<bits, Group, true, In> {
  using Prg = fss_test::TestCcrHash;
  using Type = fss::HalfTreeDpf<bits, Group, fss_test::CcrBackend<Prg>, In>;
};

void ExpectOutput(int4 actual, int4 expected, size_t input) {
  EXPECT_EQ(actual.x, expected.x) << "input " << input;
  EXPECT_EQ(actual.y, expected.y) << "input " << input;
  EXPECT_EQ(actual.z, expected.z) << "input " << input;
  EXPECT_EQ(actual.w, expected.w) << "input " << input;
}

template <int bits, int z, int threads, int keys, typename Group, bool half_tree, bool large = false,
    typename In = uint, int par_depth = 0>
void CheckFullEvaluation() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Types = SchemeType<bits, Group, half_tree, In>;
  using Scheme = typename Types::Type;
  using Prg = typename Types::Prg;
  using Key = fss_test::ComponentKey<Scheme, In>;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits + 1;
  const int4 hash_key = {11, 22, 33, 44};
  const int4 beta = {7, 19, 23, 42};
  auto make_scheme = [&](Prg prg) {
    if constexpr (half_tree) return Scheme{fss_test::Ccr(prg), hash_key};
    else return Scheme{prg};
  };
  Scheme scheme = make_scheme(Prg(0x12345678, static_cast<int>(0x9abcdef0u)));
  // Dense level-major layout the batch entry consumes: layer i of key k is
  // cws[i * keys + k], with the packed control in trs[k].
  std::vector<int4> cws(keys * stride);
  std::vector<In> trs(keys);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[key], keys1[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), 42 + key, beta);
    scheme.Gen(cuda::std::span<int4>(cws.data() + key, static_cast<size_t>(bits) * keys + 1),
        trs[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), 42 + key, beta, keys);
  }
  DeviceBuffer<int4> device_cws;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<int4> device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * n), cudaSuccess);
  std::vector<int4> actual(keys * n), first(keys * n), host_ys(large ? n : 0);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1) {
      ASSERT_EQ((scheme.template EvalAllGpuBatch<z, threads>(
          party, {device_seeds.data(), keys}, {device_cws.data(), cws.size()},
          {device_tr.data(), keys}, {device_outputs.data(), keys * n})), cudaSuccess);
    } else {
      ASSERT_EQ((scheme.template EvalAllGpu<z, threads>(
          party, party_seeds[0], {device_cws.data(), stride}, trs[0],
          {device_outputs.data(), n})), cudaSuccess);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      const Key &party_key = party == 0 ? keys0[key] : keys1[key];
      if constexpr (large)
        fss_test::EvalAllKey<par_depth>(scheme, static_cast<bool>(party), party_key, cuda::std::span<int4>(host_ys.data(), n));
      for (size_t x = 0; x < n; ++x) {
        int4 expected;
        if constexpr (large) expected = host_ys[x];
        else expected = fss_test::EvalKey(scheme, party, party_key, x);
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
  CheckFullEvaluation<8, 8, 128, 1, Uint128Group, false>();
}
TEST(EvalAllGpu, HalfTreeParentFrontierBytes) {
  CheckFullEvaluation<8, 7, 128, 1, fss::group::Bytes, true>();
}
TEST(EvalAllGpu, HalfTreeParentFrontierUint) {
  CheckFullEvaluation<8, 7, 128, 1, Uint128Group, true>();
}
TEST(EvalAllGpu, DpfBatchBytes) {
  CheckFullEvaluation<8, 8, 128, 3, fss::group::Bytes, false>();
}
TEST(EvalAllGpu, DpfBatchUint) {
  CheckFullEvaluation<8, 8, 128, 3, Uint128Group, false>();
}
TEST(EvalAllGpu, HalfTreeBatchBytes) {
  CheckFullEvaluation<8, 7, 128, 3, fss::group::Bytes, true>();
}
TEST(EvalAllGpu, HalfTreeBatchUint) {
  CheckFullEvaluation<8, 7, 128, 3, Uint128Group, true>();
}
TEST(EvalAllGpu, DpfSubtreesUint) {
  CheckFullEvaluation<14, 12, 256, 1, Uint128Group, false>();
}
TEST(EvalAllGpu, HalfTreeSubtreesUint) {
  CheckFullEvaluation<14, 12, 256, 1, Uint128Group, true>();
}
TEST(EvalAllGpu, DpfAbove24Bits) {
  CheckFullEvaluation<25, 21, 128, 1, Uint128Group, false, true>();
}
TEST(EvalAllGpu, HalfTreeAbove24Bits) {
  CheckFullEvaluation<25, 20, 128, 1, Uint128Group, true, true>();
}
TEST(EvalAllGpu, HalfTreeWideInUint16) {
  CheckFullEvaluation<8, 7, 128, 1, fss::group::Bytes, true, false, uint16_t>();
}

// The device-key full-domain launchers serve the Python binding with its
// default launch shape (frontier 16, threads 256). They must cap the frontier
// below the leaf level like the public launcher; a frontier at the full depth
// makes the tail convert a leaf pair per frontier thread and doubles the rows.
TEST(EvalAllGpu, HalfTreeDeviceKeyDefaultFrontierMatchesHost) {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss_test::TestCcrHash;
  using Hash = fss_test::CcrBackend<Prg>;
  using Scheme = fss::HalfTreeDpf<16, fss::group::Bytes, Hash, uint16_t>;
  using Key = fss_test::ComponentKey<Scheme, uint16_t>;
  constexpr size_t n = size_t{1} << 16;
  constexpr size_t key_records = Scheme::kCwCount + 1;
  const int4 instance_seed = {11, 22, 33, 44};
  const int4 beta = {7, 19, 23, 42};
  Hash hash = fss_test::Ccr(Prg(0x12345678, static_cast<int>(0x9abcdef0u)));
  Scheme scheme{hash, instance_seed};
  Key key0, key1;
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  fss_test::GenPair(scheme, key0, key1, cuda::std::span<const int4, 2>(seeds, 2), 107, beta);

  // Key-major blocks: the key's seed components then one record holding the
  // packed control at its start, the layout the binding allocates.
  auto make_block = [&](const Key &key) {
    std::vector<int4> block(key_records, int4{0, 0, 0, 0});
    std::memcpy(block.data(), key.cws, Scheme::kCwCount * sizeof(int4));
    std::memcpy(block.data() + Scheme::kCwCount, &key.tr, sizeof(uint16_t));
    return block;
  };
  const std::vector<int4> single_block = make_block(key0);
  std::vector<int4> batch_blocks = make_block(key0);
  const std::vector<int4> second_block = make_block(key1);
  batch_blocks.insert(batch_blocks.end(), second_block.begin(), second_block.end());

  DeviceBuffer<int4> device_single, device_batch, device_s0s, device_ys;
  ASSERT_EQ(device_single.Allocate(key_records), cudaSuccess);
  ASSERT_EQ(device_single.Upload(single_block.data(), key_records), cudaSuccess);
  ASSERT_EQ(device_batch.Allocate(2 * key_records), cudaSuccess);
  ASSERT_EQ(device_batch.Upload(batch_blocks.data(), 2 * key_records), cudaSuccess);
  const int4 s0s[2] = {seeds[0], seeds[1]};
  ASSERT_EQ(device_s0s.Allocate(2), cudaSuccess);
  ASSERT_EQ(device_s0s.Upload(s0s, 2), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(n), cudaSuccess);
  std::vector<int4> host_ys(n);
  fss_test::EvalAllKey<0>(scheme, false, key0, cuda::std::span<int4>(host_ys.data(), n));
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  const auto *single_tr = reinterpret_cast<const uint16_t *>(device_single.data() + Scheme::kCwCount);
  // Public host-value path on the same component key, as a fixture reference.
  {
    SCOPED_TRACE("public host-value path");
    DeviceBuffer<int4> public_ys;
    ASSERT_EQ(public_ys.Allocate(n), cudaSuccess);
    ASSERT_EQ((scheme.template EvalAllGpu<16, 256>(false, seeds[0],
        cuda::std::span<const int4>(device_single.data(), Scheme::kCwCount), key0.tr,
        cuda::std::span<int4>(public_ys.data(), n), stream)), cudaSuccess);
    std::vector<int4> public_actual(n);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(public_actual.data(), public_ys.data(), n * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t x = 0; x < n; ++x) ExpectOutput(public_actual[x], host_ys[x], x);
  }
  ASSERT_EQ((fss::gpu::detail::HalfTreeDpfEvalAllGpuDeviceKey<16, fss::group::Bytes, Hash,
      uint16_t, 16, 256>(false, device_s0s.data(), device_single.data(), key_records,
      single_tr, 1, device_ys.data(), n, instance_seed, hash, stream)), cudaSuccess);
  std::vector<int4> actual(n);
  SCOPED_TRACE("device-key single path");
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual.data(), device_ys.data(), n * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t x = 0; x < n; ++x) ExpectOutput(actual[x], host_ys[x], x);

  DeviceBuffer<int4> batch_ys;
  ASSERT_EQ(batch_ys.Allocate(2 * n), cudaSuccess);
  ASSERT_EQ((fss::gpu::detail::HalfTreeDpfEvalAllGpuBatchDeviceKey<16, fss::group::Bytes, Hash,
      uint16_t, 16, 256>(false, device_s0s.data(), 2, device_batch.data(), 2 * key_records,
      key_records, Scheme::kCwCount, batch_ys.data(), 2 * n, instance_seed, hash, stream)),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  std::vector<int4> batch_actual(2 * n);
  ASSERT_EQ(cudaMemcpy(batch_actual.data(), batch_ys.data(), 2 * n * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  // Both batch rows share the party index; the second key keeps its own seed.
  std::vector<int4> host1_ys(n);
  fss_test::EvalAllKey<0>(scheme, false, key1, cuda::std::span<int4>(host1_ys.data(), n));
  for (size_t x = 0; x < n; ++x) {
    ExpectOutput(batch_actual[x], host_ys[x], x);
    ExpectOutput(batch_actual[n + x], host1_ys[x], x);
  }
}

template <int bits, int z, int threads, typename In = uint, int par_depth = 0>
void CheckVdpfGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Blake3 = fss::hash::Blake3;
  using Group = Uint128Group;
  using Scheme = fss::Vdpf<bits, Group, Prg, Blake3, Blake3, In>;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits + 1;
  // Blake3 copies the IV by value at construction.
  const int4 iv[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  Blake3 hash{cuda::std::span<const int4, 2>(iv, 2)};
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u)), hash, hash};
  const In alpha = 42;
  const int4 beta = {7, 19, 23, 42};
  int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  fss_test::ComponentKey<Scheme, In> k0, k1;
  ASSERT_EQ(fss_test::GenPair(scheme, k0, k1, cuda::std::span<int4, 2>(seeds, 2), alpha, beta), 0);
  DeviceBuffer<int4> device_cws;
  DeviceBuffer<int4> device_ys, device_cs;
  DeviceBuffer<int4> device_pi;
  ASSERT_EQ(device_cws.Allocate(stride), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(k0.cws, stride), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(n), cudaSuccess);
  ASSERT_EQ(device_pi.Allocate(4), cudaSuccess);
  ASSERT_EQ(device_cs.Allocate(4), cudaSuccess);
  ASSERT_EQ(device_cs.Upload(k0.cs, 4), cudaSuccess);

  std::vector<int4> host_ys(n), actual(n), first(n);
  cuda::std::array<int4, 4> pi_cpu, gpu_pi[2];
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    const fss_test::ComponentKey<Scheme, In> &key = party == 0 ? k0 : k1;
    fss_test::EvalAllMerkleKey<par_depth>(scheme, party, key, cuda::std::span<int4>(host_ys.data(), n), pi_cpu);
    ASSERT_EQ((scheme.template EvalAllMerkleGpu<z, threads>(
        party, key.s0, {device_cws.data(), stride}, key.tr,
        cuda::std::span<const int4, 4>(device_cs.data(), 4), {device_ys.data(), n},
        cuda::std::span<int4, 4>(device_pi.data(), 4))), cudaSuccess);
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

template <int bits, int out_bits, int z, int threads, int keys = 1, typename In = uint, int par_depth = 0>
void CheckPackedGpu() {
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
  constexpr int kLanes = Scheme::kLanes;
  constexpr size_t n = size_t{1} << bits;
  constexpr size_t blocks = size_t{1} << kDepth;
  const int4 hash_key = {11, 22, 33, 44};
  Scheme scheme{fss_test::Ccr(Prg(0x12345678, static_cast<int>(0x9abcdef0u))), hash_key};
  const uint64_t beta = 1;
  // Dense level-major layout the batch entry consumes: layer i of key k is
  // cws[i * keys + k] and the final correction word is the tail row.
  std::vector<int4> cws(keys * (kDepth + 1));
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[key], keys1[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), 42 + key, beta);
    scheme.Gen(cuda::std::span<int4>(cws.data() + key, static_cast<size_t>(kDepth) * keys + 1),
        cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), 42 + key, beta, keys);
  }
  DeviceBuffer<int4> device_cws;
  DeviceBuffer<int4> device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * blocks), cudaSuccess);
  std::vector<int4> actual(keys * blocks), first(keys * blocks), host_ys(blocks);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1) {
      ASSERT_EQ((scheme.template EvalAllGpuBatch<z, threads>(
          party, {device_seeds.data(), keys}, {device_cws.data(), cws.size()},
          {device_outputs.data(), keys * blocks})), cudaSuccess);
    } else {
      ASSERT_EQ((scheme.template EvalAllGpu<z, threads>(
          party, party_seeds[0], {device_cws.data(), kDepth + 1}, {device_outputs.data(), blocks})), cudaSuccess);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      const Key &party_key = party == 0 ? keys0[key] : keys1[key];
      fss_test::EvalAllKey<par_depth>(scheme, party, party_key, cuda::std::span<int4>(host_ys.data(), blocks));
      for (size_t j = 0; j < blocks; ++j) {
        ExpectOutput(actual[key * blocks + j], host_ys[j], j);
        if (party == 0) first[key * blocks + j] = actual[key * blocks + j];
      }
      if (party == 1) {
        for (size_t x = 0; x < n; ++x) {
          int4 sum = fss::util::Xor(first[key * blocks + x / kLanes], actual[key * blocks + x / kLanes]);
          EXPECT_EQ(Scheme::Extract(sum, static_cast<uint>(x)), x == static_cast<size_t>(42 + key) ? beta : 0) << "input " << x;
        }
      }
    }
  }
}

template <int bits, int z, int threads, int keys, typename Group, fss::DcfPred pred = fss::DcfPred::kLt, int alpha = 42,
    typename In = uint, int par_depth = 0, bool large = false>
void CheckDcfGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Scheme = fss::Dcf<bits, Group, fss::prg::ChaCha<4>, In, pred>;
  using Key = fss_test::ComponentKey<Scheme, In>;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits + 1;
  Scheme scheme{fss::prg::ChaCha<4>(0x12345678, static_cast<int>(0x9abcdef0u))};
  const int4 beta = {7, 19, 23, 42};
  // Dense level-major layout the batch entry consumes: layer i of key k is
  // cws[i * keys + k] and vws[i * keys + k], with the packed control in trs[k].
  std::vector<int4> cws(keys * stride), vws(keys * stride);
  std::vector<In> trs(keys);
  std::vector<Key> keys0(keys), keys1(keys);
  std::vector<int4> seeds(2 * keys), party_seeds(keys);
  for (int key = 0; key < keys; ++key) {
    seeds[2 * key] = {11 + key, 22, 33, 44};
    seeds[2 * key + 1] = {55 + key, 66, 77, 88};
    fss_test::GenPair(scheme, keys0[key], keys1[key], cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), static_cast<In>(alpha + key), beta);
    scheme.Gen(cuda::std::span<int4>(cws.data() + key, static_cast<size_t>(bits) * keys + 1),
        cuda::std::span<int4>(vws.data() + key, static_cast<size_t>(bits) * keys + 1), trs[key],
        cuda::std::span<const int4, 2>(seeds.data() + 2 * key, 2), static_cast<In>(alpha + key),
        beta, keys);
  }
  DeviceBuffer<int4> device_cws, device_vws;
  DeviceBuffer<In> device_tr;
  DeviceBuffer<int4> device_seeds, device_outputs;
  ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(cws.data(), cws.size()), cudaSuccess);
  ASSERT_EQ(device_vws.Allocate(vws.size()), cudaSuccess);
  ASSERT_EQ(device_vws.Upload(vws.data(), vws.size()), cudaSuccess);
  ASSERT_EQ(device_tr.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_tr.Upload(trs.data(), keys), cudaSuccess);
  ASSERT_EQ(device_seeds.Allocate(keys), cudaSuccess);
  ASSERT_EQ(device_outputs.Allocate(keys * n), cudaSuccess);
  std::vector<int4> actual(keys * n), first(keys * n), host_ys(large ? n : 0);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    for (int key = 0; key < keys; ++key) party_seeds[key] = seeds[2 * key + party];
    ASSERT_EQ(device_seeds.Upload(party_seeds.data(), keys), cudaSuccess);
    if constexpr (keys > 1) {
      ASSERT_EQ((scheme.template EvalAllGpuBatch<z, threads>(
          party, {device_seeds.data(), keys}, {device_cws.data(), cws.size()},
          {device_vws.data(), vws.size()}, {device_tr.data(), keys},
          {device_outputs.data(), keys * n})), cudaSuccess);
    } else {
      ASSERT_EQ((scheme.template EvalAllGpu<z, threads>(
          party, party_seeds[0], {device_cws.data(), stride}, {device_vws.data(), stride}, trs[0],
          {device_outputs.data(), n})), cudaSuccess);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs.data(), actual.size() * sizeof(int4), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int key = 0; key < keys; ++key) {
      SCOPED_TRACE(::testing::Message() << "key " << key);
      const Key &party_key = party == 0 ? keys0[key] : keys1[key];
      if constexpr (large)
        fss_test::EvalAllKey<par_depth>(scheme, static_cast<bool>(party), party_key, cuda::std::span<int4>(host_ys.data(), n));
      for (size_t x = 0; x < n; ++x) {
        int4 expected;
        if constexpr (large) expected = host_ys[x];
        else expected = fss_test::EvalKey(scheme, party, party_key, x);
        ExpectOutput(actual[key * n + x], expected, x);
        if (party == 0) first[key * n + x] = actual[key * n + x];
        else {
          const bool hit = pred == fss::DcfPred::kLt ? x < static_cast<size_t>(alpha + key) : x > static_cast<size_t>(alpha + key);
          int4 sum = (Group::From(first[key * n + x]) + Group::From(actual[key * n + x])).Into();
          ExpectOutput(sum, Group::From(hit ? beta : int4{0, 0, 0, 0}).Into(), x);
        }
      }
    }
  }
}

template <int bits, int z, int threads, typename In = uint, int par_depth = 0>
void CheckGrottoGpu() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    GTEST_SKIP() << "no CUDA device available";
  }
  ASSERT_EQ(status, cudaSuccess);
  using Prg = fss::prg::ChaCha<2>;
  using Scheme = fss::GrottoDcf<bits, Prg, In>;
  constexpr size_t n = size_t{1} << bits;
  constexpr int stride = bits + 1;
  Scheme scheme{Prg(0x12345678, static_cast<int>(0x9abcdef0u))};
  int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  const uint alpha = 42;
  fss_test::ComponentKey<Scheme, In> k0, k1;
  fss_test::GenPair(scheme, k0, k1, cuda::std::span<const int4, 2>(seeds), alpha);
  DeviceBuffer<int4> device_cws;
  ASSERT_EQ(device_cws.Allocate(stride), cudaSuccess);
  ASSERT_EQ(device_cws.Upload(k0.cws, stride), cudaSuccess);

  auto host_ys = std::make_unique<bool[]>(n);
  auto first = std::make_unique<bool[]>(n);
  DeviceBuffer<bool> device_p, device_ys;
  ASSERT_EQ(device_p.Allocate(2 * n - 1), cudaSuccess);
  ASSERT_EQ(device_ys.Allocate(n), cudaSuccess);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(::testing::Message() << "party " << party);
    const fss_test::ComponentKey<Scheme, In> &key = party == 0 ? k0 : k1;
    auto host_tree = fss_test::PreprocessKey<par_depth>(scheme, static_cast<bool>(party), key);
    ASSERT_EQ((scheme.template PreprocessGpu<z, threads>(
        static_cast<bool>(party), key.s0, {device_cws.data(), stride}, key.tr,
        {device_p.data(), 2 * n - 1})), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto actual_p = std::make_unique<bool[]>(2 * n - 1);
    ASSERT_EQ(
        cudaMemcpy(actual_p.get(), device_p.data(), (2 * n - 1) * sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < 2 * n - 1; ++i) EXPECT_EQ(actual_p[i], host_tree.View().p[i]) << "node " << i;

    fss_test::EvalAllKey<par_depth>(scheme, static_cast<bool>(party), key, cuda::std::span<bool>(host_ys.get(), n));
    ASSERT_EQ((scheme.template EvalAllGpu<z, threads>(
        static_cast<bool>(party), key.s0, {device_cws.data(), stride}, key.tr,
        {device_ys.data(), n})), cudaSuccess);
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
  CheckPackedGpu<12, 8, 8, 128>();
}
TEST(EvalAllGpu, PackedOneBit) {
  CheckPackedGpu<12, 1, 5, 32>();
}
TEST(EvalAllGpu, PackedBatch) {
  CheckPackedGpu<12, 8, 8, 128, 3>();
}
TEST(EvalAllGpu, PackedAbove24Bits) {
  CheckPackedGpu<25, 8, 17, 128>();
}
TEST(EvalAllGpu, PackedWideInUint16) {
  CheckPackedGpu<12, 8, 8, 128, 1, uint16_t>();
}

TEST(EvalAllGpu, DcfLeafFrontierBytes) {
  CheckDcfGpu<8, 8, 128, 1, fss::group::Bytes>();
}
TEST(EvalAllGpu, DcfLeafFrontierUint) {
  CheckDcfGpu<8, 8, 128, 1, Uint128Group>();
}
TEST(EvalAllGpu, DcfSubtreesUint) {
  CheckDcfGpu<14, 12, 256, 1, Uint128Group>();
}
TEST(EvalAllGpu, DcfBatchUint) {
  CheckDcfGpu<8, 8, 128, 3, Uint128Group>();
}

TEST(EvalAllGpu, DcfGtLeafFrontierBytes) {
  CheckDcfGpu<8, 8, 128, 1, fss::group::Bytes, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtLeafFrontierUint) {
  CheckDcfGpu<8, 8, 128, 1, Uint128Group, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtSubtreesUint) {
  CheckDcfGpu<14, 12, 256, 1, Uint128Group, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtBatchUint) {
  CheckDcfGpu<8, 8, 128, 3, Uint128Group, fss::DcfPred::kGt>();
}
TEST(EvalAllGpu, DcfGtAlphaZero) {
  CheckDcfGpu<8, 8, 128, 1, Uint128Group, fss::DcfPred::kGt, 0>();
}
TEST(EvalAllGpu, DcfGtAlphaMax) {
  CheckDcfGpu<8, 8, 128, 1, Uint128Group, fss::DcfPred::kGt, 255>();
}
TEST(EvalAllGpu, DcfAbove24Bits) {
  CheckDcfGpu<25, 21, 128, 1, Uint128Group, fss::DcfPred::kLt, 42, uint, 0, true>();
}
TEST(EvalAllGpu, DcfGtAbove24Bits) {
  CheckDcfGpu<25, 21, 128, 1, Uint128Group, fss::DcfPred::kGt, 42, uint, 0, true>();
}
TEST(EvalAllGpu, DcfWideInUint16) {
  CheckDcfGpu<8, 8, 128, 1, Uint128Group, fss::DcfPred::kLt, 42, uint16_t>();
}

TEST(EvalAllGpu, GrottoSmall) {
  CheckGrottoGpu<8, 8, 128>();
}
TEST(EvalAllGpu, GrottoSubtrees) {
  CheckGrottoGpu<14, 12, 256>();
}
TEST(EvalAllGpu, GrottoAbove24Bits) {
  CheckGrottoGpu<25, 21, 128>();
}
TEST(EvalAllGpu, GrottoWideInUint16) {
  CheckGrottoGpu<10, 10, 128, uint16_t>();
}

TEST(EvalAllGpu, ConfiguredDepths) {
  CheckFullEvaluation<8, 8, 128, 2, fss::group::Bytes, false>();
  CheckFullEvaluation<8, 7, 128, 2, fss::group::Bytes, true>();
  CheckPackedGpu<12, 8, 8, 128, 2, uint, 0>();
  CheckPackedGpu<12, 8, 8, 128, 2, uint, 2>();
  CheckVdpfGpu<10, 10, 256, uint, 0>();
  CheckVdpfGpu<10, 10, 256, uint, 2>();
  CheckGrottoGpu<8, 8, 128, uint, 0>();
  CheckGrottoGpu<8, 8, 128, uint, 2>();
}

}  // namespace
