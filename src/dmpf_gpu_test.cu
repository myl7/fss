#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include <fss/dmpf.cuh>
#include <fss/eval_all_gpu.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/point_eval_gpu.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/chacha_feistel.cuh>

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
  cudaError_t Download(T *dest, size_t count) {
    return cudaMemcpy(dest, data_, count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  T *data() const { return data_; }
 private:
  T *data_ = nullptr;
};

constexpr int kNonce0 = 0x12345678, kNonce1 = static_cast<int>(0x9abcdef0u);

void ExpectOutput(int4 actual, int4 expected, size_t input) {
  EXPECT_EQ(actual.x, expected.x) << "input " << input;
  EXPECT_EQ(actual.y, expected.y) << "input " << input;
  EXPECT_EQ(actual.z, expected.z) << "input " << input;
  EXPECT_EQ(actual.w, expected.w) << "input " << input;
}

template <typename Group>
class DmpfGpuTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using Prp = fss::prp::ChaChaFeistel;
  using DmpfType = fss::Dmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Prp, uint16_t>;
  using Cw = typename DmpfType::InnerDpf::Cw;

  static constexpr int m = DmpfType::m;
  static constexpr size_t kDomain = size_t{1} << kInBits;
  static constexpr size_t kBucketDomain = size_t{1} << kBucketBits;

  uint16_t alphas[kT] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
      1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000};
  int4 betas[kT];

  Prg prg;
  Prp prp_;

  DmpfGpuTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    for (int i = 0; i < kT; ++i) {
      betas[i] = {(i + 1) * 11, 0, 0, 0};
    }
  }

  int4 RandomSeed(std::mt19937 &gen) {
    std::uniform_int_distribution<int> dis;
    return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void GenKeys(typename DmpfType::Key &k0, typename DmpfType::Key &k1, int4 &sigma) {
    DmpfType dmpf{prg, prp_};
    std::random_device rd;
    std::mt19937 gen(rd());

    sigma = RandomSeed(gen);
    cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = {RandomSeed(gen), RandomSeed(gen)};
    }
    ASSERT_EQ(dmpf.Gen(k0, k1, sigma, cuda::std::span<cuda::std::array<int4, 2>, m>(s0s),
        cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT), 0);
  }

  // Flatten the active bucket keys into the per-key layout of
  // DpfEvalAllGpuBatch and the per-bucket seed list.
  void PackKey(const typename DmpfType::Key &key, std::vector<int4> &s0s, std::vector<Cw> &cws) {
    int m_rt = key.m_rt;
    s0s.resize(m_rt);
    cws.resize(static_cast<size_t>(m_rt) * (kBucketBits + 1));
    for (int i = 0; i < m_rt; ++i) {
      s0s[i] = key.bks[i].s0;
      memcpy(cws.data() + static_cast<size_t>(i) * (kBucketBits + 1), key.bks[i].cws, sizeof(Cw) * (kBucketBits + 1));
    }
  }

  void TestBatchEval() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
    typename DmpfType::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    DmpfType dmpf{prg, prp_};
    int m_rt = k0.m_rt;
    int b_rt = k0.b_size_rt;

    // All alphas plus random non-alpha points.
    std::vector<uint16_t> xs(alphas, alphas + kT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);
    constexpr int kRandoms = 100;
    while (static_cast<int>(xs.size()) < kT + kRandoms) {
      uint16_t x = dis(gen);
      bool is_alpha = std::any_of(alphas, alphas + kT, [x](uint16_t a) { return a == x; });
      if (!is_alpha) xs.push_back(x);
    }
    int n = static_cast<int>(xs.size());

    std::vector<int4> h_s0s[2];
    std::vector<Cw> h_cws;
    PackKey(k0, h_s0s[0], h_cws);
    PackKey(k1, h_s0s[1], h_cws);

    DeviceBuffer<uint16_t> d_xs;
    DeviceBuffer<int4> d_ys[2];
    DeviceBuffer<int4> d_s0s[2];
    DeviceBuffer<Cw> d_cws;
    DeviceBuffer<int4> d_cw_s, d_out_cw;
    DeviceBuffer<fss::gpu::ExtBits<kBucketBits>> d_extra;
    ASSERT_EQ(d_xs.Allocate(n), cudaSuccess);
    ASSERT_EQ(d_xs.Upload(xs.data(), n), cudaSuccess);
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(d_ys[p].Allocate(n), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Allocate(m_rt), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Upload(h_s0s[p].data(), m_rt), cudaSuccess);
    }
    ASSERT_EQ(d_cws.Allocate(h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cws.Upload(h_cws.data(), h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cw_s.Allocate(static_cast<size_t>(kBucketBits) * m_rt), cudaSuccess);
    ASSERT_EQ(d_out_cw.Allocate(m_rt), cudaSuccess);
    ASSERT_EQ(d_extra.Allocate(m_rt), cudaSuccess);

    // Both parties share the correction words, so one repack serves both.
    fss::gpu::DpfRepackGpu<kBucketBits, Group, Prg, uint>(d_cws.data(), m_rt, d_cw_s.data(), d_extra.data(),
        d_out_cw.data());

    for (int p = 0; p < 2; ++p) {
      fss::gpu::DmpfBatchEvalGpu(p != 0, d_xs.data(), d_ys[p].data(), n, sigma, m_rt, b_rt, d_s0s[p].data(),
          d_cw_s.data(), d_extra.data(), d_out_cw.data(), dmpf);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<int4> ys[2];
    ys[0].resize(n);
    ys[1].resize(n);
    ASSERT_EQ(d_ys[0].Download(ys[0].data(), n), cudaSuccess);
    ASSERT_EQ(d_ys[1].Download(ys[1].data(), n), cudaSuccess);

    // Reconstruction: betas at the alphas, zero elsewhere.
    for (int i = 0; i < n; ++i) {
      int4 expected = {0, 0, 0, 0};
      for (int j = 0; j < kT; ++j) {
        if (xs[i] == alphas[j]) {
          expected = betas[j];
          expected.w &= ~1;
        }
      }
      auto result = Group::From(ys[0][i]) + Group::From(ys[1][i]);
      ExpectOutput(result.Into(), expected, xs[i]);
    }

    // Bit-exact against the host batch evaluation per party.
    std::vector<int4> host_ys(n);
    for (int p = 0; p < 2; ++p) {
      dmpf.BatchEval(p != 0, p == 0 ? k0 : k1, cuda::std::span<const uint16_t>(xs),
          cuda::std::span<int4>(host_ys));
      for (int i = 0; i < n; ++i) ExpectOutput(ys[p][i], host_ys[i], xs[i]);
    }
  }

  void TestEvalAll() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
    typename DmpfType::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    DmpfType dmpf{prg, prp_};
    int m_rt = k0.m_rt;
    int b_rt = k0.b_size_rt;

    std::vector<int4> h_s0s[2];
    std::vector<Cw> h_cws;
    PackKey(k0, h_s0s[0], h_cws);
    PackKey(k1, h_s0s[1], h_cws);

    DeviceBuffer<int4> d_ys[2];
    DeviceBuffer<int4> d_s0s[2];
    DeviceBuffer<Cw> d_cws;
    DeviceBuffer<int4> d_bucket_ys;
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(d_ys[p].Allocate(kDomain), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Allocate(m_rt), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Upload(h_s0s[p].data(), m_rt), cudaSuccess);
    }
    ASSERT_EQ(d_cws.Allocate(h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cws.Upload(h_cws.data(), h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_bucket_ys.Allocate(static_cast<size_t>(m_rt) * kBucketDomain), cudaSuccess);

    for (int p = 0; p < 2; ++p) {
      fss::gpu::DmpfEvalAllGpu(p != 0, sigma, m_rt, b_rt, d_s0s[p].data(), d_cws.data(), d_bucket_ys.data(),
          d_ys[p].data(), dmpf);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<int4> ys[2];
    ys[0].resize(kDomain);
    ys[1].resize(kDomain);
    ASSERT_EQ(d_ys[0].Download(ys[0].data(), kDomain), cudaSuccess);
    ASSERT_EQ(d_ys[1].Download(ys[1].data(), kDomain), cudaSuccess);

    std::vector<int4> expected(kDomain, int4{0, 0, 0, 0});
    for (int i = 0; i < kT; ++i) expected[alphas[i]] = betas[i];

    for (size_t i = 0; i < kDomain; ++i) {
      auto result = Group::From(ys[0][i]) + Group::From(ys[1][i]);
      int4 e = expected[i];
      e.w &= ~1;
      ExpectOutput(result.Into(), e, i);
    }

    // Bit-exact against the host full-domain evaluation per party.
    std::vector<int4> host_ys(kDomain);
    for (int p = 0; p < 2; ++p) {
      dmpf.EvalAll(p != 0, p == 0 ? k0 : k1, cuda::std::span<int4>(host_ys));
      for (size_t i = 0; i < kDomain; ++i) ExpectOutput(ys[p][i], host_ys[i], i);
    }
  }
};

using DmpfBytesGpuTest = DmpfGpuTest<fss::group::Bytes>;
using DmpfUint128GpuTest = DmpfGpuTest<Uint127Group>;

TEST_F(DmpfBytesGpuTest, BatchEval) {
  TestBatchEval();
}

TEST_F(DmpfBytesGpuTest, EvalAll) {
  TestEvalAll();
}

TEST_F(DmpfUint128GpuTest, BatchEval) {
  TestBatchEval();
}

TEST_F(DmpfUint128GpuTest, EvalAll) {
  TestEvalAll();
}

}  // namespace
