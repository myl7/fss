#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include <fss/dmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/chacha_feistel.cuh>
#include "binding_gpu_eval.cuh"
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
  cudaError_t Download(T *dest, size_t count) {
    return cudaMemcpy(dest, data_, count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  T *data() const {
    return data_;
  }

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

template <typename Group, int ch_lambda = 80, int par_depth = 0>
class DmpfGpuTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using Prp = fss::prp::ChaChaFeistel;
  using DmpfType = fss::Dmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Prp, uint16_t, ch_lambda>;
  using Cw = typename DmpfType::Cw;

  static constexpr int m = DmpfType::kBucketCount;
  static constexpr size_t kDomain = size_t{1} << kInBits;

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
    cuda::std::array<int4, size_t{m} * 2> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = RandomSeed(gen);
      s0s[size_t{m} + i] = RandomSeed(gen);
    }
    ASSERT_EQ(fss_test::GenMulti(dmpf, k0, k1, sigma, cuda::std::span<const int4>(s0s),
                  cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT),
        0);
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
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
    typename DmpfType::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    DmpfType dmpf{prg, prp_};
    int m_rt = k0.m_rt;
    __uint128_t b_rt = k0.b_size_rt;

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
    ASSERT_EQ(d_xs.Allocate(n), cudaSuccess);
    ASSERT_EQ(d_xs.Upload(xs.data(), n), cudaSuccess);
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(d_ys[p].Allocate(n), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Allocate(m_rt), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Upload(h_s0s[p].data(), m_rt), cudaSuccess);
    }
    ASSERT_EQ(d_cws.Allocate(h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cws.Upload(h_cws.data(), h_cws.size()), cudaSuccess);

    // Both parties share the correction words, so one upload serves both.
    const cuda::std::span<const Cw> cws(d_cws.data(), h_cws.size());
    fss::gpu::detail::DmpfBatchEvalGpu(false, cuda::std::span<const uint16_t>(), nullptr, 0,
        {}, 0, 0, {}, {}, dmpf);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    for (int p = 0; p < 2; ++p) {
      fss::gpu::detail::DmpfBatchEvalGpu(p != 0, {d_xs.data(), static_cast<size_t>(n)}, d_ys[p].data(), n,
          sigma, m_rt, b_rt, {d_s0s[p].data(), static_cast<size_t>(m_rt)}, cws, dmpf);
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
      fss_test::BatchEvalKey(dmpf, p != 0, p == 0 ? k0 : k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(host_ys));
      for (int i = 0; i < n; ++i) ExpectOutput(ys[p][i], host_ys[i], xs[i]);
    }
  }

  void TestEvalAll() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
    typename DmpfType::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    DmpfType dmpf{prg, prp_};
    int m_rt = k0.m_rt;
    __uint128_t b_rt = k0.b_size_rt;

    std::vector<int4> h_s0s[2];
    std::vector<Cw> h_cws;
    PackKey(k0, h_s0s[0], h_cws);
    PackKey(k1, h_s0s[1], h_cws);

    DeviceBuffer<int4> d_ys[2];
    DeviceBuffer<int4> d_s0s[2];
    DeviceBuffer<Cw> d_cws;
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(d_ys[p].Allocate(kDomain), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Allocate(m_rt), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Upload(h_s0s[p].data(), m_rt), cudaSuccess);
    }
    ASSERT_EQ(d_cws.Allocate(h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cws.Upload(h_cws.data(), h_cws.size()), cudaSuccess);

    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(dmpf.EvalAllGpu(
          p != 0, sigma, b_rt, {d_s0s[p].data(), static_cast<size_t>(m_rt)},
          {d_cws.data(), h_cws.size()},
          {d_ys[p].data(), kDomain}), cudaSuccess);
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
      fss_test::EvalAllKey<par_depth>(dmpf, p != 0, p == 0 ? k0 : k1, cuda::std::span<int4>(host_ys));
      for (size_t i = 0; i < kDomain; ++i) ExpectOutput(ys[p][i], host_ys[i], i);
    }
  }
};

using DmpfBytesGpuTest = DmpfGpuTest<fss::group::Bytes>;
using DmpfUint128GpuTest = DmpfGpuTest<fss::group::Uint128<>>;

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

using DmpfCustomSecurityGpuTest = DmpfGpuTest<fss::group::Bytes, 60, 0>;
TEST_F(DmpfCustomSecurityGpuTest, BatchEval) {
  TestBatchEval();
}
TEST_F(DmpfCustomSecurityGpuTest, EvalAll) {
  TestEvalAll();
}

}  // namespace

namespace {
namespace bucket_bound {

// The bucket-size boundary of the GPU validation entries, at both ends.
// B = ceil(3*2^in_bits/kBucketCount) is the smallest bucket size that keeps
// every located bucket below the allocated bucket count, so one below B must
// be rejected and B itself must pass validation. Symmetrically, 2^bucket_bits
// is the largest representable bucket size: it must pass and one above it must
// be rejected, which is the only upper defense of the C++ GPU entries. Both
// entries validate before any CUDA call, so the reject halves run without a
// CUDA device.
using Group = fss::group::Bytes;
using Prg = fss::prg::ChaCha<2>;
using Prp = fss::prp::ChaChaFeistel;
constexpr int kInBits = 12;
constexpr int kMaxPoints = 30;
constexpr int kBucketBits = 10;
constexpr int kBuckets = 53;  // ChBucket(30, 80)
constexpr __uint128_t kBound = 232;  // ceil(3*2^12/53)
constexpr __uint128_t kUpper = __uint128_t{1} << kBucketBits;  // 2^bucket_bits
constexpr size_t kRows = kBuckets + 8;  // Above ceil(3*2^12/(kBound-1)) = 54 buckets.
using DmpfType = fss::Dmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Prp, uint16_t>;

static_assert(DmpfType::kBucketCount == kBuckets);
static_assert(kBound == (3 * (__uint128_t{1} << kInBits) + kBuckets - 1) / kBuckets);
static_assert(kBound < kUpper);
// The local spelling of the candidate count must match the scheme's private
// one, or the derived bound stops being the acceptance boundary.
static_assert(
    fss::cuckoo_hash::detail::ValidBucketSize(kBound, DmpfType::kBucketSize, kBucketBits) &&
    !fss::cuckoo_hash::detail::ValidBucketSize(kBound - 1, DmpfType::kBucketSize, kBucketBits));

TEST(DmpfGpuBucketBound, ValidMultiKeyBoundary) {
  static_assert(DmpfType::kBucketCount == kBuckets);
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<DmpfType::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  auto valid = [&](__uint128_t b_size_rt) {
    return fss::gpu::detail::ValidMultiKey<DmpfType>(b_size_rt, s0s.size(), cws.size(),
        s0s.data(), cws.data());
  };
  EXPECT_TRUE(valid(kBound)) << "b_size_rt = ceil(3*2^in_bits/kBucketCount)";
  EXPECT_FALSE(valid(kBound - 1)) << "b_size_rt = ceil(3*2^in_bits/kBucketCount) - 1";
  EXPECT_TRUE(valid(kUpper)) << "b_size_rt = 2^bucket_bits";
  EXPECT_FALSE(valid(kUpper + 1)) << "b_size_rt = 2^bucket_bits + 1";
}

TEST(DmpfGpuBucketBound, EvalAllGpuBoundary) {
  DmpfType dmpf{Prg{kNonce0, kNonce1}, Prp{}};
  std::vector<int4> s0s(kRows, int4{0, 0, 0, 0});
  std::vector<DmpfType::Cw> cws(kRows * (kBucketBits + 1));
  std::vector<int4> ys(size_t{1} << kInBits);
  auto run = [&](__uint128_t b_size_rt) {
    return dmpf.EvalAllGpu(false, int4{0, 0, 0, 0}, b_size_rt,
        cuda::std::span<const int4>(s0s.data(), kRows),
        cuda::std::span<const DmpfType::Cw>(cws.data(), cws.size()),
        cuda::std::span<int4>(ys.data(), ys.size()));
  };
  // The too-small key's derived bucket count is 54, so the kRows = 61 rows
  // cannot mask the lower check: only the bucket-size check rejects it here.
  EXPECT_EQ(run(kBound - 1), cudaErrorInvalidValue) << "b_size_rt = ceil(3*2^in_bits/kBucketCount) - 1";
  EXPECT_EQ(run(kUpper + 1), cudaErrorInvalidValue) << "b_size_rt = 2^bucket_bits + 1";
  // The accepted keys pass validation and continue into CUDA. Device buffers
  // keep that continuation meaningful on a real device; without a device the
  // call stops at the allocation with a device error. Either way the bucket
  // size itself must not be rejected.
  int devices = 0;
  const bool have_device = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
  if (have_device) {
    DeviceBuffer<int4> device_s0s;
    DeviceBuffer<DmpfType::Cw> device_cws;
    DeviceBuffer<int4> device_ys;
    ASSERT_EQ(device_s0s.Allocate(kRows), cudaSuccess);
    ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
    ASSERT_EQ(device_ys.Allocate(ys.size()), cudaSuccess);
    auto run_device = [&](__uint128_t b_size_rt) {
      return dmpf.EvalAllGpu(false, int4{0, 0, 0, 0}, b_size_rt,
          cuda::std::span<const int4>(device_s0s.data(), kRows),
          cuda::std::span<const DmpfType::Cw>(device_cws.data(), cws.size()),
          cuda::std::span<int4>(device_ys.data(), ys.size()));
    };
    EXPECT_NE(run_device(kBound), cudaErrorInvalidValue) << "b_size_rt = ceil(3*2^in_bits/kBucketCount)";
    EXPECT_NE(run_device(kUpper), cudaErrorInvalidValue) << "b_size_rt = 2^bucket_bits";
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    EXPECT_NE(run(kBound), cudaErrorInvalidValue) << "b_size_rt = ceil(3*2^in_bits/kBucketCount), no device";
    EXPECT_NE(run(kUpper), cudaErrorInvalidValue) << "b_size_rt = 2^bucket_bits, no device";
  }
}

}  // namespace bucket_bound
}  // namespace

// The span checks run on the host before any CUDA call.
TEST(DmpfBatchEvalGpu, RejectsInvalidSpansBeforeLaunch) {
  using Scheme = fss::Dmpf<16, 30, 14, fss::group::Bytes, fss::prg::ChaCha<2>,
      fss::prp::ChaChaFeistel, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<2>(1, 2), fss::prp::ChaChaFeistel{}};
  const int4 sigma = {1, 2, 3, 4};
  const int4 s0s[1] = {{5, 6, 7, 8}};
  Scheme::Cw cws[15] = {};
  const uint16_t xs[1] = {0};
  int4 ys[1] = {};

  // Short xs.
  EXPECT_THROW(fss::gpu::detail::DmpfBatchEvalGpu<4>(false, cuda::std::span<const uint16_t>(),
                   ys, 1, sigma, 1, Scheme::kBucketSize, cuda::std::span<const int4>(s0s, 1),
                   cuda::std::span<const Scheme::Cw>(cws, 15), scheme),
      std::invalid_argument);
  // Short cws.
  EXPECT_THROW(fss::gpu::detail::DmpfBatchEvalGpu<4>(false, cuda::std::span<const uint16_t>(xs, 1),
                   ys, 1, sigma, 1, Scheme::kBucketSize, cuda::std::span<const int4>(s0s, 1),
                   cuda::std::span<const Scheme::Cw>(cws, 1), scheme),
      std::invalid_argument);
  // Short s0s.
  EXPECT_THROW(fss::gpu::detail::DmpfBatchEvalGpu<4>(false, cuda::std::span<const uint16_t>(xs, 1),
                   ys, 1, sigma, 1, Scheme::kBucketSize, cuda::std::span<const int4>(),
                   cuda::std::span<const Scheme::Cw>(cws, 15), scheme),
      std::invalid_argument);
}

// Zero points write no output, so empty spans and a null ys are a valid no-op,
// while a negative count is still rejected. The check runs before any CUDA
// call, so no device is needed.
TEST(DmpfBatchEvalGpu, ZeroPointsWithEmptySpansIsNoOp) {
  using Scheme = fss::Dmpf<16, 30, 14, fss::group::Bytes, fss::prg::ChaCha<2>,
      fss::prp::ChaChaFeistel, uint16_t>;
  Scheme scheme{fss::prg::ChaCha<2>(1, 2), fss::prp::ChaChaFeistel{}};
  EXPECT_NO_THROW(fss::gpu::detail::DmpfBatchEvalGpu<4>(false, cuda::std::span<const uint16_t>(),
      nullptr, 0, int4{}, 1, Scheme::kBucketSize, cuda::std::span<const int4>(),
      cuda::std::span<const Scheme::Cw>(), scheme));
  EXPECT_THROW(fss::gpu::detail::DmpfBatchEvalGpu<4>(false, cuda::std::span<const uint16_t>(),
      nullptr, -1, int4{}, 1, Scheme::kBucketSize, cuda::std::span<const int4>(),
      cuda::std::span<const Scheme::Cw>(), scheme), std::invalid_argument);
}
