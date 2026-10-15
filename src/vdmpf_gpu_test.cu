#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/point_eval_gpu.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/chacha_feistel.cuh>
#include <fss/vdmpf.cuh>

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
constexpr int4 kHashIv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};

void ExpectOutput(int4 actual, int4 expected, size_t input) {
  EXPECT_EQ(actual.x, expected.x) << "input " << input;
  EXPECT_EQ(actual.y, expected.y) << "input " << input;
  EXPECT_EQ(actual.z, expected.z) << "input " << input;
  EXPECT_EQ(actual.w, expected.w) << "input " << input;
}

template <typename Group>
class VdmpfGpuTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using Prp = fss::prp::ChaChaFeistel;
  using Blake3 = fss::hash::Blake3;
  using VdmpfT = fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Blake3, Blake3, Prp, uint16_t>;
  using Cw = typename VdmpfT::InnerVdpf::Cw;

  static constexpr int m = VdmpfT::m;

  uint16_t alphas[kT] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
      1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000};
  int4 betas[kT];

  Prg prg;
  Prp prp_;
  Blake3 xor_hash{cuda::std::span<const int4, 2>(kHashIv, 2)};
  Blake3 hash{cuda::std::span<const int4, 2>(kHashIv, 2)};

  VdmpfGpuTest() : prg(kNonce0, kNonce1) {}

  void SetUp() override {
    for (int i = 0; i < kT; ++i) {
      betas[i] = {(i + 1) * 11, 0, 0, 0};
    }
  }

  int4 RandomSeed(std::mt19937 &gen) {
    std::uniform_int_distribution<int> dis;
    return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
  }

  void GenKeys(typename VdmpfT::Key &k0, typename VdmpfT::Key &k1, int4 &sigma) {
    VdmpfT vdmpf{prg, xor_hash, hash, prp_};
    std::random_device rd;
    std::mt19937 gen(rd());

    sigma = RandomSeed(gen);
    cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = {RandomSeed(gen), RandomSeed(gen)};
    }
    ASSERT_EQ(vdmpf.Gen(k0, k1, sigma, cuda::std::span<cuda::std::array<int4, 2>, m>(s0s),
        cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT), 0);
  }

  void TestBatchEval() {
    typename VdmpfT::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    VdmpfT vdmpf{prg, xor_hash, hash, prp_};
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
    std::vector<int4> h_ocws(m_rt);
    std::vector<cuda::std::array<int4, 4>> h_cs(m);
    h_cws.resize(static_cast<size_t>(m_rt) * kBucketBits);
    for (int i = 0; i < m_rt; ++i) {
      h_ocws[i] = k0.bks[i].ocw;
      for (int j = 0; j < kBucketBits; ++j) {
        h_cws[static_cast<size_t>(i) * kBucketBits + j] = k0.bks[i].cws[j];
      }
    }
    for (int i = 0; i < 2; ++i) {
      h_s0s[i].resize(m_rt);
      for (int j = 0; j < m_rt; ++j) {
        h_s0s[i][j] = (i ? k1 : k0).bks[j].s0;
      }
    }
    // cs is shared between the parties; the fold covers all m buckets.
    for (int i = 0; i < m; ++i) h_cs[i] = k0.bks[i].cs;

    DeviceBuffer<uint16_t> d_xs;
    DeviceBuffer<int4> d_ys[2];
    DeviceBuffer<cuda::std::array<int4, 4>> d_pi[2];
    DeviceBuffer<int4> d_s0s[2];
    DeviceBuffer<Cw> d_cws;
    DeviceBuffer<int4> d_cw_s, d_ocws;
    DeviceBuffer<cuda::std::array<int4, 4>> d_cs;
    DeviceBuffer<fss::gpu::ExtBits<kBucketBits>> d_extra;
    ASSERT_EQ(d_xs.Allocate(n), cudaSuccess);
    ASSERT_EQ(d_xs.Upload(xs.data(), n), cudaSuccess);
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(d_ys[p].Allocate(n), cudaSuccess);
      ASSERT_EQ(d_pi[p].Allocate(1), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Allocate(m_rt), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Upload(h_s0s[p].data(), m_rt), cudaSuccess);
    }
    ASSERT_EQ(d_cws.Allocate(h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cws.Upload(h_cws.data(), h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_ocws.Allocate(m_rt), cudaSuccess);
    ASSERT_EQ(d_ocws.Upload(h_ocws.data(), m_rt), cudaSuccess);
    ASSERT_EQ(d_cs.Allocate(m), cudaSuccess);
    ASSERT_EQ(d_cs.Upload(h_cs.data(), m), cudaSuccess);
    ASSERT_EQ(d_cw_s.Allocate(static_cast<size_t>(kBucketBits) * m_rt), cudaSuccess);
    ASSERT_EQ(d_extra.Allocate(m_rt), cudaSuccess);

    // Both parties share the correction words, so one repack serves both.
    fss::gpu::VdpfRepackGpu<kBucketBits, Group, Prg, Blake3, Blake3, uint>(
        d_cws.data(), m_rt, d_cw_s.data(), d_extra.data());

    for (int p = 0; p < 2; ++p) {
      fss::gpu::VdmpfBatchEvalGpu(p != 0, d_xs.data(), d_ys[p].data(), d_pi[p].data(), n, sigma, m_rt, b_rt,
          d_s0s[p].data(), d_cw_s.data(), d_extra.data(), d_cs.data(), d_ocws.data(), vdmpf);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<int4> ys[2];
    cuda::std::array<int4, 4> pi[2];
    ys[0].resize(n);
    ys[1].resize(n);
    ASSERT_EQ(d_ys[0].Download(ys[0].data(), n), cudaSuccess);
    ASSERT_EQ(d_ys[1].Download(ys[1].data(), n), cudaSuccess);
    ASSERT_EQ(d_pi[0].Download(&pi[0], 1), cudaSuccess);
    ASSERT_EQ(d_pi[1].Download(&pi[1], 1), cudaSuccess);

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

    // The honest proofs must match.
    EXPECT_TRUE(VdmpfT::Verify(cuda::std::span<const int4, 4>(pi[0]), cuda::std::span<const int4, 4>(pi[1])));

    // Bit-exact against the host Merkle-fold evaluation per party: shares and
    // proof.
    std::vector<int4> host_ys(n);
    cuda::std::array<int4, 4> host_pi;
    for (int p = 0; p < 2; ++p) {
      vdmpf.BatchEvalMerkle(p != 0, p == 0 ? k0 : k1, cuda::std::span<const uint16_t>(xs),
          cuda::std::span<int4>(host_ys), host_pi);
      for (int i = 0; i < n; ++i) ExpectOutput(ys[p][i], host_ys[i], xs[i]);
      for (int w = 0; w < 4; ++w) ExpectOutput(pi[p][w], host_pi[w], w);
    }
  }
};

using VdmpfBytesGpuTest = VdmpfGpuTest<fss::group::Bytes>;
using VdmpfUint128GpuTest = VdmpfGpuTest<Uint127Group>;

TEST_F(VdmpfBytesGpuTest, BatchEval) {
  TestBatchEval();
}

TEST_F(VdmpfUint128GpuTest, BatchEval) {
  TestBatchEval();
}

}  // namespace
