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
#include <fss/repack_gpu.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/chacha_feistel.cuh>
#include <fss/vdmpf.cuh>
#include "core_api_test_helpers.cuh"

namespace {

// A named wrapper keeps the 128-bit modulus out of nvcc kernel stubs.
struct Uint127Group {
  using Impl = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
  Impl value;
  __host__ __device__ Uint127Group operator+(Uint127Group rhs) const {
    return {value + rhs.value};
  }
  __host__ __device__ Uint127Group operator-() const {
    return {-value};
  }
  __host__ __device__ static Uint127Group From(int4 input) {
    return {Impl::From(input)};
  }
  __host__ __device__ int4 Into() const {
    return value.Into();
  }
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
constexpr int4 kHashIv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};

void ExpectOutput(int4 actual, int4 expected, size_t input) {
  EXPECT_EQ(actual.x, expected.x) << "input " << input;
  EXPECT_EQ(actual.y, expected.y) << "input " << input;
  EXPECT_EQ(actual.z, expected.z) << "input " << input;
  EXPECT_EQ(actual.w, expected.w) << "input " << input;
}

template <typename Group, int ch_lambda = 80>
class VdmpfGpuTest : public ::testing::Test {
protected:
  static constexpr int kInBits = 16;
  static constexpr int kMaxPoints = 30;
  static constexpr int kBucketBits = 14;
  static constexpr int kT = 30;

  using Prg = fss::prg::ChaCha<2>;
  using Prp = fss::prp::ChaChaFeistel;
  using Blake3 = fss::hash::Blake3;
  using VdmpfT = fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Blake3, Blake3, Prp, uint16_t, ch_lambda>;
  using Cw = typename VdmpfT::Cw;

  static constexpr int m = VdmpfT::kBucketCount;

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
    cuda::std::array<int4, size_t{m} * 2> s0s;
    for (int i = 0; i < m; ++i) {
      s0s[i] = RandomSeed(gen);
      s0s[size_t{m} + i] = RandomSeed(gen);
    }
    ASSERT_EQ(fss_test::GenMulti(vdmpf, k0, k1, sigma, cuda::std::span<const int4>(s0s),
                  cuda::std::span<const uint16_t>(alphas, kT), cuda::std::span<const int4>(betas, kT), kT, 1000),
        0);
  }

  void TestBatchEval(bool check_execution_parameters = false) {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
    typename VdmpfT::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    VdmpfT vdmpf{prg, xor_hash, hash, prp_};
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
    std::vector<cuda::std::array<int4, 4>> h_cs(m);
    h_cws.resize(static_cast<size_t>(m_rt) * (kBucketBits + 1));
    for (int i = 0; i < m_rt; ++i) {
      for (int j = 0; j <= kBucketBits; ++j) {
        h_cws[static_cast<size_t>(i) * (kBucketBits + 1) + j] = k0.bks[i].cws[j];
      }
    }
    for (int i = 0; i < 2; ++i) {
      h_s0s[i].resize(m_rt);
      for (int j = 0; j < m_rt; ++j) {
        h_s0s[i][j] = (i ? k1 : k0).bks[j].s0;
      }
    }
    // cs is shared between the parties; the fold covers all m buckets.
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < 4; ++j) h_cs[i][j] = k0.bks[i].cs[j];

    DeviceBuffer<uint16_t> d_xs;
    DeviceBuffer<int4> d_ys[2];
    DeviceBuffer<cuda::std::array<int4, 4>> d_pi[2];
    DeviceBuffer<int4> d_s0s[2];
    DeviceBuffer<Cw> d_cws;
    DeviceBuffer<int4> d_cw_s, d_ocws;
    DeviceBuffer<cuda::std::array<int4, 4>> d_cs;
    DeviceBuffer<fss::gpu::detail::ExtBits<kBucketBits>> d_extra;
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
    ASSERT_EQ(d_cs.Allocate(m), cudaSuccess);
    ASSERT_EQ(d_cs.Upload(h_cs.data(), m), cudaSuccess);
    ASSERT_EQ(d_cw_s.Allocate(static_cast<size_t>(kBucketBits) * m_rt), cudaSuccess);
    ASSERT_EQ(d_extra.Allocate(m_rt), cudaSuccess);

    // Both parties share the correction words, so one repack serves both.
    fss::gpu::RepackGpu<fss::Vdpf<kBucketBits, Group, Prg, Blake3, Blake3, __uint128_t>>({d_cws.data(), h_cws.size()},
        m_rt,
        {d_cw_s.data(), static_cast<size_t>(kBucketBits) * m_rt},
        {d_extra.data(), static_cast<size_t>(m_rt)},
        {d_ocws.data(), static_cast<size_t>(m_rt)},
        {d_cs.data(), static_cast<size_t>(m)},
        {d_cs.data(), static_cast<size_t>(m)});

    const cuda::std::span<const int4> cws(d_cw_s.data(), static_cast<size_t>(kBucketBits) * m_rt);
    const cuda::std::span<const fss::gpu::detail::ExtBits<kBucketBits>> controls(d_extra.data(), m_rt);
    const cuda::std::span<const int4> terminals(d_ocws.data(), m_rt);
    const cuda::std::span<const cuda::std::array<int4, 4>> checks(d_cs.data(), m);
    // Empty batches use every bucket's cs and require no query or key buffers.
    for (int p = 0; p < 2; ++p) {
      fss::gpu::detail::VdmpfBatchEvalMerkleGpu(p != 0, static_cast<const uint16_t *>(nullptr),
          nullptr, d_pi[p].data(), 0, {}, 0, 0, nullptr, nullptr, nullptr, checks.data(), nullptr, vdmpf);
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      cuda::std::array<int4, 4> expected, actual;
      fss_test::BatchEvalMerkleKey(vdmpf,
          p != 0, p == 0 ? k0 : k1, cuda::std::span<const uint16_t>(), cuda::std::span<int4>(), expected);
      ASSERT_EQ(d_pi[p].Download(&actual, 1), cudaSuccess);
      for (int w = 0; w < 4; ++w) ExpectOutput(actual[w], expected[w], w);
    }

    for (int p = 0; p < 2; ++p) {
      fss::gpu::detail::VdmpfBatchEvalMerkleGpu(p != 0, d_xs.data(), d_ys[p].data(), d_pi[p].data(), n,
          sigma, m_rt, b_rt, d_s0s[p].data(), cws.data(), controls.data(), checks.data(),
          terminals.data(), vdmpf);
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
      fss_test::BatchEvalMerkleKey(vdmpf,
          p != 0, p == 0 ? k0 : k1, cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(host_ys), host_pi);
      for (int i = 0; i < n; ++i) ExpectOutput(ys[p][i], host_ys[i], xs[i]);
      for (int w = 0; w < 4; ++w) ExpectOutput(pi[p][w], host_pi[w], w);
    }

    if (check_execution_parameters) {
      constexpr int kCapacity = 513;
      DeviceBuffer<uint16_t> tuning_xs;
      DeviceBuffer<int4> tuning_ys;
      DeviceBuffer<cuda::std::array<int4, 4>> tuning_pi;
      ASSERT_EQ(tuning_xs.Allocate(kCapacity), cudaSuccess);
      ASSERT_EQ(tuning_ys.Allocate(kCapacity), cudaSuccess);
      ASSERT_EQ(tuning_pi.Allocate(1), cudaSuccess);
      // Repeated queries concentrate records into a few buckets. The tails
      // cross several possible GPU chunk boundaries without changing the tree.
      for (int count : {0, 1, 257, 513}) {
        SCOPED_TRACE(::testing::Message() << "queries=" << count);
        std::vector<uint16_t> repeated(count, alphas[0]);
        if (count != 0) ASSERT_EQ(tuning_xs.Upload(repeated.data(), count), cudaSuccess);
        std::vector<int4> expected_ys[2];
        cuda::std::array<int4, 4> expected_pi[2];
        for (int party = 0; party < 2; ++party) {
          expected_ys[party].resize(count);
          fss_test::BatchEvalMerkleKey(vdmpf, party != 0, party == 0 ? k0 : k1,
              cuda::std::span<const uint16_t>(repeated), cuda::std::span<int4>(expected_ys[party]),
              expected_pi[party]);
        }
        auto check_configuration = [&]<int chunk_size, int threads>() {
          SCOPED_TRACE(::testing::Message() << "chunk=" << chunk_size << ", threads=" << threads);
          for (int party = 0; party < 2; ++party) {
            SCOPED_TRACE(::testing::Message() << "party=" << party);
            fss::gpu::detail::VdmpfBatchEvalMerkleGpu<4, chunk_size, threads>(party != 0,
                tuning_xs.data(), tuning_ys.data(), tuning_pi.data(), count,
                count == 0 ? int4{} : sigma, count == 0 ? 0 : m_rt,
                count == 0 ? __uint128_t{} : b_rt,
                count == 0 ? nullptr : d_s0s[party].data(),
                count == 0 ? nullptr : cws.data(),
                count == 0 ? nullptr : controls.data(), checks.data(),
                count == 0 ? nullptr : terminals.data(), vdmpf);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cuda::std::array<int4, 4> actual_pi;
            ASSERT_EQ(tuning_pi.Download(&actual_pi, 1), cudaSuccess);
            for (int word = 0; word < 4; ++word) ExpectOutput(actual_pi[word], expected_pi[party][word], word);
            if (count != 0) {
              std::vector<int4> actual_ys(count);
              ASSERT_EQ(tuning_ys.Download(actual_ys.data(), count), cudaSuccess);
              for (int i = 0; i < count; ++i) ExpectOutput(actual_ys[i], expected_ys[party][i], i);
            }
          }
        };
        check_configuration.template operator()<1, 64>();
        check_configuration.template operator()<32, 64>();
        check_configuration.template operator()<128, 128>();
        check_configuration.template operator()<256, 256>();
        check_configuration.template operator()<512, 128>();
      }
    }

  }

  void TestPublicBatchEvalMerkleGpu() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ASSERT_EQ(status, cudaSuccess);
    typename VdmpfT::Key k0, k1;
    int4 sigma;
    GenKeys(k0, k1, sigma);
    VdmpfT vdmpf{prg, xor_hash, hash, prp_};

    // Sparse ordered queries: the alphas, random non-alphas, an unsorted tail
    // of domain extremes, and duplicates.
    std::vector<uint16_t> xs(alphas, alphas + kT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);
    constexpr int kRandoms = 100;
    while (static_cast<int>(xs.size()) < kT + kRandoms) {
      uint16_t x = dis(gen);
      bool is_alpha = std::any_of(alphas, alphas + kT, [x](uint16_t a) { return x == a; });
      if (!is_alpha) xs.push_back(x);
    }
    xs.push_back(0);
    xs.push_back(0xFFFF);
    xs.push_back(alphas[0]);
    xs.push_back(alphas[kT - 1]);
    const int n = static_cast<int>(xs.size());

    cuda::std::array<int4, m> h_s0s[2];
    std::vector<Cw> h_cws(size_t{m} * (kBucketBits + 1));
    std::vector<int4> h_cs(size_t{m} * 4);
    for (int i = 0; i < m; ++i) {
      h_s0s[0][i] = k0.bks[i].s0;
      h_s0s[1][i] = k1.bks[i].s0;
      for (int j = 0; j <= kBucketBits; ++j)
        h_cws[static_cast<size_t>(i) * (kBucketBits + 1) + j] = k0.bks[i].cws[j];
      for (int j = 0; j < 4; ++j) h_cs[static_cast<size_t>(i) * 4 + j] = k0.bks[i].cs[j];
    }

    // Host reference from the raw key components.
    std::vector<int4> host_ys[2];
    cuda::std::array<int4, 4> host_pi[2];
    for (int p = 0; p < 2; ++p) {
      host_ys[p].resize(n);
      vdmpf.BatchEvalMerkle(p != 0, sigma, k0.b_size_rt, h_s0s[p],
          cuda::std::span<const Cw>(h_cws), cuda::std::span<const int4>(h_cs),
          cuda::std::span<const uint16_t>(xs), cuda::std::span<int4>(host_ys[p]), host_pi[p]);
    }
    EXPECT_TRUE(VdmpfT::Verify(cuda::std::span<const int4, 4>(host_pi[0]),
        cuda::std::span<const int4, 4>(host_pi[1])));

    DeviceBuffer<uint16_t> d_xs;
    DeviceBuffer<int4> d_ys[2], d_s0s[2], d_pi[2];
    DeviceBuffer<Cw> d_cws;
    DeviceBuffer<int4> d_cs;
    ASSERT_EQ(d_xs.Allocate(n), cudaSuccess);
    ASSERT_EQ(d_xs.Upload(xs.data(), n), cudaSuccess);
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(d_ys[p].Allocate(n), cudaSuccess);
      ASSERT_EQ(d_pi[p].Allocate(4), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Allocate(m), cudaSuccess);
      ASSERT_EQ(d_s0s[p].Upload(h_s0s[p].data(), m), cudaSuccess);
    }
    ASSERT_EQ(d_cws.Allocate(h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cws.Upload(h_cws.data(), h_cws.size()), cudaSuccess);
    ASSERT_EQ(d_cs.Allocate(h_cs.size()), cudaSuccess);
    ASSERT_EQ(d_cs.Upload(h_cs.data(), h_cs.size()), cudaSuccess);

    for (int p = 0; p < 2; ++p) {
      SCOPED_TRACE(::testing::Message() << "party " << p);
      const cudaError_t launch = vdmpf.BatchEvalMerkleGpu(p != 0, sigma, k0.b_size_rt,
          cuda::std::span<const int4, m>(d_s0s[p].data(), m),
          cuda::std::span<const Cw>(d_cws.data(), h_cws.size()),
          cuda::std::span<const int4>(d_cs.data(), h_cs.size()),
          cuda::std::span<const uint16_t>(d_xs.data(), n),
          cuda::std::span<int4>(d_ys[p].data(), n),
          cuda::std::span<int4, 4>(d_pi[p].data(), 4));
      ASSERT_EQ(launch, cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<int4> ys[2];
    std::vector<int4> pi[2];
    ys[0].resize(n);
    ys[1].resize(n);
    pi[0].resize(4);
    pi[1].resize(4);
    ASSERT_EQ(d_ys[0].Download(ys[0].data(), n), cudaSuccess);
    ASSERT_EQ(d_ys[1].Download(ys[1].data(), n), cudaSuccess);
    ASSERT_EQ(d_pi[0].Download(pi[0].data(), 4), cudaSuccess);
    ASSERT_EQ(d_pi[1].Download(pi[1].data(), 4), cudaSuccess);

    for (int p = 0; p < 2; ++p) {
      for (int i = 0; i < n; ++i) ExpectOutput(ys[p][i], host_ys[p][i], xs[i]);
      for (int w = 0; w < 4; ++w) ExpectOutput(pi[p][w], host_pi[p][w], w);
    }
    EXPECT_TRUE(VdmpfT::Verify(cuda::std::span<const int4, 4>(pi[0].data(), 4),
        cuda::std::span<const int4, 4>(pi[1].data(), 4)));

    // Non-default block size over a short odd prefix, still bit-exact.
    constexpr int kShort = 37;
    std::vector<uint16_t> short_xs(xs.begin(), xs.begin() + kShort);
    std::vector<int4> short_host(kShort);
    cuda::std::array<int4, 4> short_pi;
    vdmpf.BatchEvalMerkle(false, sigma, k0.b_size_rt, h_s0s[0],
        cuda::std::span<const Cw>(h_cws), cuda::std::span<const int4>(h_cs),
        cuda::std::span<const uint16_t>(short_xs), cuda::std::span<int4>(short_host), short_pi);
    DeviceBuffer<uint16_t> d_short_xs;
    DeviceBuffer<int4> d_short_ys, d_short_pi;
    ASSERT_EQ(d_short_xs.Allocate(kShort), cudaSuccess);
    ASSERT_EQ(d_short_xs.Upload(short_xs.data(), kShort), cudaSuccess);
    ASSERT_EQ(d_short_ys.Allocate(kShort), cudaSuccess);
    ASSERT_EQ(d_short_pi.Allocate(4), cudaSuccess);
    const cudaError_t launch = vdmpf.template BatchEvalMerkleGpu<64>(false, sigma, k0.b_size_rt,
        cuda::std::span<const int4, m>(d_s0s[0].data(), m),
        cuda::std::span<const Cw>(d_cws.data(), h_cws.size()),
        cuda::std::span<const int4>(d_cs.data(), h_cs.size()),
        cuda::std::span<const uint16_t>(d_short_xs.data(), kShort),
        cuda::std::span<int4>(d_short_ys.data(), kShort),
        cuda::std::span<int4, 4>(d_short_pi.data(), 4));
    ASSERT_EQ(launch, cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int4> short_actual(kShort);
    std::vector<int4> short_actual_pi(4);
    ASSERT_EQ(d_short_ys.Download(short_actual.data(), kShort), cudaSuccess);
    ASSERT_EQ(d_short_pi.Download(short_actual_pi.data(), 4), cudaSuccess);
    for (int i = 0; i < kShort; ++i) ExpectOutput(short_actual[i], short_host[i], short_xs[i]);
    for (int w = 0; w < 4; ++w) ExpectOutput(short_actual_pi[w], short_pi[w], w);
  }
};

using VdmpfBytesGpuTest = VdmpfGpuTest<fss::group::Bytes>;
using VdmpfUint128GpuTest = VdmpfGpuTest<Uint127Group>;

TEST_F(VdmpfBytesGpuTest, BatchEval) {
  TestBatchEval(true);
}

TEST_F(VdmpfUint128GpuTest, BatchEval) {
  TestBatchEval();
}

TEST_F(VdmpfBytesGpuTest, PublicBatchEvalMerkle) {
  TestPublicBatchEvalMerkleGpu();
}

TEST_F(VdmpfUint128GpuTest, PublicBatchEvalMerkle) {
  TestPublicBatchEvalMerkleGpu();
}

using VdmpfCustomSecurityGpuTest = VdmpfGpuTest<fss::group::Bytes, 60>;
TEST_F(VdmpfCustomSecurityGpuTest, BatchEval) {
  TestBatchEval();
}

}  // namespace

namespace {
namespace query_bounds {
using Group = fss::group::Bytes;
using Prg = fss::prg::ChaCha<2>;
using Prp = fss::prp::ChaChaFeistel;
using ProofHash = fss::hash::Blake3;
ProofHash MakeProofHash() {
  const int4 iv[2] = {{2, 3, 5, 7}, {11, 13, 17, 19}};
  return ProofHash{iv};
}
TEST(VdmpfGpuArguments, RejectsOversizedQueryCount) {
  using Vdmpf = fss::Vdmpf<8, 30, 4, Group, Prg, ProofHash, ProofHash, Prp, uint8_t>;
  Vdmpf scheme{Prg{23, 47}, MakeProofHash(), MakeProofHash(), Prp{}};

  auto evaluate = [&](int count) {
    fss::gpu::detail::VdmpfBatchEvalMerkleGpu(false, static_cast<const uint8_t *>(nullptr),
        nullptr, nullptr, count, {}, Vdmpf::kBucketCount, Vdmpf::kBucketSize,
        nullptr, nullptr, nullptr, nullptr, nullptr, scheme);
  };
  EXPECT_THROW(evaluate(std::numeric_limits<int>::max() / 3 + 1), std::invalid_argument);
}

// Validation-only paths of the public entry; they run without a device.
TEST(VdmpfGpuArguments, PublicBatchRejectsInvalidInputs) {
  using Vdmpf = fss::Vdmpf<8, 30, 4, Group, Prg, ProofHash, ProofHash, Prp, uint8_t>;
  Vdmpf scheme{Prg{23, 47}, MakeProofHash(), MakeProofHash(), Prp{}};
  constexpr int kBuckets = Vdmpf::kBucketCount;
  cuda::std::array<int4, kBuckets> s0s{};
  std::vector<Vdmpf::Cw> cws(size_t{kBuckets} * 5);
  std::vector<int4> cs(size_t{kBuckets} * 4);
  uint8_t xs[1] = {0};
  int4 ys[1], pi[4];
  // Empty query sequence.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, Vdmpf::kBucketSize, s0s,
      {cws.data(), cws.size()}, {cs.data(), cs.size()}, cuda::std::span<const uint8_t>(),
      {ys, 1}, pi), cudaErrorInvalidValue);
  // Output storage shorter than the query sequence.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, Vdmpf::kBucketSize, s0s,
      {cws.data(), cws.size()}, {cs.data(), cs.size()}, {xs, 1},
      cuda::std::span<int4>(), pi), cudaErrorInvalidValue);
  // Too few correction words.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, Vdmpf::kBucketSize, s0s,
      cuda::std::span<const Vdmpf::Cw>(cws.data(), cws.size() - 1), {cs.data(), cs.size()},
      {xs, 1}, {ys, 1}, pi), cudaErrorInvalidValue);
  // Too few bucket seeds.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, Vdmpf::kBucketSize,
      cuda::std::span<const int4>(s0s.data(), size_t{kBuckets} - 1),
      {cws.data(), cws.size()}, {cs.data(), cs.size()},
      {xs, 1}, {ys, 1}, pi), cudaErrorInvalidValue);
  // Too few correction-seed blocks.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, Vdmpf::kBucketSize, s0s,
      {cws.data(), cws.size()},
      cuda::std::span<const int4>(cs.data(), size_t{kBuckets} * 4 - 1),
      {xs, 1}, {ys, 1}, pi), cudaErrorInvalidValue);
  // Null query buffer.
  EXPECT_EQ(scheme.BatchEvalMerkleGpu(false, int4{}, Vdmpf::kBucketSize, s0s,
      {cws.data(), cws.size()}, {cs.data(), cs.size()},
      cuda::std::span<const uint8_t>(static_cast<const uint8_t *>(nullptr), size_t{1}),
      {ys, 1}, pi), cudaErrorInvalidValue);
}
}  // namespace query_bounds
}  // namespace

// A named namespace: a third anonymous section makes nvcc's merged host-stub
// references to the earlier anonymous sections ambiguous.
namespace bucket_bound {

// The bucket-size boundary of the GPU validation entry, at both ends.
// B = ceil(3*2^in_bits/kBucketCount) is the smallest bucket size that keeps
// every located bucket below the allocated bucket count, so one below B must
// be rejected and B itself must pass validation. Symmetrically, 2^bucket_bits
// is the largest representable bucket size: it must pass and one above it must
// be rejected, which is the only upper defense of this C++ GPU entry. The
// entry validates before any CUDA call, so the reject half runs without a CUDA
// device.
constexpr int kInBits = 12;
constexpr int kMaxPoints = 30;
constexpr int kBucketBits = 10;
constexpr int kBuckets = 53;  // ChBucket(30, 80)
constexpr __uint128_t kBound = 232;  // ceil(3*2^12/53)
constexpr __uint128_t kUpper = __uint128_t{1} << kBucketBits;  // 2^bucket_bits
// The kernels of EvalAllMerkleGpu take the backends as template parameters,
// and nvcc host stubs cannot reference anonymous-namespace types, so the
// backends are spelled with their named-namespace types instead of aliases.
using Vdmpf = fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, fss::group::Bytes, fss::prg::ChaCha<2>,
    fss::hash::Blake3, fss::hash::Blake3, fss::prp::ChaChaFeistel, uint16_t>;

static_assert(Vdmpf::kBucketCount == kBuckets);
static_assert(kBound == (3 * (__uint128_t{1} << kInBits) + kBuckets - 1) / kBuckets);
static_assert(kBound < kUpper);
// The local spelling of the candidate count must match the scheme's private
// one, or the derived bound stops being the acceptance boundary.
static_assert(
    fss::cuckoo_hash::detail::ValidBucketSize(kBound, Vdmpf::kBucketSize, kBucketBits) &&
    !fss::cuckoo_hash::detail::ValidBucketSize(kBound - 1, Vdmpf::kBucketSize, kBucketBits));

TEST(VdmpfGpuBucketBound, EvalAllMerkleGpuBoundary) {
  const int4 iv[2] = {{2, 3, 5, 7}, {11, 13, 17, 19}};
  Vdmpf scheme{fss::prg::ChaCha<2>{23, 47}, fss::hash::Blake3{iv}, fss::hash::Blake3{iv},
      fss::prp::ChaChaFeistel{}};
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<Vdmpf::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<int4> cs(size_t{kBuckets} * 4);
  std::vector<int4> ys(size_t{1} << kInBits);
  std::vector<int4> pi(4);
  auto run = [&](__uint128_t b_size_rt) {
    return scheme.EvalAllMerkleGpu(false, int4{0, 0, 0, 0}, b_size_rt,
        cuda::std::span<const int4>(s0s.data(), kBuckets),
        cuda::std::span<const Vdmpf::Cw>(cws.data(), cws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<int4>(ys.data(), ys.size()),
        cuda::std::span<int4, 4>(pi.data(), 4));
  };
  EXPECT_EQ(run(kBound - 1), cudaErrorInvalidValue) << "b_size_rt = ceil(3*2^in_bits/kBucketCount) - 1";
  EXPECT_EQ(run(kUpper + 1), cudaErrorInvalidValue) << "b_size_rt = 2^bucket_bits + 1";
  // The accepted keys pass validation and continue into CUDA. Device buffers
  // keep that continuation meaningful on a real device; without a device the
  // call stops at the CUB size query with a device error. Either way the
  // bucket size itself must not be rejected.
  int devices = 0;
  const bool have_device = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
  if (have_device) {
    DeviceBuffer<int4> device_s0s;
    DeviceBuffer<Vdmpf::Cw> device_cws;
    DeviceBuffer<int4> device_cs;
    DeviceBuffer<int4> device_ys;
    DeviceBuffer<int4> device_pi;
    ASSERT_EQ(device_s0s.Allocate(kBuckets), cudaSuccess);
    ASSERT_EQ(device_cws.Allocate(cws.size()), cudaSuccess);
    ASSERT_EQ(device_cs.Allocate(cs.size()), cudaSuccess);
    ASSERT_EQ(device_ys.Allocate(ys.size()), cudaSuccess);
    ASSERT_EQ(device_pi.Allocate(4), cudaSuccess);
    auto run_device = [&](__uint128_t b_size_rt) {
      return scheme.EvalAllMerkleGpu(false, int4{0, 0, 0, 0}, b_size_rt,
          cuda::std::span<const int4>(device_s0s.data(), kBuckets),
          cuda::std::span<const Vdmpf::Cw>(device_cws.data(), cws.size()),
          cuda::std::span<const int4>(device_cs.data(), cs.size()),
          cuda::std::span<int4>(device_ys.data(), ys.size()),
          cuda::std::span<int4, 4>(device_pi.data(), 4));
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
