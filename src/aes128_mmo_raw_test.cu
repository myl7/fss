#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <random>
#include <cstdint>
#include <cstring>
#include <fss/prg/aes128_mmo_raw.cuh>
#include <fss/util.cuh>
#include <fss/hash/aes128_ccr.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>
#include <fss/prg/aes128_mmo.cuh>

namespace {

using fss::prg::Aes128MmoRaw;

constexpr int kTrials = 64;

int4 RandomInt4(std::mt19937 &gen) {
  std::uniform_int_distribution<int> dis;
  return {dis(gen), dis(gen), dis(gen), dis(gen)};
}

template <int mul>
void RandomKeys(std::mt19937 &gen, uint8_t (&keys)[mul][16]) {
  std::uniform_int_distribution<int> dis;
  for (auto &key : keys) {
    for (auto &b : key) b = static_cast<uint8_t>(dis(gen) & 0xff);
  }
}

// Gen8(seeds)[j] must be bit-exact with the scalar Gen(seeds[j]).
template <int mul>
void TestGen8MatchesGen(uint64_t seed) {
  std::mt19937 gen(seed);

  uint8_t keys[mul][16];
  RandomKeys(gen, keys);
  Aes128MmoRaw<mul> prg(keys);

  for (int trial = 0; trial < kTrials; ++trial) {
    cuda::std::array<int4, 8> seeds;
    for (auto &seed : seeds) seed = RandomInt4(gen);

    auto batched = prg.Gen8(seeds);
    for (int j = 0; j < 8; ++j) {
      auto scalar = prg.Gen(seeds[j]);
      for (int i = 0; i < mul; ++i) {
        EXPECT_EQ(memcmp(&batched[j][i], &scalar[i], sizeof(int4)), 0);
      }
    }
  }
}

}  // namespace

TEST(Aes128MmoRawGen8Test, MatchesGenMul1) {
  TestGen8MatchesGen<1>(0x101);
}

TEST(Aes128MmoRawGen8Test, MatchesGenMul2) {
  TestGen8MatchesGen<2>(0x202);
}

TEST(Aes128MmoRawGen8Test, MatchesGenMul4) {
  TestGen8MatchesGen<4>(0x404);
}

namespace {

// detail::GenBatch() forwards to Gen8() for PRGs that provide it and must stay
// bit-exact with per-seed Gen() calls.
template <int mul>
void TestGenBatchForwardsToGen8(uint64_t seed) {
  std::mt19937 gen(seed);

  uint8_t keys[mul][16];
  RandomKeys(gen, keys);
  Aes128MmoRaw<mul> prg(keys);

  for (int trial = 0; trial < kTrials; ++trial) {
    int4 seeds[8];
    for (auto &seed : seeds) seed = RandomInt4(gen);

    auto batched = fss::detail::GenBatch<mul>(prg, seeds);
    for (int j = 0; j < 8; ++j) {
      auto scalar = prg.Gen(seeds[j]);
      for (int i = 0; i < mul; ++i) {
        EXPECT_EQ(memcmp(&batched[j][i], &scalar[i], sizeof(int4)), 0);
      }
    }
  }
}

}  // namespace

TEST(Aes128MmoRawGenBatchTest, ForwardsToGen8Mul1) {
  TestGenBatchForwardsToGen8<1>(0x105);
}

TEST(Aes128MmoRawGenBatchTest, ForwardsToGen8Mul2) {
  TestGenBatchForwardsToGen8<2>(0x205);
}

namespace {

// A PRG that provides only Gen() and no Gen8(): detail::GenBatch() must fall
// back to looping Gen() over the seeds, so PRGs without a batched entry stay
// usable through the batched interfaces.
template <int mul>
class PlainPrg {
 public:
  __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
    cuda::std::array<int4, mul> out{};
    for (int i = 0; i < mul; ++i) {
      unsigned k = 0x9e3779b9u * static_cast<unsigned>(i);
      out[i] = {static_cast<int>(static_cast<unsigned>(seed.x) ^ k),
          static_cast<int>(static_cast<unsigned>(seed.y) + k),
          static_cast<int>(static_cast<unsigned>(seed.z) ^ (k << 1)),
          static_cast<int>(static_cast<unsigned>(seed.w) - k)};
    }
    return out;
  }
};

template <int mul>
void TestGenBatchFallsBackToGen(uint64_t seed) {
  std::mt19937 gen(seed);
  PlainPrg<mul> prg;

  for (int trial = 0; trial < kTrials; ++trial) {
    int4 seeds[8];
    for (auto &seed : seeds) seed = RandomInt4(gen);

    auto batched = fss::detail::GenBatch<mul>(prg, seeds);
    for (int j = 0; j < 8; ++j) {
      auto scalar = prg.Gen(seeds[j]);
      for (int i = 0; i < mul; ++i) {
        EXPECT_EQ(memcmp(&batched[j][i], &scalar[i], sizeof(int4)), 0);
      }
    }
  }
}

}  // namespace

TEST(DetailGenBatchTest, FallsBackToGenLoopMul1) {
  TestGenBatchFallsBackToGen<1>(0x106);
}

TEST(DetailGenBatchTest, FallsBackToGenLoopMul2) {
  TestGenBatchFallsBackToGen<2>(0x206);
}

// DPF-style both-children expansion: the frontier clamps the seed LSB before the
// PRG call and takes the children LSBs as control bits. The batched path must
// reproduce the scalar children bits and clamped bodies for mul=2 seeds.
TEST(Aes128MmoRawGen8Test, BothChildrenExpansion) {
  std::mt19937 gen(0x808);

  uint8_t keys[2][16];
  RandomKeys(gen, keys);
  Aes128MmoRaw<2> prg(keys);

  for (int trial = 0; trial < kTrials; ++trial) {
    cuda::std::array<int4, 8> seeds;
    for (auto &seed : seeds) seed = fss::util::SetLsb(RandomInt4(gen), false);

    auto children = prg.Gen8(seeds);
    for (int j = 0; j < 8; ++j) {
      auto [sl_ref, sr_ref] = prg.Gen(seeds[j]);
      auto [sl, sr] = children[j];

      EXPECT_EQ(fss::util::GetLsb(sl), fss::util::GetLsb(sl_ref));
      EXPECT_EQ(fss::util::GetLsb(sr), fss::util::GetLsb(sr_ref));
      int4 sl_clamped = fss::util::SetLsb(sl, false);
      int4 sr_clamped = fss::util::SetLsb(sr, false);
      int4 sl_ref_clamped = fss::util::SetLsb(sl_ref, false);
      int4 sr_ref_clamped = fss::util::SetLsb(sr_ref, false);
      EXPECT_EQ(memcmp(&sl_clamped, &sl_ref_clamped, sizeof(int4)), 0);
      EXPECT_EQ(memcmp(&sr_clamped, &sr_ref_clamped, sizeof(int4)), 0);
    }
  }
}

namespace {
namespace aes_ccr {
using RawHash = fss::hash::Aes128Ccr<fss::prg::Aes128MmoRaw<1>>;
using SoftHash = fss::hash::Aes128Ccr<fss::prg::Aes128Soft<1>>;
static_assert(CcrHashable<RawHash> && CcrHashable<SoftHash>);
void ExpectBlock(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

TEST(Aes128Ccr, KnownAnswer) {
  const uint8_t keys[1][16] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
  uint32_t te0[256];
  uint8_t sbox[256];
  fss::prg::aes_detail::InitTe0(te0);
  fss::prg::aes_detail::InitSbox(sbox);
  RawHash raw{fss::prg::Aes128MmoRaw<1>{keys}};
  SoftHash soft{fss::prg::Aes128Soft<1>{keys, te0, sbox}};
  int4 block = {static_cast<int>(0xbbaa9988u), static_cast<int>(0xffeeddccu),
      static_cast<int>(0x88888888u), static_cast<int>(0x88888888u)};
  // sigma(block) is 00112233445566778899aabbccddeeff, the AES-128 known-answer plaintext.
  // AES output is 69c4e0d86a7b0430d8cdb78070b4c55a. MMO xors the plaintext once.
  int4 expected = {static_cast<int>(0xebc2d569u), static_cast<int>(0x47622e2eu),
      static_cast<int>(0x3b1d5450u), static_cast<int>(0xa52b69bcu)};
  ExpectBlock(raw.Hash(block), expected);
  ExpectBlock(soft.Hash(block), expected);
}
}  // namespace aes_ccr
}  // namespace

namespace {

struct CountingPrg {
  int calls = 0;
  __host__ __device__ cuda::std::array<int4, 4> Gen(int4 seed) {
    ++calls;
    return {seed, int4{1, 2, 3, 4}, int4{5, 6, 7, 8}, int4{9, 10, 11, 12}};
  }
};

struct HalfCountingPrg : CountingPrg {
  int half_calls = 0;
  cuda::std::array<int4, 2> GenHalf(int4 seed, bool right) {
    ++half_calls;
    return right ? cuda::std::array<int4, 2>{int4{5, 6, 7, 8}, int4{9, 10, 11, 12}}
                 : cuda::std::array<int4, 2>{seed, int4{1, 2, 3, 4}};
  }
};

struct WrongHalfPrg : CountingPrg {
  cuda::std::array<int4, 1> GenHalf(int4 seed, bool right) { return {seed}; }
};

struct ReferenceHalfPrg : CountingPrg {
  cuda::std::array<int4, 2> value{};
  cuda::std::array<int4, 2> &GenHalf(int4 seed, bool right) { return value; }
};

struct WrongFullPrg {
  cuda::std::array<int4, 2> Gen(int4 seed) { return {seed, seed}; }
  cuda::std::array<int4, 2> GenHalf(int4 seed, bool right) { return {seed, seed}; }
};

template <template <int> class Prg>
constexpr bool HalfCapabilities() {
  return Prgable<Prg<1>, 1> && Prgable<Prg<3>, 3> && !HasGenHalf<Prg<1>, 1> &&
      !HasGenHalf<Prg<3>, 3> && HasGenHalf<Prg<2>, 2> && HasGenHalf<Prg<4>, 4> &&
      HasGenHalf<Prg<6>, 6>;
}

static_assert(HalfCapabilities<fss::prg::Aes128Mmo>());
static_assert(HalfCapabilities<fss::prg::Aes128MmoRaw>());
static_assert(HalfCapabilities<fss::prg::Aes128Soft>());
static_assert(!HasGenHalf<CountingPrg, 4> && HasGenHalf<HalfCountingPrg, 4>);
static_assert(!HasGenHalf<WrongHalfPrg, 4> && !HasGenHalf<ReferenceHalfPrg, 4>);
static_assert(!HasGenHalf<WrongFullPrg, 4> && !HasGenHalf<HalfCountingPrg, 0>);
static_assert(!HasGenHalf<HalfCountingPrg, -2> && !HasGenHalf<HalfCountingPrg, 3>);

void ExpectHalfBlock(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

template <int mul>
void TestActualHalfHost() {
  std::mt19937 random(0x123400 + mul);
  uint32_t te0[256];
  uint8_t sbox[256];
  fss::prg::aes_detail::InitTe0(te0);
  fss::prg::aes_detail::InitSbox(sbox);
  for (int config = 0; config < 3; ++config) {
    uint8_t keys[mul][16]{};
    if (config != 0) RandomKeys(random, keys);
    const unsigned char *key_ptrs[mul];
    for (int i = 0; i < mul; ++i) key_ptrs[i] = keys[i];
    auto ctxs = fss::prg::Aes128Mmo<mul>::CreateCtxs(key_ptrs);
    fss::prg::Aes128Mmo<mul> openssl(ctxs);
    fss::prg::Aes128MmoRaw<mul> raw(keys);
    fss::prg::Aes128Soft<mul> soft(keys, te0, sbox);
    for (int trial = 0; trial <= kTrials; ++trial) {
      int4 seed{};
      if (trial != 0) seed = {static_cast<int>(random()), static_cast<int>(random()),
          static_cast<int>(random()), static_cast<int>(random())};
      auto full = raw.Gen(seed);
      auto openssl_full = openssl.Gen(seed);
      auto soft_full = soft.Gen(seed);
      for (bool right : {false, true}) {
        auto raw_half = raw.GenHalf(seed, right);
        auto openssl_half = openssl.GenHalf(seed, right);
        auto soft_half = soft.GenHalf(seed, right);
        auto dispatched = fss::detail::GenHalf<mul>(raw, seed, right);
        for (int i = 0; i < mul / 2; ++i) {
          int offset = i + (right ? mul / 2 : 0);
          ExpectHalfBlock(openssl_full[offset], full[offset]);
          ExpectHalfBlock(soft_full[offset], full[offset]);
          ExpectHalfBlock(raw_half[i], full[offset]);
          ExpectHalfBlock(openssl_half[i], full[offset]);
          ExpectHalfBlock(soft_half[i], full[offset]);
          ExpectHalfBlock(dispatched[i], full[offset]);
        }
      }
    }
    fss::prg::Aes128Mmo<mul>::FreeCtxs(ctxs);
  }
}

template <int mul>
__global__ void ActualSoftHalfKernel(const uint8_t (*keys)[16], const int4 *seeds, int4 *out, int count) {
  __shared__ uint32_t te0[256];
  __shared__ uint8_t sbox[256];
  if (threadIdx.x == 0) {
    fss::prg::aes_detail::InitTe0(te0);
    fss::prg::aes_detail::InitSbox(sbox);
  }
  __syncthreads();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  fss::prg::Aes128Soft<mul> prg(keys, te0, sbox);
  auto full = prg.Gen(seeds[tid]);
  auto left = fss::detail::GenHalf<mul>(prg, seeds[tid], false);
  auto right = fss::detail::GenHalf<mul>(prg, seeds[tid], true);
  for (int i = 0; i < mul; ++i) out[tid * mul * 2 + i] = full[i];
  for (int i = 0; i < mul / 2; ++i) {
    out[tid * mul * 2 + mul + i] = left[i];
    out[tid * mul * 2 + mul + mul / 2 + i] = right[i];
  }
}

template <int mul>
void TestActualHalfDevice() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) GTEST_SKIP();
  std::mt19937 random(0x567800 + mul);
  constexpr int kCount = kTrials + 1;
  uint8_t keys[mul][16];
  RandomKeys(random, keys);
  int4 seeds[kCount]{};
  for (int i = 1; i < kCount; ++i) seeds[i] = {static_cast<int>(random()), static_cast<int>(random()),
      static_cast<int>(random()), static_cast<int>(random())};
  uint8_t (*device_keys)[16] = nullptr;
  int4 *device_seeds = nullptr;
  int4 *device_out = nullptr;
  ASSERT_EQ(cudaMalloc(&device_keys, sizeof(keys)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_seeds, sizeof(seeds)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_out, kCount * mul * 2 * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_keys, keys, sizeof(keys), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_seeds, seeds, sizeof(seeds), cudaMemcpyHostToDevice), cudaSuccess);
  ActualSoftHalfKernel<mul><<<2, 64>>>(device_keys, device_seeds, device_out, kCount);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int4 out[kCount * mul * 2];
  ASSERT_EQ(cudaMemcpy(out, device_out, sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);
  fss::prg::Aes128MmoRaw<mul> raw(keys);
  for (int trial = 0; trial < kCount; ++trial) {
    auto expected = raw.Gen(seeds[trial]);
    for (int i = 0; i < mul; ++i) {
      ExpectHalfBlock(out[trial * mul * 2 + i], expected[i]);
      ExpectHalfBlock(out[trial * mul * 2 + mul + i], expected[i]);
    }
  }
  EXPECT_EQ(cudaFree(device_keys), cudaSuccess);
  EXPECT_EQ(cudaFree(device_seeds), cudaSuccess);
  EXPECT_EQ(cudaFree(device_out), cudaSuccess);
}

TEST(GenHalfTest, ActualHostMul2) { TestActualHalfHost<2>(); }
TEST(GenHalfTest, ActualHostMul4) { TestActualHalfHost<4>(); }
TEST(GenHalfTest, ActualHostMul6) { TestActualHalfHost<6>(); }
TEST(GenHalfTest, ActualDeviceMul2) { TestActualHalfDevice<2>(); }
TEST(GenHalfTest, ActualDeviceMul4) { TestActualHalfDevice<4>(); }
TEST(GenHalfTest, ActualDeviceMul6) { TestActualHalfDevice<6>(); }

TEST(GenHalfTest, FallbackCallsGenOncePerHalf) {
  CountingPrg prg;
  int4 seed{13, 14, 15, 16};
  auto left = fss::detail::GenHalf<4>(prg, seed, false);
  EXPECT_EQ(prg.calls, 1);
  auto right = fss::detail::GenHalf<4>(prg, seed, true);
  EXPECT_EQ(prg.calls, 2);
  ExpectHalfBlock(left[0], seed);
  ExpectHalfBlock(left[1], {1, 2, 3, 4});
  ExpectHalfBlock(right[0], {5, 6, 7, 8});
  ExpectHalfBlock(right[1], {9, 10, 11, 12});
}

TEST(GenHalfTest, OptionalMemberAvoidsGen) {
  HalfCountingPrg prg;
  int4 seed{13, 14, 15, 16};
  for (bool right : {false, true}) {
    auto out = fss::detail::GenHalf<4>(prg, seed, right);
    ExpectHalfBlock(out[0], right ? int4{5, 6, 7, 8} : seed);
    ExpectHalfBlock(out[1], right ? int4{9, 10, 11, 12} : int4{1, 2, 3, 4});
  }
  EXPECT_EQ(prg.calls, 0);
  EXPECT_EQ(prg.half_calls, 2);
}

}  // namespace
