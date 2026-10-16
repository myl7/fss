#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <random>
#include <cstdint>
#include <cstring>
#include <fss/prg/aes128_mmo_raw.cuh>
#include <fss/util.cuh>

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
    cuda::std::array<int4, 8> seeds;
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
    cuda::std::array<int4, 8> seeds;
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
