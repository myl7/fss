// SPDX-License-Identifier: Apache-2.0
/**
 * @file par_depth_test.cu
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <fss/dcf.cuh>
#include <fss/dpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/vdpf.cuh>
#include <fss/group/bytes.cuh>

// Tests of the par_depth execution contract: a par_depth of 0 must keep every
// evaluation path sequential no matter how many OpenMP threads are available,
// and positive or default par_depth values must still produce correct outputs.
// The schemes run with instrumented PRG and hash stubs whose only job is to
// count how many calls are in flight at once.

using BytesGroup = fss::group::Bytes;

namespace {

// Shared between a test and its instrumented stubs. inside tracks calls in
// flight, max_in_flight records the highest overlap ever seen, and stall parks
// every entering call until cleared so tests can force overlap deterministically.
struct CallCounters {
  std::atomic<int> inside{0};
  std::atomic<int> max_in_flight{1};
  std::atomic<bool> stall{false};

  void Enter() {
    int now = inside.fetch_add(1, std::memory_order_relaxed) + 1;
    int prev = max_in_flight.load(std::memory_order_relaxed);
    while (now > prev && !max_in_flight.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {
    }
    while (stall.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
  }

  void Leave() { inside.fetch_sub(1, std::memory_order_relaxed); }
};

__host__ __device__ inline unsigned Mix32(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Deterministic weak expansion of a 16B value into the i-th 16B block. Correctness
// formulas hold for any PRG output, so a mixer is enough to keep the stubs honest.
__host__ __device__ inline int4 MixBlock(int4 seed, int i) {
  unsigned k = 0x9e3779b9u * static_cast<unsigned>(i);
  return {static_cast<int>(Mix32(static_cast<unsigned>(seed.x) + k)),
      static_cast<int>(Mix32(static_cast<unsigned>(seed.y) + k)),
      static_cast<int>(Mix32(static_cast<unsigned>(seed.z) + k)),
      static_cast<int>(Mix32(static_cast<unsigned>(seed.w) + k))};
}

// PRG stub that records how many Gen() calls are in flight.
template <int mul>
class CountingPrg {
public:
  CallCounters *counters = nullptr;

  CountingPrg() = default;
  explicit CountingPrg(CallCounters &c) : counters(&c) {}

  __host__ __device__ cuda::std::array<int4, mul> Gen(int4 seed) {
    cuda::std::array<int4, mul> out{};
#ifndef __CUDA_ARCH__
    if (counters != nullptr) counters->Enter();
#endif
    for (int i = 0; i < mul; ++i) out[i] = MixBlock(seed, i);
#ifndef __CUDA_ARCH__
    if (counters != nullptr) counters->Leave();
#endif
    return out;
  }
};

// XorHash stub with the same instrumentation.
class CountingXorHash {
public:
  CallCounters *counters = nullptr;

  CountingXorHash() = default;
  explicit CountingXorHash(CallCounters &c) : counters(&c) {}

  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4> msg) {
#ifndef __CUDA_ARCH__
    if (counters != nullptr) counters->Enter();
#endif
    int4 mixed = fss::util::Xor(cuda::std::get<0>(msg), cuda::std::get<1>(msg));
    cuda::std::array<int4, 4> out = {MixBlock(mixed, 0), MixBlock(mixed, 1), MixBlock(mixed, 2), MixBlock(mixed, 3)};
#ifndef __CUDA_ARCH__
    if (counters != nullptr) counters->Leave();
#endif
    return out;
  }
};

// Hash stub with the same instrumentation.
class CountingHash {
public:
  CallCounters *counters = nullptr;

  CountingHash() = default;
  explicit CountingHash(CallCounters &c) : counters(&c) {}

  __host__ __device__ cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4> msg) {
#ifndef __CUDA_ARCH__
    if (counters != nullptr) counters->Enter();
#endif
    int4 mixed = fss::util::Xor(fss::util::Xor(msg[0], msg[1]), fss::util::Xor(msg[2], msg[3]));
    cuda::std::array<int4, 2> out = {MixBlock(mixed, 0), MixBlock(mixed, 1)};
#ifndef __CUDA_ARCH__
    if (counters != nullptr) counters->Leave();
#endif
    return out;
  }
};

bool EqualBuffers(const std::vector<int4> &lhs, const std::vector<int4> &rhs) {
  return lhs.size() == rhs.size() && memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(int4)) == 0;
}

constexpr int4 kSeed0 = {static_cast<int>(0x243f6a88u), static_cast<int>(0x85a308d3u), static_cast<int>(0x13198a2eu),
    static_cast<int>(0x03707344u)};
constexpr int4 kSeed1 = {static_cast<int>(0xa4093822u), static_cast<int>(0x299f31d0u), static_cast<int>(0x082efa98u),
    static_cast<int>(0xec4e6c88u)};
constexpr int4 kBeta = {604, 0, 0, 0};
constexpr int4 kHashKey = {static_cast<int>(0x7f4a7c15u), static_cast<int>(0x9e3779b9u), static_cast<int>(0x85ebca6bu),
    static_cast<int>(0xc2b2ae35u)};

void FillSeeds(int4 s0s[2]) {
  s0s[0] = kSeed0;
  s0s[1] = kSeed1;
}

template <int in_bits, int par_depth>
void RunDpfEvalAll(CallCounters &c, std::vector<int4> &ys0, std::vector<int4> &ys1, uint16_t alpha) {
  using DpfType = fss::Dpf<in_bits, BytesGroup, CountingPrg<2>, uint16_t, par_depth>;
  CountingPrg<2> prg{c};
  DpfType dpf{prg};

  int4 s0s[2];
  FillSeeds(s0s);
  typename DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, kBeta);

  ys0.resize(size_t{1} << in_bits);
  ys1.resize(size_t{1} << in_bits);
  dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
  dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));
}

template <int in_bits, int par_depth>
void RunDcfEvalAll(CallCounters &c, std::vector<int4> &ys0, std::vector<int4> &ys1, uint16_t alpha) {
  using DcfType = fss::Dcf<in_bits, BytesGroup, CountingPrg<4>, uint16_t, fss::DcfPred::kLt, par_depth>;
  CountingPrg<4> prg{c};
  DcfType dcf{prg};

  int4 s0s[2];
  FillSeeds(s0s);
  typename DcfType::Key k0, k1;
  dcf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, kBeta);

  ys0.resize(size_t{1} << in_bits);
  ys1.resize(size_t{1} << in_bits);
  dcf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
  dcf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));
}

template <int in_bits, int par_depth>
void RunHalfTreeEvalAll(CallCounters &c, std::vector<int4> &ys0, std::vector<int4> &ys1, uint16_t alpha) {
  using DpfType = fss::HalfTreeDpf<in_bits, BytesGroup, CountingPrg<1>, uint16_t, par_depth>;
  CountingPrg<1> prg{c};
  DpfType dpf{prg, kHashKey};

  int4 s0s[2];
  FillSeeds(s0s);
  typename DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, kBeta);

  ys0.resize(size_t{1} << in_bits);
  ys1.resize(size_t{1} << in_bits);
  dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
  dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));
}

template <int in_bits, int par_depth>
void RunPackedEvalAll(CallCounters &c, std::vector<int4> &ys0, std::vector<int4> &ys1, uint16_t alpha) {
  using DpfType = fss::PackedHalfTreeDpf<in_bits, 1, CountingPrg<1>, uint16_t, par_depth>;
  CountingPrg<1> prg{c};
  DpfType dpf{prg, kHashKey};

  int4 s0s[2];
  FillSeeds(s0s);
  typename DpfType::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, 1);

  constexpr size_t blocks = size_t{1} << DpfType::kDepth;
  ys0.resize(blocks);
  ys1.resize(blocks);
  dpf.EvalAll(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()));
  dpf.EvalAll(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()));
}

template <int in_bits, int par_depth>
void RunVdpfEvalAllMerkle(CallCounters &prg_c, CallCounters &xhash_c, CallCounters &hash_c,
    std::vector<int4> &ys0, std::vector<int4> &ys1, cuda::std::array<int4, 4> &pi0,
    cuda::std::array<int4, 4> &pi1, uint16_t alpha) {
  using VdpfType =
      fss::Vdpf<in_bits, BytesGroup, CountingPrg<2>, CountingXorHash, CountingHash, uint16_t, par_depth>;
  CountingPrg<2> prg{prg_c};
  CountingXorHash xhash{xhash_c};
  CountingHash hash{hash_c};
  VdpfType vdpf{prg, xhash, hash};

  int4 s0s[2];
  FillSeeds(s0s);
  typename VdpfType::Key k0, k1;
  ASSERT_EQ(vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(s0s, 2), alpha, kBeta), 0);

  ys0.resize(size_t{1} << in_bits);
  ys1.resize(size_t{1} << in_bits);
  vdpf.EvalAllMerkle(false, k0, cuda::std::span<int4>(ys0.data(), ys0.size()), pi0);
  vdpf.EvalAllMerkle(true, k1, cuda::std::span<int4>(ys1.data(), ys1.size()), pi1);
}

// Checks the FSS reconstruction: the parties' shares add up to beta on alpha and
// to zero everywhere else.
void ExpectReconstruction(const std::vector<int4> &ys0, const std::vector<int4> &ys1, uint16_t alpha) {
  int4 expected = BytesGroup::From(kBeta).Into();
  int4 zero = {0, 0, 0, 0};
  for (size_t x = 0; x < ys0.size(); ++x) {
    int4 sum = (BytesGroup::From(ys0[x]) + BytesGroup::From(ys1[x])).Into();
    EXPECT_EQ(memcmp(&sum, x == alpha ? &expected : &zero, sizeof(int4)), 0) << "Failed at x=" << x;
  }
}

// Checks the DCF reconstruction with the lt predicate: the parties' shares add up to
// beta below alpha and to zero everywhere else.
void ExpectDcfLtReconstruction(const std::vector<int4> &ys0, const std::vector<int4> &ys1, uint16_t alpha) {
  int4 expected = BytesGroup::From(kBeta).Into();
  int4 zero = {0, 0, 0, 0};
  for (size_t x = 0; x < ys0.size(); ++x) {
    int4 sum = (BytesGroup::From(ys0[x]) + BytesGroup::From(ys1[x])).Into();
    EXPECT_EQ(memcmp(&sum, x < alpha ? &expected : &zero, sizeof(int4)), 0) << "Failed at x=" << x;
  }
}

}  // namespace

TEST(ParDepthTest, CounterDetectsConcurrentCalls) {
  CallCounters c;
  CountingPrg<2> prg{c};
  c.stall = true;

  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&prg] { prg.Gen({1, 2, 3, 4}); });
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  c.stall = false;
  for (auto &t : threads) t.join();

  // Without this proof the "no concurrency" expectations below would be vacuous.
  EXPECT_EQ(c.max_in_flight.load(), 4);
}

TEST(ParDepthTest, DpfEvalAllHonorsZeroParDepth) {
  omp_set_num_threads(4);

  // in_bits 5 takes the per-input small-domain branch, 12 the batched branch.
  CallCounters c_small_seq, c_small_par, c_large_seq, c_large_par;
  std::vector<int4> ys0_small_seq, ys1_small_seq, ys0_small_par, ys1_small_par;
  std::vector<int4> ys0_large_seq, ys1_large_seq, ys0_large_par, ys1_large_par;

  RunDpfEvalAll<5, 0>(c_small_seq, ys0_small_seq, ys1_small_seq, 13);
  RunDpfEvalAll<5, -1>(c_small_par, ys0_small_par, ys1_small_par, 13);
  RunDpfEvalAll<12, 0>(c_large_seq, ys0_large_seq, ys1_large_seq, 107);
  RunDpfEvalAll<12, 2>(c_large_par, ys0_large_par, ys1_large_par, 107);

  EXPECT_EQ(c_small_seq.max_in_flight.load(), 1);
  EXPECT_EQ(c_large_seq.max_in_flight.load(), 1);
  EXPECT_TRUE(EqualBuffers(ys0_small_seq, ys0_small_par));
  EXPECT_TRUE(EqualBuffers(ys1_small_seq, ys1_small_par));
  EXPECT_TRUE(EqualBuffers(ys0_large_seq, ys0_large_par));
  EXPECT_TRUE(EqualBuffers(ys1_large_seq, ys1_large_par));
  ExpectReconstruction(ys0_large_seq, ys1_large_seq, 107);
}

TEST(ParDepthTest, DcfEvalAllHonorsZeroParDepth) {
  omp_set_num_threads(4);

  // in_bits 5 takes the recursive tree branch, 12 the batched frontier.
  CallCounters c_small_seq, c_small_par, c_large_seq, c_large_par;
  std::vector<int4> ys0_small_seq, ys1_small_seq, ys0_small_par, ys1_small_par;
  std::vector<int4> ys0_large_seq, ys1_large_seq, ys0_large_par, ys1_large_par;

  RunDcfEvalAll<5, 0>(c_small_seq, ys0_small_seq, ys1_small_seq, 13);
  RunDcfEvalAll<5, -1>(c_small_par, ys0_small_par, ys1_small_par, 13);
  RunDcfEvalAll<12, 0>(c_large_seq, ys0_large_seq, ys1_large_seq, 107);
  RunDcfEvalAll<12, 2>(c_large_par, ys0_large_par, ys1_large_par, 107);

  EXPECT_EQ(c_small_seq.max_in_flight.load(), 1);
  EXPECT_EQ(c_large_seq.max_in_flight.load(), 1);
  EXPECT_TRUE(EqualBuffers(ys0_small_seq, ys0_small_par));
  EXPECT_TRUE(EqualBuffers(ys1_small_seq, ys1_small_par));
  EXPECT_TRUE(EqualBuffers(ys0_large_seq, ys0_large_par));
  EXPECT_TRUE(EqualBuffers(ys1_large_seq, ys1_large_par));
  ExpectDcfLtReconstruction(ys0_large_seq, ys1_large_seq, 107);
}

TEST(ParDepthTest, HalfTreeDpfEvalAllHonorsZeroParDepth) {
  omp_set_num_threads(4);

  // in_bits 3 takes the per-input small-domain branch, 10 the batched branch.
  CallCounters c_small_seq, c_large_seq;
  std::vector<int4> ys0_small_seq, ys1_small_seq, ys0_large_seq, ys1_large_seq;

  RunHalfTreeEvalAll<3, 0>(c_small_seq, ys0_small_seq, ys1_small_seq, 5);
  RunHalfTreeEvalAll<10, 0>(c_large_seq, ys0_large_seq, ys1_large_seq, 107);

  EXPECT_EQ(c_small_seq.max_in_flight.load(), 1);
  EXPECT_EQ(c_large_seq.max_in_flight.load(), 1);
  ExpectReconstruction(ys0_large_seq, ys1_large_seq, 107);
}

TEST(ParDepthTest, PackedHalfTreeDpfEvalAllHonorsZeroParDepth) {
  omp_set_num_threads(4);

  // in_bits 9 keeps kDepth at 2 for the per-block small-domain branch; 13 makes
  // kDepth 6 so the batched frontier descent runs.
  CallCounters c_small_seq, c_frontier_seq, c_frontier_par;
  std::vector<int4> ys0_small_seq, ys1_small_seq, ys0_frontier_seq, ys1_frontier_seq;
  std::vector<int4> ys0_frontier_par, ys1_frontier_par;

  RunPackedEvalAll<9, 0>(c_small_seq, ys0_small_seq, ys1_small_seq, 13);
  RunPackedEvalAll<13, 0>(c_frontier_seq, ys0_frontier_seq, ys1_frontier_seq, 107);
  RunPackedEvalAll<13, 2>(c_frontier_par, ys0_frontier_par, ys1_frontier_par, 107);

  EXPECT_EQ(c_small_seq.max_in_flight.load(), 1);
  EXPECT_EQ(c_frontier_seq.max_in_flight.load(), 1);
  EXPECT_TRUE(EqualBuffers(ys0_frontier_seq, ys0_frontier_par));
  EXPECT_TRUE(EqualBuffers(ys1_frontier_seq, ys1_frontier_par));

  constexpr int in_bits = 13;
  using DpfType = fss::PackedHalfTreeDpf<in_bits, 1, CountingPrg<1>, uint16_t>;
  for (size_t x = 0; x < (size_t{1} << in_bits); ++x) {
    int4 diff = fss::util::Xor(ys0_frontier_seq[x / DpfType::kLanes], ys1_frontier_seq[x / DpfType::kLanes]);
    EXPECT_EQ(DpfType::Extract(diff, static_cast<uint16_t>(x)), x == 107 ? 1ULL : 0ULL) << "Failed at x=" << x;
  }
}

TEST(ParDepthTest, VdpfEvalAllMerkleHonorsZeroParDepth) {
  omp_set_num_threads(4);

  CallCounters prg_seq, xhash_seq, hash_seq, prg_par, xhash_par, hash_par;
  std::vector<int4> ys0_seq, ys1_seq, ys0_par, ys1_par;
  cuda::std::array<int4, 4> pi0_seq, pi1_seq, pi0_par, pi1_par;

  RunVdpfEvalAllMerkle<12, 0>(prg_seq, xhash_seq, hash_seq, ys0_seq, ys1_seq, pi0_seq, pi1_seq, 107);
  RunVdpfEvalAllMerkle<12, 2>(prg_par, xhash_par, hash_par, ys0_par, ys1_par, pi0_par, pi1_par, 107);

  // The PRG covers the phase-1 tree walk, the XorHash the phase-2 leaf loop, and
  // the Hash the ProveMerkle fold levels.
  EXPECT_EQ(prg_seq.max_in_flight.load(), 1);
  EXPECT_EQ(xhash_seq.max_in_flight.load(), 1);
  EXPECT_EQ(hash_seq.max_in_flight.load(), 1);
  EXPECT_TRUE(EqualBuffers(ys0_seq, ys0_par));
  EXPECT_TRUE(EqualBuffers(ys1_seq, ys1_par));
  ExpectReconstruction(ys0_seq, ys1_seq, 107);

  using VdpfType = fss::Vdpf<12, BytesGroup, CountingPrg<2>, CountingXorHash, CountingHash, uint16_t>;
  EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0_seq), cuda::std::span<const int4, 4>(pi1_seq)));
  EXPECT_TRUE(VdpfType::Verify(cuda::std::span<const int4, 4>(pi0_par), cuda::std::span<const int4, 4>(pi1_par)));
}

TEST(ParDepthTest, VdpfProveMerkleHonorsZeroParDepth) {
  omp_set_num_threads(4);

  CallCounters hash_seq;
  using VdpfType = fss::Vdpf<10, BytesGroup, CountingPrg<2>, CountingXorHash, CountingHash, uint16_t, 0>;
  CountingPrg<2> prg;
  CountingXorHash xhash;
  CountingHash hash{hash_seq};
  VdpfType vdpf{prg, xhash, hash};

  std::vector<cuda::std::array<int4, 4>> leaves(5);
  for (int i = 0; i < 5; ++i) {
    leaves[i] = {MixBlock({i, i + 1, i + 2, i + 3}, 0), MixBlock({i, i + 1, i + 2, i + 3}, 1),
        MixBlock({i, i + 1, i + 2, i + 3}, 2), MixBlock({i, i + 1, i + 2, i + 3}, 3)};
  }
  cuda::std::array<int4, 4> cs = {kHashKey, kSeed0, kSeed1, kBeta};
  cuda::std::array<int4, 4> pi;
  vdpf.ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>>(leaves),
      cuda::std::span<const int4, 4>(cs.data(), 4), pi);

  EXPECT_EQ(hash_seq.max_in_flight.load(), 1);
}
