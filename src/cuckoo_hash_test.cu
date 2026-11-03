// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for the runtime bucket size. b_size_rt is key metadata
// that the key generator supplies together with the key, and every public CPU
// entry validates it: a value outside [kBucketSize, 2^bucket_bits] throws
// std::invalid_argument before any key array or output is touched. The
// kRejectedBucket checks exercised by the Locate-level tests are detail-level
// defense-in-depth inside cuckoo_hash::detail. Only EXPECT_THROW and
// EXPECT_NO_THROW are used, so both Debug and Release runs cover everything.
//
// The reproducing configuration is in_bits = 31, bucket_bits = 27 and
// b_size_rt = 1: y / b_size_rt >= 2^31 for roughly half of the permuted
// values, and the pre-fix `static_cast<int>` wrapped them negative. A negative
// bucket passed every `bucket >= bound` filter and then indexed the per-bucket
// key arrays out of bounds; Dmpf::BatchEval segfaulted.
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <cuda/std/span>
#include <fss/cuckoo_hash.cuh>
#include <fss/dmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/hash/sha256.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prp/chacha_feistel.cuh>
#include <fss/vdmpf.cuh>

namespace fss {
namespace {

using Group = group::Bytes;
using Prg = prg::ChaCha<2>;
using Prp = prp::ChaChaFeistel;
using Hash = hash::Sha256;

constexpr int kInBits = 31;
constexpr int kMaxPoints = 30;
constexpr int kBucketBits = 27;
using Scheme = Dmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Prp, uint32_t>;
using VdmpfScheme = Vdmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, Hash, Hash, Prp, uint32_t>;

static_assert(Scheme::kBucketCount == 53, "ChBucket(30, 80)");

// Fixed by the three-choice Cuckoo construction, private in the schemes. This
// test sweeps every candidate, so it spells the count out.
constexpr int kCuckooChoices = 3;
// Anchors the local count to the scheme's private one: the smallest bucket
// size derived from this count is exactly the scheme's acceptance boundary.
constexpr __uint128_t kLocalBound =
    (Scheme::kDomainSize * kCuckooChoices + Scheme::kBucketCount - 1) / Scheme::kBucketCount;
static_assert(
    fss::cuckoo_hash::detail::ValidBucketSize(kLocalBound, Scheme::kBucketSize, kBucketBits) &&
    !fss::cuckoo_hash::detail::ValidBucketSize(kLocalBound - 1, Scheme::kBucketSize, kBucketBits));

// Fixed seed for reproducibility. Any seed locates out-of-range candidates.
constexpr int4 kSigma = {0x01234567, static_cast<int>(0x89abcdefu), 0x0fedcba9, 0x76543210};

// Documented rejection sentinel of PrpHash::Locate(): the largest int, so the
// callers' `bucket >= bound` filters reject it. Spelled out instead of
// referencing the header constant so that this test also compiles against the
// pre-fix header, where it must fail.
constexpr int kRejected = std::numeric_limits<int>::max();

// A located bucket is an index into the per-bucket key arrays, so it is either
// inside the allocation or the rejection sentinel. It must never be negative.
TEST(CuckooHashTest, LocateRejectsBucketAboveIntRange) {
  cuckoo_hash::detail::PrpHash<Prp, uint32_t> hasher{Prp{}};
  int rejected = 0;
  for (uint32_t x = 0; x < 4096; ++x) {
    for (int k = 0; k < kCuckooChoices; ++k) {
      const cuckoo_hash::detail::LocateResult location =
          hasher.Locate(kSigma, x, k, Scheme::kDomainSize, 1);
      EXPECT_GE(location.bucket, 0) << "x = " << x << ", k = " << k;
      if (location.bucket == kRejected) {
        ++rejected;
        EXPECT_EQ(location.index, __uint128_t{0}) << "x = " << x << ", k = " << k;
      }
    }
  }
  // The sweep must reach the overflow condition, or it would not cover it.
  EXPECT_GT(rejected, 0);
}

// A key with no bucket domain places no candidate: Locate must reject rather
// than divide by zero.
TEST(CuckooHashTest, LocateRejectsZeroBucketSize) {
  cuckoo_hash::detail::PrpHash<Prp, uint32_t> hasher{Prp{}};
  const cuckoo_hash::detail::LocateResult location =
      hasher.Locate(kSigma, 0, 2, Scheme::kDomainSize, 0);
  EXPECT_EQ(location.bucket, kRejected);
  EXPECT_EQ(location.index, __uint128_t{0});
}

// A b_size_rt outside [kBucketSize, 2^bucket_bits] is a parameter bug: the
// public entries reject it with std::invalid_argument before any assert or
// output write. Before the entry validation this call routed with the tampered
// value and relied on kRejectedBucket to drop every candidate.
TEST(CuckooHashTest, DmpfEntriesRejectOutOfRangeCandidates) {
  Scheme scheme{Prg{0x12345678, static_cast<int>(0x9abcdef0u)}, Prp{}};
  constexpr int kBuckets = Scheme::kBucketCount;
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<Scheme::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<uint32_t> xs;
  for (uint32_t i = 0; i < 64; ++i) xs.push_back(i);
  std::vector<int4> ys(xs.size(), int4{1, 1, 1, 1});
  // Empty on purpose: the entry rejects before the output-size assert, so no
  // full-domain allocation is needed.
  std::vector<int4> all;

  auto batch_eval = [&](__uint128_t b_size_rt) {
    scheme.BatchEval(false, kSigma, b_size_rt, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const Scheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()), cuda::std::span<int4>(ys.data(), ys.size()));
  };
  auto eval_all = [&](__uint128_t b_size_rt) {
    scheme.EvalAll(false, kSigma, b_size_rt, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const Scheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<int4>(all.data(), all.size()));
  };
  const __uint128_t oversized = (__uint128_t{1} << kBucketBits) + 1;
  for (__uint128_t b : {__uint128_t{0}, __uint128_t{1}, Scheme::kBucketSize - 1, oversized}) {
    EXPECT_THROW(batch_eval(b), std::invalid_argument);
    EXPECT_THROW(eval_all(b), std::invalid_argument);
  }
  // The strong guarantee: a rejected call leaves the outputs untouched.
  const int4 one = {1, 1, 1, 1};
  for (size_t i = 0; i < ys.size(); ++i) {
    EXPECT_EQ(memcmp(&ys[i], &one, sizeof(int4)), 0) << "i = " << i;
  }
}

// The same parameter rule on every public Vdmpf CPU entry.
TEST(CuckooHashTest, VdmpfEntriesRejectOutOfRangeCandidates) {
  VdmpfScheme scheme{Prg{0x12345678, static_cast<int>(0x9abcdef0u)}, Hash{int4{1, 2, 3, 4}},
      Hash{int4{5, 6, 7, 8}}, Prp{}};
  constexpr int kBuckets = VdmpfScheme::kBucketCount;
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<VdmpfScheme::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<int4> cs(size_t{kBuckets} * 4);
  std::vector<uint32_t> xs;
  for (uint32_t i = 0; i < 64; ++i) xs.push_back(i);
  std::vector<int4> ys(xs.size(), int4{1, 1, 1, 1});
  std::vector<int4> pi(4, int4{1, 1, 1, 1});
  std::vector<VdmpfScheme::EvalProof> proof_values(xs.size());
  std::vector<int4> all;  // Empty: the full-domain entries reject first.
  int4 y = {1, 1, 1, 1};

  auto batch_eval = [&](__uint128_t b_size_rt) {
    scheme.BatchEval(false, kSigma, b_size_rt, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const VdmpfScheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()),
        cuda::std::span<int4>(ys.data(), ys.size()), cuda::std::span<int4, 4>(pi.data(), 4));
  };
  auto eval = [&](__uint128_t b_size_rt) {
    (void)scheme.Eval(false, kSigma, b_size_rt, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const VdmpfScheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()), 0, y);
  };
  auto prove = [&](__uint128_t b_size_rt) {
    scheme.Prove(kSigma, b_size_rt, cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()),
        cuda::std::span<const VdmpfScheme::EvalProof>(proof_values.data(), proof_values.size()),
        cuda::std::span<int4, 4>(pi.data(), 4));
  };
  auto eval_all = [&](__uint128_t b_size_rt) {
    scheme.EvalAll(false, kSigma, b_size_rt, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const VdmpfScheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<int4>(all.data(), all.size()), cuda::std::span<int4, 4>(pi.data(), 4));
  };
  auto batch_eval_merkle = [&](__uint128_t b_size_rt) {
    scheme.BatchEvalMerkle(false, kSigma, b_size_rt,
        cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const VdmpfScheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()),
        cuda::std::span<int4>(ys.data(), ys.size()), cuda::std::span<int4, 4>(pi.data(), 4));
  };
  auto prove_merkle = [&](__uint128_t b_size_rt) {
    scheme.ProveMerkle(kSigma, b_size_rt, cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()),
        cuda::std::span<const VdmpfScheme::EvalProof>(proof_values.data(), proof_values.size()),
        cuda::std::span<int4, 4>(pi.data(), 4));
  };
  auto eval_all_merkle = [&](__uint128_t b_size_rt) {
    scheme.EvalAllMerkle(false, kSigma, b_size_rt,
        cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const VdmpfScheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<int4>(all.data(), all.size()), cuda::std::span<int4, 4>(pi.data(), 4));
  };
  const __uint128_t oversized = (__uint128_t{1} << kBucketBits) + 1;
  for (__uint128_t b : {__uint128_t{0}, __uint128_t{1}, VdmpfScheme::kBucketSize - 1, oversized}) {
    EXPECT_THROW(batch_eval(b), std::invalid_argument);
    EXPECT_THROW(eval(b), std::invalid_argument);
    EXPECT_THROW(prove(b), std::invalid_argument);
    EXPECT_THROW(eval_all(b), std::invalid_argument);
    EXPECT_THROW(batch_eval_merkle(b), std::invalid_argument);
    EXPECT_THROW(prove_merkle(b), std::invalid_argument);
    EXPECT_THROW(eval_all_merkle(b), std::invalid_argument);
  }
  // The strong guarantee: a rejected call leaves the outputs untouched.
  const int4 one = {1, 1, 1, 1};
  for (size_t i = 0; i < ys.size(); ++i) {
    EXPECT_EQ(memcmp(&ys[i], &one, sizeof(int4)), 0) << "i = " << i;
  }
  for (size_t w = 0; w < 4; ++w) {
    EXPECT_EQ(memcmp(&pi[w], &one, sizeof(int4)), 0) << "w = " << w;
  }
  EXPECT_EQ(memcmp(&y, &one, sizeof(int4)), 0);
}

// Both ends of the valid range evaluate without throwing. The full-domain
// entries are not exercised here; the header comment explains the policy.
TEST(CuckooHashTest, EntriesAcceptBoundaryBucketSizes) {
  Scheme dmpf_scheme{Prg{0x12345678, static_cast<int>(0x9abcdef0u)}, Prp{}};
  VdmpfScheme vdmpf_scheme{Prg{0x12345678, static_cast<int>(0x9abcdef0u)}, Hash{int4{1, 2, 3, 4}},
      Hash{int4{5, 6, 7, 8}}, Prp{}};
  constexpr int kBuckets = VdmpfScheme::kBucketCount;
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<Scheme::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<VdmpfScheme::Cw> vcws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<int4> cs(size_t{kBuckets} * 4);
  std::vector<uint32_t> xs;
  for (uint32_t i = 0; i < 64; ++i) xs.push_back(i);
  std::vector<int4> ys(xs.size());
  std::vector<int4> pi(4);
  std::vector<VdmpfScheme::EvalProof> proof_values(xs.size());

  for (__uint128_t b : {Scheme::kBucketSize, __uint128_t{1} << kBucketBits}) {
    EXPECT_NO_THROW(dmpf_scheme.BatchEval(false, kSigma, b,
        cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const Scheme::Cw>(cws.data(), cws.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()),
        cuda::std::span<int4>(ys.data(), ys.size())));
    int4 y;
    EXPECT_NO_THROW((void)vdmpf_scheme.Eval(false, kSigma, b,
        cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
        cuda::std::span<const VdmpfScheme::Cw>(vcws.data(), vcws.size()),
        cuda::std::span<const int4>(cs.data(), cs.size()), 0, y));
    EXPECT_NO_THROW(vdmpf_scheme.Prove(kSigma, b, cuda::std::span<const int4>(cs.data(), cs.size()),
        cuda::std::span<const uint32_t>(xs.data(), xs.size()),
        cuda::std::span<const VdmpfScheme::EvalProof>(proof_values.data(), proof_values.size()),
        cuda::std::span<int4, 4>(pi.data(), 4)));
  }
}

}  // namespace
}  // namespace fss
