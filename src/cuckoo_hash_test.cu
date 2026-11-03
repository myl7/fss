// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for the Cuckoo-hash bucket identifier. b_size_rt is key
// metadata that the key generator supplies together with the key, so evaluation
// must not let a tampered value place a candidate outside the allocated
// buckets. PrpHash::Locate() rejects such a candidate with kRejectedBucket
// (INT_MAX), which every caller filter of the form `bucket >= bound` rejects.
//
// The reproducing configuration is in_bits = 31, bucket_bits = 27 and
// b_size_rt = 1: y / b_size_rt >= 2^31 for roughly half of the permuted
// values, and the pre-fix `static_cast<int>` wrapped them negative. A negative
// bucket passed every `bucket >= bound` filter and then indexed the per-bucket
// key arrays out of bounds; Dmpf::BatchEval segfaulted.
#include <cstdint>
#include <cstring>
#include <limits>
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
    for (int k = 0; k < Scheme::kCuckooChoices; ++k) {
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

// End-to-end consequence of the defect: with the same tampered b_size_rt every
// candidate is rejected, so the shares stay zero and no per-bucket storage is
// read or written. Before the fix this call wrote through the negative bucket
// index (a std::vector operator[] does not bounds-check) and segfaulted.
TEST(CuckooHashTest, BatchEvalRejectsOutOfRangeCandidates) {
  Scheme scheme{Prg{0x12345678, static_cast<int>(0x9abcdef0u)}, Prp{}};
  constexpr int kBuckets = Scheme::kBucketCount;
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<Scheme::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<uint32_t> xs;
  for (uint32_t i = 0; i < 64; ++i) xs.push_back(i);
  std::vector<int4> ys(xs.size(), int4{1, 1, 1, 1});

  scheme.BatchEval(false, kSigma, 1, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
      cuda::std::span<const Scheme::Cw>(cws.data(), cws.size()),
      cuda::std::span<const uint32_t>(xs.data(), xs.size()), cuda::std::span<int4>(ys.data(), ys.size()));

  const int4 zero = {0, 0, 0, 0};
  for (size_t i = 0; i < ys.size(); ++i) {
    EXPECT_EQ(memcmp(&ys[i], &zero, sizeof(int4)), 0) << "i = " << i;
  }
}

// The split single-input path shares LocateCandidates(), so Vdmpf::Eval must
// reject the same tampered b_size_rt and return a zero share and proof. Before
// the fix its `bucket >= kBucketCount` filter passed the negative bucket and
// Eval() read s0s, cws and cs out of bounds.
TEST(CuckooHashTest, VdmpfEvalRejectsOutOfRangeCandidates) {
  VdmpfScheme scheme{Prg{0x12345678, static_cast<int>(0x9abcdef0u)}, Hash{int4{1, 2, 3, 4}},
      Hash{int4{5, 6, 7, 8}}, Prp{}};
  constexpr int kBuckets = VdmpfScheme::kBucketCount;
  std::vector<int4> s0s(kBuckets, int4{0, 0, 0, 0});
  std::vector<VdmpfScheme::Cw> cws(size_t{kBuckets} * (kBucketBits + 1));
  std::vector<int4> cs(size_t{kBuckets} * 4);
  int4 y = {1, 1, 1, 1};
  auto proof = scheme.Eval(false, kSigma, 1, cuda::std::span<const int4, kBuckets>(s0s.data(), kBuckets),
      cuda::std::span<const VdmpfScheme::Cw>(cws.data(), cws.size()),
      cuda::std::span<const int4>(cs.data(), cs.size()), 0, y);

  const int4 zero = {0, 0, 0, 0};
  EXPECT_EQ(memcmp(&y, &zero, sizeof(int4)), 0);
  for (size_t i = 0; i < proof.size(); ++i) {
    for (size_t w = 0; w < 4; ++w) {
      EXPECT_EQ(memcmp(&proof[i][w], &zero, sizeof(int4)), 0) << "i = " << i << ", w = " << w;
    }
  }
}

}  // namespace
}  // namespace fss
