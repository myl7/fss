// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuckoo_hash.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Cuckoo-hashing utility for DMPF and VDMPF.
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret
 * Sharing_](https://eprint.iacr.org/2021/580) (@ref cuckoo_hash "1: the published version"),
 * Section 4.1 and Algorithm 4.
 *
 * ## References
 *
 * 1. Leo de Castro, Antigoni Polychroniadou: Lightweight, Maliciously Secure Verifiable Function
 *    Secret Sharing. EUROCRYPT 2022: 150-179. <https://doi.org/10.1007/978-3-031-06944-4_6>.
 *    @anchor cuckoo_hash
 */

#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <random>
#include <fss/prp.cuh>

namespace fss::cuckoo_hash {

/**
 * @ingroup fss_api
 * Estimate the number of buckets for three-choice Cuckoo hashing.
 *
 * Uses the fit in Lemma 5 of the paper, simplified per Remark 1 for t >= 30:
 * e = (lambda + 130 + log2(t)) / 123.5 and m = ceil(e * t).
 * This estimate is monotonic in t.
 *
 * @param t Number of elements, at least 30.
 * @param lambda Target Cuckoo-hashing failure exponent in bits.
 * @return Estimated bucket count m.
 * @pre The estimated bucket count is representable by int.
 */
constexpr int ChBucket(int t, int lambda);

namespace detail {

/**
 * Bucket identifier of a rejected candidate.
 *
 * PrpHash::Locate() returns it when the exact bucket index does not fit int,
 * or when the bucket domain size is zero. It equals INT_MAX, so every caller
 * filter of the form `bucket >= bound`, whose bound is an int, rejects the
 * candidate.
 */
constexpr int kRejectedBucket = std::numeric_limits<int>::max();

/** Bucket identifier and 128-bit position returned by PrpHash::Locate(). */
struct LocateResult {
  int bucket;  ///< Bucket index, or kRejectedBucket for a rejected candidate.
  __uint128_t index;  ///< Position within a bucket, 0 for a rejected candidate.
};

/**
 * Three PRP-based Cuckoo-hash choices from Equations (1) and (2) of the paper.
 *
 * @tparam Prp Permutation backend satisfying Permutable.
 * @tparam In Unsigned input type, up to __uint128_t.
 */
template <typename Prp, typename In>
  requires Permutable<Prp>
struct PrpHash {
  Prp prp;  ///< Permutation backend.

  /**
   * Locate x using choice k, where y = PRP(sigma, x + n * k, 3 * n).
   * The bucket is y / b_size and the local position is y % b_size.
   *
   * Evaluation must not trust b_size, because it is key metadata: a candidate
   * whose exact bucket index does not fit int, and every candidate of a zero
   * b_size, is rejected with kRejectedBucket and index 0 instead of wrapping
   * negative. A negative bucket would pass the callers' `bucket >= bound`
   * filters and then index the per-bucket key arrays out of bounds.
   *
   * @param sigma Public permutation seed.
   * @param x Input in [0, n).
   * @param k Hash-function index, 0, 1, or 2.
   * @param n Original input domain size.
   * @param b_size Bucket domain size, positive for a locateable bucket.
   * @return Bucket identifier and local position, or kRejectedBucket and 0.
   * @pre 3 * n fits __uint128_t.
   * Calls support host and device execution when Prp does.
   */
  __host__ __device__ LocateResult Locate(int4 sigma, In x, int k, __uint128_t n, __uint128_t b_size);
};

/**
 * Compact three-choice Cuckoo hashing using Algorithm 4 of the paper.
 *
 * @tparam Prp Permutation backend satisfying Permutable.
 * @tparam In Unsigned input type, up to __uint128_t.
 */
template <typename Prp, typename In>
  requires Permutable<Prp>
struct Compact {
  Prp prp;  ///< Permutation backend.

  /**
   * Place elements into a bucket table using random-walk evictions.
   *
   * Each entry stores an index into as and its hash-function index.
   * An empty entry contains (-1, -1).
   *
   * @param as Input elements in [0, n), with a count representable by int.
   * @param m Positive number of buckets.
   * @param sigma Public permutation seed.
   * @param n Original input domain size, with 3 * n representable by __uint128_t.
   * @param b_size Bucket domain size, equal to ceil(3 * n / m).
   * @param max_evictions Maximum number of evictions before failure.
   * @param table Output table with at least m entries.
   * @return 0 on success or 1 if the eviction limit is exceeded.
   */
  int Run(std::span<const In> as, int m, int4 sigma, __uint128_t n, __uint128_t b_size, int max_evictions,
      std::span<std::pair<int, int>> table);
};

}  // namespace detail

// -----------------------------------------------------------------------------

namespace detail {

constexpr double Log2(double x) {
  if (x <= 0) return -1e308;
  int e = 0;
  double m = x;
  while (m >= 2.0) {
    m /= 2.0;
    ++e;
  }
  while (m < 1.0) {
    m *= 2.0;
    --e;
  }
  double y = (m - 1.0) / (m + 1.0);
  double y2 = y * y, sum = 0, term = y;
  for (int k = 0; k < 40; ++k) {
    sum += term / (2 * k + 1);
    term *= y2;
  }
  return e + 2.0 * sum / 0.6931471805599453;
}

constexpr double Ceil(double x) {
  auto i = static_cast<long long>(x);
  return (x > static_cast<double>(i)) ? static_cast<double>(i + 1) : static_cast<double>(i);
}

/**
 * Exclusive upper bound on the bucket index of a located candidate.
 *
 * PrpHash::Locate() maps every candidate with a positive b_size and an
 * int-representable bucket index into [0, choices * n / b_size), so the largest
 * such bucket is ceil(choices * n / b_size) - 1, and this is the smallest
 * bucket count that covers every located bucket. A candidate outside that
 * interval locates kRejectedBucket instead.
 *
 * Written as (choices * n - 1) / b_size + 1, the equal wrap-free form of the
 * ceiling division: (choices * n + b_size - 1) / b_size gives the same value
 * for every in-range input, but wraps in __uint128_t once b_size exceeds
 * 2^126 with choices * n = 3 * 2^126.
 *
 * @param n Original input domain size, at least 1.
 * @param choices Candidate count per input, at least 1.
 * @param b_size Positive bucket domain size.
 * @return ceil(choices * n / b_size).
 */
constexpr __uint128_t BucketBound(__uint128_t n, int choices, __uint128_t b_size) {
  return (n * choices - 1) / b_size + 1;
}

}  // namespace detail

constexpr int ChBucket(int t, int lambda) {
  // Remark 1 applies for t >= 30.
  // For smaller t, the full Lemma 5 formula with Normal CDF factors is needed,
  // but the paper considers t >= 30 to capture nearly all practical use cases.
  assert(t >= 30);
  double td = static_cast<double>(t);
  double e = (static_cast<double>(lambda) + 130.0 + detail::Log2(td)) / 123.5;
  return static_cast<int>(detail::Ceil(e * td));
}

namespace detail {

template <typename Prp, typename In>
  requires Permutable<Prp>
inline __host__ __device__ LocateResult PrpHash<Prp, In>::Locate(
    int4 sigma, In x, int k, __uint128_t n, __uint128_t b_size) {
  __uint128_t val = static_cast<__uint128_t>(x) + n * k;
  __uint128_t domain_size = n * 3;
  __uint128_t y = prp.Permu(sigma, val, domain_size);

  // Compare the quotient in __uint128_t: a zero b_size has no bucket at all,
  // and an over-int quotient must saturate rather than wrap negative through
  // the cast, because the callers filter buckets with `bucket >= bound` and
  // would accept a negative value before indexing the per-bucket key arrays.
  if (b_size == 0) {
    return {kRejectedBucket, 0};
  }
  __uint128_t bucket = y / b_size;
  if (bucket > static_cast<__uint128_t>(kRejectedBucket)) {
    return {kRejectedBucket, 0};
  }
  return {static_cast<int>(bucket), y % b_size};
}

template <typename Prp, typename In>
  requires Permutable<Prp>
inline int Compact<Prp, In>::Run(std::span<const In> as, int m, int4 sigma, __uint128_t n,
    __uint128_t b_size, int max_evictions, std::span<std::pair<int, int>> table) {
  PrpHash<Prp, In> hasher{prp};
  int t = static_cast<int>(as.size());

  // Initialize table to empty.
  for (int i = 0; i < m; ++i) {
    table[i] = {-1, -1};
  }

  std::mt19937 rng(42);

  for (int omega = 0; omega < t; ++omega) {
    int cur_idx = omega;
    int cur_k = static_cast<int>(rng() % 3);
    int evictions = 0;

    for (;;) {
      auto [bucket, _index] = hasher.Locate(sigma, as[cur_idx], cur_k, n, b_size);
      // Clamp bucket to [0, m).
      bucket = bucket % m;

      if (table[bucket].first == -1) {
        // Empty slot found.
        table[bucket] = {cur_idx, cur_k};
        break;
      }

      // Evict the existing entry and take its place.
      int evicted_idx = table[bucket].first;
      table[bucket] = {cur_idx, cur_k};

      // The evicted element becomes the current element with a new random hash.
      cur_idx = evicted_idx;
      cur_k = static_cast<int>(rng() % 3);

      ++evictions;
      if (evictions > max_evictions) {
        return 1;
      }
    }
  }

  return 0;
}

}  // namespace detail

}  // namespace fss::cuckoo_hash
