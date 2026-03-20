// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuckoo_hash.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Cuckoo-hashing utility for VDMPF.
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret
 * Sharing_](https://eprint.iacr.org/2024/677) (@ref cuckoo_hash "1: the published version"),
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
#include <span>
#include <utility>
#include <random>
#include <fss/prp.cuh>

namespace fss::cuckoo_hash {

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

}  // namespace detail

/**
 * Compute the number of Cuckoo-hash buckets from Lemma 5 of the paper, simplified per Remark 1.
 *
 * For sufficiently large t (>= 30), the Normal CDF factors in Lemma 5 become effectively 1,
 * giving the simplified formula:
 *   lambda = 123.5 * e - 130 - log2(t)
 *   e = (lambda + 130 + log2(t)) / 123.5
 *   m = ceil(e * t)
 *
 * This formula is monotonic in t, so ChBucket(max_t) >= ChBucket(t) for all t <= max_t.
 *
 * @param t Number of elements. Must be >= 30.
 * @param lambda Cuckoo-hashing security parameter in bits.
 * @return Number of buckets m.
 */
constexpr int ChBucket(int t, int lambda) {
    // Remark 1 applies for t >= 30.
    // For smaller t, the full Lemma 5 formula with Normal CDF factors is needed,
    // but the paper considers t >= 30 to capture nearly all practical use cases.
    assert(t >= 30);
    double td = static_cast<double>(t);
    double e = (static_cast<double>(lambda) + 130.0 + detail::Log2(td)) / 123.5;
    return static_cast<int>(detail::Ceil(e * td));
}

/**
 * PRP-based hash for Cuckoo hashing.
 *
 * Implements the hash functions from Equations (1) and (2) of the paper using a PRP.
 *
 * @tparam Prp PRP type satisfying Permutable.
 * @tparam In Input domain type (unsigned integer up to __uint128_t).
 */
template <typename Prp, typename In>
    requires Permutable<Prp>
struct PrpHash {
    Prp prp;

    /**
     * Compute bucket index and within-bucket index for hash function k.
     *
     * Implements the paper's Equations (1) and (2):
     *   y = PRP(sigma, x + n * k)
     *   bucket = y / b_size
     *   index  = y % b_size
     *
     * @param sigma PRP seed.
     * @param x Input element.
     * @param k Hash function index (0-indexed, in [0, kappa)).
     * @param n Domain size (N = 2^{in_bits}).
     * @param b_size Bucket size B.
     * @return (bucket_index, within_bucket_index).
     */
    std::pair<int, int> Locate(int4 sigma, In x, int k, __uint128_t n, int b_size) {
        __uint128_t val = static_cast<__uint128_t>(x) + n * k;
        __uint128_t domain = n * 3;  // kappa = 3
        __uint128_t y = prp.Permu(sigma, val, domain);

        auto bs = static_cast<__uint128_t>(b_size);
        int bucket = static_cast<int>(y / bs);
        int index = static_cast<int>(y % bs);
        return {bucket, index};
    }
};

/**
 * Compact Cuckoo hashing (Algorithm 4 from the paper).
 *
 * Inserts elements into m buckets using kappa hash functions via Cuckoo hashing with random walk
 * eviction.
 *
 * @tparam Prp PRP type satisfying Permutable.
 * @tparam In Input domain type (unsigned integer up to __uint128_t).
 */
template <typename Prp, typename In, int kappa = 3>
    requires Permutable<Prp>
struct Compact {
    Prp prp;

    /**
     * Run Cuckoo hashing to place elements into a table.
     *
     * Each table entry stores (index_into_as, hash_fn_k), or (-1, -1) if empty.
     *
     * @param as Input elements.
     * @param m Number of buckets.
     * @param sigma PRP seed.
     * @param n Domain size (N = 2^{in_bits}).
     * @param b_size Bucket size B.
     * @param ch_retry Max eviction attempts before failure.
     * @param table Pre-allocated span of size m. Filled on success.
     * @return 0 on success, 1 on failure (too many evictions).
     */
    int Run(std::span<const In> as, int m, int4 sigma, __uint128_t n, int b_size, int ch_retry,
        std::span<std::pair<int, int>> table) {
        PrpHash<Prp, In> hasher{prp};
        int t = static_cast<int>(as.size());

        // Initialize table to empty.
        for (int i = 0; i < m; ++i) {
            table[i] = {-1, -1};
        }

        std::mt19937 rng(42);

        for (int omega = 0; omega < t; ++omega) {
            int cur_idx = omega;
            int cur_k = static_cast<int>(rng() % kappa);
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
                int evicted_k = table[bucket].second;
                table[bucket] = {cur_idx, cur_k};

                // The evicted element becomes the current element with a new random hash.
                cur_idx = evicted_idx;
                cur_k = static_cast<int>(rng() % kappa);

                ++evictions;
                if (evictions > ch_retry) {
                    return 1;
                }
            }
        }

        return 0;
    }
};

}  // namespace fss::cuckoo_hash
