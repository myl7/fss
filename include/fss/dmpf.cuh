// SPDX-License-Identifier: Apache-2.0
/**
 * @file dmpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed multi-point function (DMPF).
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret
 * Sharing_](https://eprint.iacr.org/2021/580) (@ref vdmpf "1: the published version"),
 * Section 4.
 *
 * ## Definitions
 *
 * **Multi-point function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain
 * $(\sG_{out}, +)$ that is a group, a set of $t$ pairs $(a_j, b_j)$ where $a_j \in \sG_{in}$ and
 * $b_j \in \sG_{out}$, a multi-point function $f$ is a function that for any input $x$, the output
 * $y$ has $y = b_j$ when $x = a_j$ for some $j$, otherwise $y = 0$.
 *
 * - Key generation: $Gen(1^\lambda, \{(a_j, b_j)\}) \rightarrow (k_0, k_1)$.
 * - Batch evaluation: $BatchEval(k_i, \{x\}) \rightarrow \{y_i\}$.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's LSB to 0, corresponding to
 * $\lambda = 127$. See Groupable for more details.
 *
 * We limit the max input domain bit size to 128.
 *
 * The inner DPF uses `uint` as its input type and `bucket_bits` as the domain bit size.
 *
 * ## References
 *
 * 1. Leo de Castro, Antigoni Polychroniadou: Lightweight, Maliciously Secure Verifiable Function
 *    Secret Sharing. EUROCRYPT 2022: 150-179. <https://doi.org/10.1007/978-3-031-06944-4_6>.
 *    @anchor vdmpf
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <span>
#include <vector>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/hash.cuh>
#include <fss/prp.cuh>
#include <fss/util.cuh>
#include <fss/dpf.cuh>
#include <fss/cuckoo_hash.cuh>

namespace fss {

/**
 * 2-party DMPF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam max_points Maximum number of point functions. Must be >= 30.
 *   Sizes arrays at compile time.
 * @tparam bucket_bits Bit size of the inner DPF domain (per bucket).
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam Prp See Permutable. Used for Cuckoo hashing.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam kappa Number of Cuckoo hash functions. 3 is good enough for all practical use cases
 *   (Lemma 5 and Remark 1 of the paper).
 * @tparam ch_lambda Cuckoo-hashing security parameter in bits. Controls the failure probability
 *   of Cuckoo hashing: inserting t elements fails with probability at most $2^{-\text{ch\_lambda}}$.
 */
template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp, typename In = uint,
    int kappa = 3, int ch_lambda = 80>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Groupable<Group> && Prgable<Prg, 2> && Permutable<Prp>)
class Dmpf {
public:
  static_assert(max_points >= 30, "max_points must be >= 30 (Remark 1 of the paper)");
  static constexpr int m = cuckoo_hash::ChBucket(max_points, ch_lambda);
  static constexpr __uint128_t n = __uint128_t(1) << in_bits;
  // b_size = ceil(n * kappa / m), use __uint128_t to avoid overflow.
  static constexpr int b_size = static_cast<int>((static_cast<__uint128_t>(n) * kappa + m - 1) / m);
  static_assert(b_size <= (1 << bucket_bits));

  using InnerDpf = Dpf<bucket_bits, Group, Prg, uint>;

  Prg prg;
  Prp prp;

  /**
     * Per-bucket key containing the inner DPF key data.
     */
  struct BucketKey {
    typename InnerDpf::Cw cws[bucket_bits + 1];
    int4 s0;
  };

  /**
     * DMPF key for one party.
     *
     * Stores the PRP seed, runtime parameters from Gen, and per-bucket inner DPF keys.
     */
  struct Key {
    int4 sigma;
    int m_rt;       ///< Runtime bucket count used during Gen.
    int b_size_rt;  ///< Runtime bucket size used during Gen.
    BucketKey bks[m];
  };

  /**
     * Key generation method.
     *
     * @param k0 Key output for party 0.
     * @param k1 Key output for party 1.
     * @param sigma PRP seed. Users can randomly sample it.
     * @param s0s m pairs of initial seeds for inner DPFs. Users can randomly sample them.
     * @param as Alpha values of t point functions.
     * @param b_bufs Corresponding beta values. Will be clamped.
     * @param t Actual number of points (<= max_points).
     * @param ch_retry Max Cuckoo hash eviction attempts.
     * @return 0 on success, 1 on failure (Cuckoo hash or inner DPF Gen failed).
     */
  int Gen(Key &k0, Key &k1, int4 sigma, cuda::std::span<const cuda::std::array<int4, 2>, m> s0s, std::span<const In> as,
      std::span<const int4> b_bufs, int t, int ch_retry = 1000) {
    assert(t <= max_points);

    k0.sigma = sigma;
    k1.sigma = sigma;

    // Compute runtime bucket count and bucket size.
    assert(t >= 30);
    int m_ = cuckoo_hash::ChBucket(t, ch_lambda);
    assert(m_ <= m);
    __uint128_t n128 = static_cast<__uint128_t>(n);
    int b_rt = static_cast<int>((n128 * kappa + m_ - 1) / m_);
    assert(b_rt <= (1 << bucket_bits));

    k0.m_rt = m_;
    k1.m_rt = m_;
    k0.b_size_rt = b_rt;
    k1.b_size_rt = b_rt;

    // Run Cuckoo hashing.
    cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};
    std::vector<std::pair<int, int>> table(m_, {-1, -1});
    cuckoo_hash::Compact<Prp, In, kappa> compact{prp};
    int ret = compact.Run(as.first(t), m_, sigma, n, b_rt, ch_retry, std::span<std::pair<int, int>>(table));
    if (ret != 0) return 1;

    // Generate inner DPF keys for each bucket.
    InnerDpf inner_dpf{prg};

    for (int i = 0; i < m; ++i) {
      uint a_prime = 0;
      int4 b_buf_prime = {0, 0, 0, 0};

      if (i < m_ && table[i].first != -1) {
        int j = table[i].first;   // index into as
        int k = table[i].second;  // hash function that placed it
        auto [bucket, index] = prp_hash.Locate(sigma, as[j], k, n, b_rt);
        a_prime = static_cast<uint>(index);
        assert(a_prime < (1u << bucket_bits));
        b_buf_prime = b_bufs[j];
      }

      inner_dpf.Gen(k0.bks[i].cws, s0s[i].data(), static_cast<uint>(a_prime), b_buf_prime);

      k0.bks[i].s0 = s0s[i][0];
      k1.bks[i].s0 = s0s[i][1];

      for (int l = 0; l < bucket_bits + 1; ++l) {
        k1.bks[i].cws[l] = k0.bks[i].cws[l];
      }
    }

    return 0;
  }

  /**
     * Batch evaluation method.
     *
     * Evaluates the DMPF key on a batch of input points and produces output shares.
     *
     * @param b Party index. False for 0 and true for 1.
     * @param key This party's key.
     * @param xs Input points to evaluate.
     * @param ys Output shares (pre-allocated, size >= xs.size()). Will be zero-initialized.
     */
  void BatchEval(bool b, const Key &key, std::span<const In> xs, std::span<int4> ys) {
    size_t eta = xs.size();
    assert(ys.size() >= eta);

    int m_ = key.m_rt;
    int b_rt = key.b_size_rt;

    cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};

    // Build per-bucket input lists.
    // inputs[i] = vector of (within_bucket_index, original_input_index).
    std::vector<std::vector<std::pair<uint, size_t>>> inputs(m);

    for (size_t omega = 0; omega < eta; ++omega) {
      for (int k = 0; k < kappa; ++k) {
        auto [bucket, index] = prp_hash.Locate(key.sigma, xs[omega], k, n, b_rt);
        if (bucket >= m) continue;

        uint j = static_cast<uint>(index);
        assert(j < (1u << bucket_bits));

        // Deduplicate within each bucket (linear scan, fine for small kappa).
        bool dup = false;
        for (auto &[existing_j, existing_omega] : inputs[bucket]) {
          if (existing_j == j && existing_omega == omega) {
            dup = true;
            break;
          }
        }
        if (!dup) {
          inputs[bucket].push_back({j, omega});
        }
      }
    }

    // Initialize outputs.
    for (size_t i = 0; i < eta; ++i) {
      ys[i] = {0, 0, 0, 0};
    }

    // Evaluate per bucket.
    InnerDpf inner_dpf{prg};
    for (int i = 0; i < m; ++i) {
      for (auto &[j, omega] : inputs[i]) {
        int4 y = inner_dpf.Eval(b, key.bks[i].s0, key.bks[i].cws, j);

        // Accumulate output.
        ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();
      }
    }
  }

  /**
   * Evaluate the DMPF on all input points.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param key This party's key.
   * @param ys Output shares (pre-allocated, size == n). Will be zero-initialized.
   */
  void EvalAll(bool b, const Key &key, std::span<int4> ys) {
    assert(ys.size() == n);

    int bucket_count = key.m_rt;
    int b_rt = key.b_size_rt;

    // build per-bucket input lists.
    // inputs[i] = vector of (within_bucket_index, original_input_index).
    std::vector<std::vector<std::pair<uint, size_t>>> inputs(bucket_count);
    cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};

    for (size_t omega = 0; omega < n; ++omega) {
      In x = static_cast<In>(omega);

      // store for easier deduplication for this omega
      std::array<std::pair<int, uint>, kappa> seen{};
      int num_seen = 0;

      for (int k = 0; k < kappa; ++k) {
        auto [bucket, index] = prp_hash.Locate(key.sigma, x, k, n, b_rt);
        if (bucket >= bucket_count) continue;

        uint j = static_cast<uint>(index);
        assert(j < (1u << bucket_bits));

        // Deduplicate within each bucket (linear scan, fine for small kappa).
        bool dup = false;

        for (int s = 0; s < num_seen; ++s) {
          if (seen[s].first == bucket && seen[s].second == j) {
            dup = true;
            break;
          }
        }

        if (!dup) {
          inputs[bucket].push_back({j, omega});
          seen[num_seen++] = {bucket, j};
        }
      }
    }

    // initialize outputs.
    for (size_t i = 0; i < n; ++i) {
      ys[i] = {0, 0, 0, 0};
    }

    // evaluate per bucket and merge.
    InnerDpf inner_dpf{prg};
    std::vector<int4> bucket_ys(1ULL << bucket_bits, {0, 0, 0, 0});
    for (int i = 0; i < bucket_count; ++i) {
      inner_dpf.EvalAll(b, key.bks[i].s0, key.bks[i].cws, bucket_ys.data());

      // merge
      for (auto &[j, omega] : inputs[i]) {
        ys[omega] = (Group::From(ys[omega]) + Group::From(bucket_ys[j])).Into();
      }
    }
  }
};
}  // namespace fss
