// SPDX-License-Identifier: Apache-2.0
/**
 * @file vdmpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party verifiable distributed multi-point function (VDMPF).
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret
 * Sharing_](https://eprint.iacr.org/2024/677) (@ref vdmpf "1: the published version"),
 * Section 4.
 *
 * ## Definitions
 *
 * **Multi-point function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain
 * $(\sG_{out}, +)$ that is a group, a set of $t$ pairs $(a_j, b_j)$ where $a_j \in \sG_{in}$ and
 * $b_j \in \sG_{out}$, a multi-point function $f$ is a function that for any input $x$, the output
 * $y$ has $y = b_j$ when $x = a_j$ for some $j$, otherwise $y = 0$.
 *
 * **VDMPF**: extends DMPF with verifiability. Uses Cuckoo hashing to distribute point functions
 * across buckets, then evaluates an inner VDPF per bucket.
 *
 * - Key generation: $Gen(1^\lambda, \{(a_j, b_j)\}) \rightarrow (k_0, k_1)$.
 * - Batch evaluation: $BatchEval(k_i, \{x\}) \rightarrow (\{y_i\}, \pi_i)$.
 * - Verification: $Verify(\pi_0, \pi_1) \rightarrow \{Accept, Reject\}$.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's LSB to 0, corresponding to
 * $\lambda = 127$. See Groupable for more details.
 *
 * We limit the max input domain bit size to 128.
 *
 * The inner VDPF uses `uint` as its input type and `bucket_bits` as the domain bit size.
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
#include <fss/vdpf.cuh>
#include <fss/cuckoo_hash.cuh>

namespace fss {

/**
 * 2-party VDMPF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam max_points Maximum number of point functions. Must be >= 30.
 *   Sizes arrays at compile time.
 * @tparam bucket_bits Bit size of the inner VDPF domain (per bucket).
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam XorHash See XorHashable. Paper's $H$: maps $(x, s)$ to $4\lambda$ bits.
 * @tparam Hash See Hashable. Paper's $H'$: maps $4\lambda$ bits to $2\lambda$ bits.
 * @tparam Prp See Permutable. Used for Cuckoo hashing.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam kappa Number of Cuckoo hash functions. 3 is good enough for all practical use cases
 *   (Lemma 5 and Remark 1 of the paper).
 * @tparam ch_lambda Cuckoo-hashing security parameter in bits. Controls the failure probability
 *   of Cuckoo hashing: inserting t elements fails with probability at most $2^{-\text{ch\_lambda}}$.
 */
template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg,
    typename XorHash, typename Hash, typename Prp, typename In = uint, int kappa = 3,
    int ch_lambda = 80>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> &&
        Hashable<Hash> && Permutable<Prp>)
class Vdmpf {
public:
    static_assert(max_points >= 30, "max_points must be >= 30 (Remark 1 of the paper)");
    static constexpr int m = cuckoo_hash::ChBucket(max_points, ch_lambda);
    static constexpr __uint128_t n = __uint128_t(1) << in_bits;
    // b_size = ceil(n * kappa / m), use __uint128_t to avoid overflow.
    static constexpr int b_size =
        static_cast<int>((static_cast<__uint128_t>(n) * kappa + m - 1) / m);
    static_assert(b_size <= (1 << bucket_bits));

    using InnerVdpf = Vdpf<bucket_bits, Group, Prg, XorHash, Hash, uint>;

    Prg prg;
    XorHash xor_hash;
    Hash hash;
    Prp prp;

    /**
     * Per-bucket key containing the inner VDPF key data.
     */
    struct BucketKey {
        typename InnerVdpf::Cw cws[bucket_bits];
        cuda::std::array<int4, 4> cs;
        int4 ocw;
        int4 s0;
    };

    /**
     * VDMPF key for one party.
     *
     * Stores the PRP seed, runtime parameters from Gen, and per-bucket inner VDPF keys.
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
     * @param s0s m pairs of initial seeds for inner VDPFs. Users can randomly sample them.
     * @param as Alpha values of t point functions.
     * @param b_bufs Corresponding beta values. Will be clamped.
     * @param t Actual number of points (<= max_points).
     * @param ch_retry Max Cuckoo hash eviction attempts.
     * @return 0 on success, 1 on failure (Cuckoo hash or inner VDPF Gen failed).
     */
    int Gen(Key &k0, Key &k1, int4 sigma, cuda::std::span<const cuda::std::array<int4, 2>, m> s0s,
        std::span<const In> as, std::span<const int4> b_bufs, int t, int ch_retry = 1000) {
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
        cuckoo_hash::PrpHash<Prp, In> prp_hash{prp};
        std::vector<std::pair<int, int>> table(m_, {-1, -1});
        cuckoo_hash::Compact<Prp, In, kappa> compact{prp};
        int ret = compact.Run(
            as.first(t), m_, sigma, n, b_rt, ch_retry, std::span<std::pair<int, int>>(table));
        if (ret != 0) return 1;

        // Generate inner VDPF keys for each bucket.
        InnerVdpf inner_vdpf{prg, xor_hash, hash};
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

            ret = inner_vdpf.Gen(k0.bks[i].cws, k0.bks[i].cs, k0.bks[i].ocw, s0s[i],
                static_cast<uint>(a_prime), b_buf_prime);
            if (ret != 0) return 1;

            k0.bks[i].s0 = s0s[i][0];
            k1.bks[i].s0 = s0s[i][1];
            // Copy shared parts.
            for (int l = 0; l < bucket_bits; ++l) k1.bks[i].cws[l] = k0.bks[i].cws[l];
            k1.bks[i].cs = k0.bks[i].cs;
            k1.bks[i].ocw = k0.bks[i].ocw;
        }

        return 0;
    }

    /**
     * Batch verifiable evaluation method.
     *
     * Evaluates the VDMPF key on a batch of input points and produces output shares and a proof.
     *
     * @param b Party index. False for 0 and true for 1.
     * @param key This party's key.
     * @param xs Input points to evaluate.
     * @param ys Output shares (pre-allocated, size >= xs.size()). Will be zero-initialized.
     * @param pi Proof output.
     */
    void BatchEval(bool b, const Key &key, std::span<const In> xs, std::span<int4> ys,
        cuda::std::array<int4, 4> &pi) {
        size_t eta = xs.size();
        assert(ys.size() >= eta);

        int m_ = key.m_rt;
        int b_rt = key.b_size_rt;

        cuckoo_hash::PrpHash<Prp, In> prp_hash{prp};

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

        // Initialize outputs and proof.
        for (size_t i = 0; i < eta; ++i) {
            ys[i] = {0, 0, 0, 0};
        }
        pi = {int4{0, 0, 0, 0}, int4{0, 0, 0, 0}, int4{0, 0, 0, 0}, int4{0, 0, 0, 0}};

        // Evaluate per bucket.
        InnerVdpf inner_vdpf{prg, xor_hash, hash};
        for (int i = 0; i < m; ++i) {
            // Initialize per-bucket proof from this bucket's cs.
            cuda::std::array<int4, 4> pi_bucket = {
                key.bks[i].cs[0], key.bks[i].cs[1], key.bks[i].cs[2], key.bks[i].cs[3]};

            for (auto &[j, omega] : inputs[i]) {
                int4 y;
                auto pi_tilde = inner_vdpf.Eval(b, key.bks[i].s0,
                    cuda::std::span<const typename InnerVdpf::Cw>(key.bks[i].cws, bucket_bits),
                    cuda::std::span<const int4, 4>(key.bks[i].cs), key.bks[i].ocw, j, y);

                // Accumulate output.
                ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();

                // Accumulate per-bucket proof (same as Vdpf::Prove logic).
                cuda::std::array<int4, 4> h_input =
                    util::Xor(cuda::std::span<const int4, 4>(pi_bucket),
                        cuda::std::span<const int4, 4>(pi_tilde));
                auto h_out = hash.Hash(cuda::std::span<const int4, 4>(h_input));
                pi_bucket[0] = util::Xor(pi_bucket[0], h_out[0]);
                pi_bucket[1] = util::Xor(pi_bucket[1], h_out[1]);
            }

            // Cross-bucket proof accumulation: pi = pi XOR H'(pi XOR pi_bucket).
            cuda::std::array<int4, 4> cross_input = util::Xor(
                cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_bucket));
            auto cross_out = hash.Hash(cuda::std::span<const int4, 4>(cross_input));
            pi[0] = util::Xor(pi[0], cross_out[0]);
            pi[1] = util::Xor(pi[1], cross_out[1]);
        }
    }

    /**
     * Verification method.
     *
     * @return True if proofs match (Accept), false otherwise (Reject).
     */
    static bool Verify(cuda::std::span<const int4, 4> pi0, cuda::std::span<const int4, 4> pi1) {
        return InnerVdpf::Verify(pi0, pi1);
    }
};

}  // namespace fss
