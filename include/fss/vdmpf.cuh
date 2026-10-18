// SPDX-License-Identifier: Apache-2.0
/**
 * @file vdmpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party verifiable distributed multi-point function (VDMPF).
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
template <int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash, typename Hash,
    typename Prp, typename In = uint, int kappa = 3, int ch_lambda = 80>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
class Vdmpf {
public:
  static_assert(max_points >= 30, "max_points must be >= 30 (Remark 1 of the paper)");
  static_assert(bucket_bits >= 1 && bucket_bits <= 30, "bucket_bits must be in [1, 30]");
  static_assert(kappa > 0, "kappa must be positive");
  static constexpr int m = cuckoo_hash::ChBucket(max_points, ch_lambda);
  static_assert(m > 0, "bucket count must be positive");
  static constexpr __uint128_t n = __uint128_t(1) << in_bits;
  // Bound capacity before forming n * kappa, which may overflow even __uint128_t.
  static_assert(n <= ((__uint128_t{1} << bucket_bits) * m) / kappa,
      "bucket_bits is too small for the bucket size");
  // b_size = ceil(n * kappa / m). The capacity bound keeps the numerator in range.
  static constexpr __uint128_t kBucketSize = (n * kappa + m - 1) / m;
  static constexpr int b_size = static_cast<int>(kBucketSize);

  using InnerVdpf = Vdpf<bucket_bits, Group, Prg, XorHash, Hash, uint>;

  Prg prg;
  XorHash xor_hash;
  Hash hash;
  Prp prp;

  /** Per-bucket key with the released v1.2.0 field order and aggregate layout. */
  struct BucketKey {
    typename InnerVdpf::Cw cws[bucket_bits];
    cuda::std::array<int4, 4> cs;
    int4 ocw;
    int4 s0;
  };

  /** VDMPF key for one party, containing the PRP seed and per-bucket keys. */
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
    __uint128_t bucket_size = (n128 * kappa + m_ - 1) / m_;
    assert(bucket_size <= (uint64_t{1} << bucket_bits));
    int b_rt = static_cast<int>(bucket_size);

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

      ret = inner_vdpf.Gen(k0.bks[i].cws, k0.bks[i].cs, k0.bks[i].ocw, s0s[i], static_cast<uint>(a_prime), b_buf_prime);
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
     * Key generation method.
     *
     * Retries automatically on failure (Cuckoo hashing failed or an inner VDPF Gen failed):
     * up to 128 attempts, each deterministically perturbing sigma and the seeds from the
     * previous one. The effective sigma and seeds of the successful attempt are written back,
     * so after a successful call sigma and s0s hold the values the returned keys carry.
     *
     * @param k0 Key output for party 0.
     * @param k1 Key output for party 1.
     * @param sigma PRP seed. Users can randomly sample it. Written back with the effective
     *   value used by the successful attempt.
     * @param s0s m pairs of initial seeds for inner VDPFs. Users can randomly sample them.
     *   Written back with the effective seeds used by the successful attempt.
     * @param as Alpha values of t point functions.
     * @param b_bufs Corresponding beta values. Will be clamped.
     * @param t Actual number of points (<= max_points).
     * @param ch_retry Max Cuckoo hash eviction attempts.
     * @param retry Whether to retry with perturbed sigma and seeds on failure. If false, Gen
     *   returns 1 immediately after the first failed attempt. Both ch_retry and retry are
     *   explicit in this overload so omitted arguments keep the released fail-fast behavior.
     * @return 0 on success, 1 if the first attempt failed (with retry = false) or all 128
     *   attempts failed (with retry = true).
     */
  int Gen(Key &k0, Key &k1, int4 &sigma, cuda::std::span<cuda::std::array<int4, 2>, m> s0s,
      cuda::std::span<const In> as, cuda::std::span<const int4> b_bufs, int t, int ch_retry,
      bool retry) {
    assert(t <= max_points);
    assert(t >= 30);

    // Compute runtime bucket count and bucket size.
    int m_ = cuckoo_hash::ChBucket(t, ch_lambda);
    assert(m_ <= m);
    __uint128_t n128 = static_cast<__uint128_t>(n);
    __uint128_t bucket_size = (n128 * kappa + m_ - 1) / m_;
    assert(bucket_size <= (uint64_t{1} << bucket_bits));
    int b_rt = static_cast<int>(bucket_size);

    k0.m_rt = m_;
    k1.m_rt = m_;
    k0.b_size_rt = b_rt;
    k1.b_size_rt = b_rt;

    cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};
    cuckoo_hash::Compact<Prp, In, kappa> compact{prp};

    for (int attempt = 0; attempt < 128; ++attempt) {
      k0.sigma = sigma;
      k1.sigma = sigma;

      // Run Cuckoo hashing.
      std::vector<std::pair<int, int>> table(m_, {-1, -1});
      int ret = compact.Run(std::span<const In>(as.data(), t), m_, sigma, n, b_rt, ch_retry,
          std::span<std::pair<int, int>>(table));

      // Generate inner VDPF keys for each bucket.
      if (ret == 0) {
        InnerVdpf inner_vdpf{prg, xor_hash, hash};
        for (int i = 0; i < m && ret == 0; ++i) {
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

          // The inner Gen shares cws, cs, and ocw between the parties and differs in the seed.
          typename InnerVdpf::Key bk0, bk1;
          ret = inner_vdpf.Gen(bk0, bk1, s0s[i], static_cast<uint>(a_prime), b_buf_prime);
          if (ret == 0) {
            for (int l = 0; l < bucket_bits; ++l) {
              k0.bks[i].cws[l] = bk0.cws[l];
              k1.bks[i].cws[l] = bk1.cws[l];
            }
            k0.bks[i].cs = bk0.cs;
            k1.bks[i].cs = bk1.cs;
            k0.bks[i].ocw = bk0.ocw;
            k1.bks[i].ocw = bk1.ocw;
            k0.bks[i].s0 = bk0.s0;
            k1.bks[i].s0 = bk1.s0;
          }
        }
      }
      if (ret == 0) return 0;

      if (!retry) return 1;

      // Deterministically perturb the PRP seed and the initial seeds, then retry.
      sigma.x = util::IncrementWord(sigma.x);
      for (int i = 0; i < m; ++i) {
        s0s[i][0].x = util::IncrementWord(s0s[i][0].x);
        s0s[i][1].y = util::IncrementWord(s0s[i][1].y);
      }
    }

    return 1;
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
    BatchEval<>(b, key, cuda::std::span<const In>(xs.data(), xs.size()),
        cuda::std::span<int4>(ys.data(), ys.size()), pi);
  }

  template <size_t x_extent, size_t y_extent>
  void BatchEval(bool b, const Key &key, cuda::std::span<const In, x_extent> xs,
      cuda::std::span<int4, y_extent> ys, cuda::std::array<int4, 4> &pi) {
    size_t eta = xs.size();
    assert(ys.size() >= eta);

    int m_ = key.m_rt;
    int b_rt = key.b_size_rt;

    cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};

    // Build per-bucket input lists.
    // inputs[i] = vector of (within_bucket_index, original_input_index).
    std::vector<std::vector<std::pair<uint, size_t>>> inputs(m);
    for (size_t omega = 0; omega < eta; ++omega) {
      // store for easier deduplication for this omega
      cuda::std::array<std::pair<int, uint>, kappa> seen{};
      int num_seen = 0;

      for (int k = 0; k < kappa; ++k) {
        auto [bucket, index] = prp_hash.Locate(key.sigma, xs[omega], k, n, b_rt);
        if (bucket >= m) continue;
        uint j = static_cast<uint>(index);
        assert(j < (1u << bucket_bits));
        // Deduplicate among this omega's located records only. Permu is a permutation over
        // [0, n * kappa), so distinct k never collide and the check is only a safeguard.
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

    // Initialize outputs and proof.
    for (size_t i = 0; i < eta; ++i) {
      ys[i] = {0, 0, 0, 0};
    }
    pi = {int4{0, 0, 0, 0}, int4{0, 0, 0, 0}, int4{0, 0, 0, 0}, int4{0, 0, 0, 0}};

    // Evaluate per bucket.
    InnerVdpf inner_vdpf{prg, xor_hash, hash};
    for (int i = 0; i < m; ++i) {
      // Initialize per-bucket proof from this bucket's cs.
      cuda::std::array<int4, 4> pi_bucket = {key.bks[i].cs[0], key.bks[i].cs[1], key.bks[i].cs[2], key.bks[i].cs[3]};

      for (auto &[j, omega] : inputs[i]) {
        int4 y;
        auto pi_tilde = inner_vdpf.Eval(b, key.bks[i].s0, key.bks[i].cws, key.bks[i].cs, key.bks[i].ocw, j, y);

        // Accumulate output.
        ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();

        // Accumulate per-bucket proof (same as Vdpf::Prove logic).
        cuda::std::array<int4, 4> h_input =
            util::Xor(cuda::std::span<const int4, 4>(pi_bucket), cuda::std::span<const int4, 4>(pi_tilde));
        auto h_out = hash.Hash(cuda::std::span<const int4, 4>(h_input));
        pi_bucket[0] = util::Xor(pi_bucket[0], h_out[0]);
        pi_bucket[1] = util::Xor(pi_bucket[1], h_out[1]);
      }

      // Cross-bucket proof accumulation: pi = pi XOR H'(pi XOR pi_bucket).
      cuda::std::array<int4, 4> cross_input =
          util::Xor(cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_bucket));
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

  /**
   * Leaf-chunk size of the Merkle-fold proof accumulation, in leaf nodes.
   * Part of the proof definition: both parties must chunk identically, and the
   * GPU path folds the same chunks.
   */
  static constexpr int kMerkleChunkSize = 256;

  /**
   * Batch evaluation method with the Merkle-fold proof.
   *
   * Parallel-friendly alternative to BatchEval(): same output shares, and the
   * proof replaces the paper's sequential chains with Merkle-style folds. Per
   * bucket, the leaf nodes (hashes of the corrected per-point hashes, in the
   * bucket's canonical (input, hash) order) fold pairwise level by level in
   * fixed kMerkleChunkSize-leaf chunks, an odd trailing node carrying to the
   * next level unchanged; the chunk roots then combine left to right starting
   * from the bucket's hashed cs. The m bucket nodes fold level by level the
   * same way into the final proof, whose last two words stay zero. Every
   * combining step hashes 64B to 32B, so soundness reduces to the pairwise
   * collision resistance of the fold hash exactly like the VDPF Merkle fold;
   * see doc/vdpf_merkle_fold_security.md.
   *
   * The proof keeps the 64B shape, so Verify() is shared, but chain and fold
   * accumulators produce different proofs: both parties must pick the same
   * one. This host path is the sequential reference; the GPU path folds the
   * same construction and matches it bit-exactly.
   *
   * @param b Party index. False for 0 and true for 1.
   * @param key This party's key.
   * @param xs Input points to evaluate.
   * @param ys Output shares (pre-allocated, size >= xs.size()). Will be zero-initialized.
   * @param pi Proof output.
   */
  void BatchEvalMerkle(bool b, const Key &key, cuda::std::span<const In> xs, cuda::std::span<int4> ys,
      cuda::std::array<int4, 4> &pi) {
    size_t eta = xs.size();
    assert(ys.size() >= eta);

    int m_ = key.m_rt;
    int b_rt = key.b_size_rt;

    cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};

    // Build per-bucket input lists in the canonical (input, hash) order: the
    // GPU path reproduces this order through a stable sort of the same
    // per-(input, hash) records.
    std::vector<std::vector<std::pair<uint, size_t>>> inputs(m);

    for (size_t omega = 0; omega < eta; ++omega) {
      for (int k = 0; k < kappa; ++k) {
        auto [bucket, index] = prp_hash.Locate(key.sigma, xs[omega], k, n, b_rt);
        if (bucket >= m) continue;
        uint j = static_cast<uint>(index);
        assert(j < (1u << bucket_bits));
        inputs[bucket].push_back({j, omega});
      }
    }

    for (size_t i = 0; i < eta; ++i) {
      ys[i] = {0, 0, 0, 0};
    }

    InnerVdpf inner_vdpf{prg, xor_hash, hash};
    std::vector<cuda::std::array<int4, 2>> bucket_nodes(m);
    std::vector<cuda::std::array<int4, 2>> nodes;
    for (int i = 0; i < m; ++i) {
      // The bucket's node starts at its hashed cs and absorbs every chunk
      // root of its leaf sequence left to right.
      cuda::std::array<int4, 2> bucket_node = hash.Hash(cuda::std::span<const int4, 4>(key.bks[i].cs));
      size_t consumed = 0;
      while (consumed < inputs[i].size()) {
        size_t take = inputs[i].size() - consumed;
        if (take > static_cast<size_t>(kMerkleChunkSize)) take = kMerkleChunkSize;
        nodes.clear();
        nodes.reserve(take);
        for (size_t c = 0; c < take; ++c) {
          auto [j, omega] = inputs[i][consumed + c];
          int4 y;
        auto pi_tilde = inner_vdpf.Eval(b, key.bks[i].s0, key.bks[i].cws, key.bks[i].cs, key.bks[i].ocw, j, y);
          ys[omega] = (Group::From(ys[omega]) + Group::From(y)).Into();
          nodes.push_back(hash.Hash(cuda::std::span<const int4, 4>(pi_tilde)));
        }
        cuda::std::array<int4, 2> root = FoldNodes(nodes);
        int4 msg[4] = {bucket_node[0], bucket_node[1], root[0], root[1]};
        bucket_node = hash.Hash(cuda::std::span<const int4, 4>(msg));
        consumed += take;
      }
      bucket_nodes[i] = bucket_node;
    }

    cuda::std::array<int4, 2> root = FoldNodes(bucket_nodes);
    pi = {root[0], root[1], int4{0, 0, 0, 0}, int4{0, 0, 0, 0}};
  }

private:
  // Level-by-level pairwise fold with an odd trailing node carrying to the
  // next level unchanged, as in Vdpf::ProveMerkle.
  cuda::std::array<int4, 2> FoldNodes(std::vector<cuda::std::array<int4, 2>> &nodes) {
    size_t count = nodes.size();
    assert(count >= 1);
    while (count > 1) {
      size_t pairs = count / 2;
      for (size_t i = 0; i < pairs; ++i) {
        int4 msg[4] = {nodes[2 * i][0], nodes[2 * i][1], nodes[2 * i + 1][0], nodes[2 * i + 1][1]};
        nodes[i] = hash.Hash(cuda::std::span<const int4, 4>(msg));
      }
      if (count % 2 != 0) nodes[pairs] = nodes[count - 1];
      count = pairs + count % 2;
    }
    return nodes[0];
  }
};

}  // namespace fss
