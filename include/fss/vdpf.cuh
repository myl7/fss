// SPDX-License-Identifier: Apache-2.0
/**
 * @file vdpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party verifiable distributed point function (VDPF).
 *
 * The scheme is from the paper, [_Lightweight, Maliciously Secure Verifiable Function Secret Sharing_](https://eprint.iacr.org/2021/580) (@ref vdpf "1: the published version").
 *
 * ## Definitions
 *
 * **Point function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, and $b \in \sG_{out}$, a point function $f_{a, b}$ is a function that for any input $x$, the output $y$ has $y = b$ only when $x = a$, otherwise $y = 0$.
 *
 * **VDPF**: extends DPF with verifiability. The evaluation produces both output shares and a proof. Two parties can compare proofs to detect malicious key modification.
 *
 * - Key generation: $Gen(1^\lambda, f_{a, b}) \rightarrow (k_0, k_1)$.
 * - Evaluation: $Eval(k_i, x) \rightarrow (y_{i,x}, \tilde\pi_{i,x})$.
 * - Proof accumulation: $Prove(\{\tilde\pi\}, cs) \rightarrow \pi$. An order-independent
 *   alternative, `ProveMerkle`/`EvalAllMerkle`, replaces the sequential chain with a Merkle
 *   fold over the corrected per-point hashes; see doc/vdpf_merkle_fold_security.md.
 * - Verification: $Verify(\pi_0, \pi_1) \rightarrow \{Accept, Reject\}$.
 *
 * ## Implementation Details
 *
 * We fix the output domain size at 16B and always set the last word's LSB to 0, corresponding to $\lambda = 127$.
 * See Groupable for more details.
 *
 * We limit the max input domain bit size to 128.
 * This is enough for most applications and allows us to represent the input as an integer.
 *
 * ## References
 *
 * 1. Leo de Castro, Antigoni Polychroniadou: Lightweight, Maliciously Secure Verifiable Function Secret Sharing. EUROCRYPT 2022: 150-179. <https://doi.org/10.1007/978-3-031-06944-4_6>. @anchor vdpf
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <vector>
#include <omp.h>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/hash.cuh>
#include <fss/util.cuh>

namespace fss {

/**
 * 2-party VDPF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam XorHash See XorHashable. Paper's $H$: maps $(x, s)$ to $4\lambda$ bits.
 * @tparam Hash See Hashable. Paper's $H'$: maps $4\lambda$ bits to $2\lambda$ bits.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In = uint,
    int par_depth = -1>
  requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) && in_bits <= sizeof(In) * 8 &&
      Groupable<Group> && Prgable<Prg, 2> && XorHashable<XorHash> && Hashable<Hash>)
class Vdpf {
public:
  Prg prg;
  XorHash xor_hash;
  Hash hash;

  /**
     * Correction word.
     *
     * ## Layout
     *
     * According to the paper, there are s, tl, tr to be stored.
     * tl is stored at the clamped bit of s.
     */
  struct __align__(32) Cw {
    int4 s;
    bool tr;
  };
  // For only 1 and aligned memory access on GPU
  static_assert(sizeof(Cw) == 32);

  /**
     * A party's key: the correction words, the correction seeds, the output correction word,
     * plus the party's initial seed.
     * Both parties share the same cws, cs, and ocw and differ in the seed.
     */
  struct Key {
    Cw cws[in_bits];
    cuda::std::array<int4, 4> cs;
    int4 ocw;
    int4 s0;
  };

  /**
     * Key generation method.
     *
     * @param k0, k1 Returned keys of the two parties.
     * @param s0s 2 initial seeds. Users can randomly sample them.
     * @param a $a$.
     * @param b_buf $b$. Will be clamped and converted to the group element.
     * @return 0 on success, 1 if cs is all zeros or $t_0 = t_1$ at the end (caller should resample seeds and retry).
     *
     * Both parties share the returned cws, cs, and ocw and differ in s0.
     */
  __host__ __device__ int Gen(Key &k0, Key &k1, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf) {
    int ret = Gen(k0.cws, k0.cs, k0.ocw, s0s, a, b_buf);
    k0.s0 = s0s[0];
    for (int i = 0; i < in_bits; ++i) k1.cws[i] = k0.cws[i];
    k1.cs = k0.cs;
    k1.ocw = k0.ocw;
    k1.s0 = s0s[1];
    return ret;
  }

  /**
     * Evaluation result.
     */
  struct EvalResult {
    int4 y;  ///< Output share. $y_{i,x}$.
    cuda::std::array<int4, 4> pi_tilde;  ///< Corrected per-point hash ($\correct(\tilde\pi, cs, t)$).
  };

  /**
     * Evaluation method.
     *
     * @param b Party index. False for 0 and true for 1. $i$.
     * @param key The party's key returned by Gen().
     * @param x Evaluated input. $x$.
     * @return The output share $y_{i,x}$ plus the corrected per-point hash for proof accumulation.
     */
  __host__ __device__ EvalResult Eval(bool b, const Key &key, In x) {
    int4 s = key.s0;
    s = util::SetLsb(s, false);
    bool t = b;

    for (int i = 0; i < in_bits; ++i) {
      Cw cw = key.cws[i];
      int4 s_cw = cw.s;
      bool tl_cw = util::GetLsb(s_cw);
      s_cw = util::SetLsb(s_cw, false);
      bool tr_cw = cw.tr;

      auto [sl, sr] = prg.Gen(s);

      bool tl = util::GetLsb(sl);
      sl = util::SetLsb(sl, false);
      bool tr = util::GetLsb(sr);
      sr = util::SetLsb(sr, false);

      if (t) {
        sl = util::Xor(sl, s_cw);
        sr = util::Xor(sr, s_cw);
        tl = tl ^ tl_cw;
        tr = tr ^ tr_cw;
      }

      bool x_bit = (x >> (in_bits - 1 - i)) & 1;

      if (!x_bit) {
        s = sl;
        t = tl;
      } else {
        s = sr;
        t = tr;
      }
    }

    // Output share
    auto g = Group::From(s);
    assert((key.ocw.w & 1) == 0);
    if (t) g = g + Group::From(key.ocw);
    if (b) g = -g;

    // Corrected verification hash
    int4 x_buf = util::Pack(x);

    auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, s});
    if (t) {
      pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(key.cs));
    }
    return {g.Into(), pi_tilde};
  }

  /**
     * Proof accumulation method.
     *
     * Accumulates corrected per-point hashes (from Eval()) into a single proof.
     *
     * @param pi_tildes Corrected per-point hashes returned by Eval().
     * @param cs Returned by Gen().
     * @param pi Proof output.
     */
  void Prove(cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes, cuda::std::span<const int4, 4> cs,
      cuda::std::array<int4, 4> &pi) {
    pi = {cs[0], cs[1], cs[2], cs[3]};
    for (size_t i = 0; i < pi_tildes.size(); ++i) {
      cuda::std::array<int4, 4> h_input =
          util::Xor(cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_tildes[i]));
      auto h_out = hash.Hash(cuda::std::span<const int4, 4>(h_input));
      pi[0] = util::Xor(pi[0], h_out[0]);
      pi[1] = util::Xor(pi[1], h_out[1]);
    }
  }

  /**
     * Verification method.
     *
     * @return True if proofs match (Accept), false otherwise (Reject).
     */
  __host__ __device__ static bool Verify(cuda::std::span<const int4, 4> pi0, cuda::std::span<const int4, 4> pi1) {
    for (int i = 0; i < 4; ++i) {
      if (pi0[i].x != pi1[i].x || pi0[i].y != pi1[i].y || pi0[i].z != pi1[i].z || pi0[i].w != pi1[i].w) return false;
    }
    return true;
  }

  /**
     * Full domain evaluation method.
     *
     * Evaluate the key on each input, i.e., 0b00...0 - 0b11...1.
     * Store the outputs sequentially and accumulate the proof.
     *
     * @param b Party index. False for 0 and true for 1.
     * @param key The party's key returned by Gen().
     * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
     * @param pi Proof output.
     *
     * Support parallel using OpenMP for the tree traversal phase.
     * The task is divided to 2 ** par_depth parallel sub-tasks with the equal workloads.
     * par_depth = -1: use ceil(log(num of threads)).
     * par_depth = 0: no parallelism, i.e., sequential execution.
     */
  void EvalAll(bool b, const Key &key, cuda::std::span<int4> ys, cuda::std::array<int4, 4> &pi) {
    int4 st = key.s0;
    bool t = b;
    st = util::SetLsb(st, t);

    assert(in_bits < sizeof(size_t) * 8);
    size_t l = 0;
    size_t r = 1ULL << in_bits;
    int i = 0;

    int par_depth_ = util::ResolveParDepth(par_depth);

    // Phase 1: tree traversal, store (s, t) packed into ys temporarily
#pragma omp parallel
#pragma omp single
    EvalTree(st, key.cws, ys, l, r, i, par_depth_);

    // Phase 2: sequential output computation and proof accumulation
    pi = {key.cs[0], key.cs[1], key.cs[2], key.cs[3]};
    size_t n = 1ULL << in_bits;
    assert((key.ocw.w & 1) == 0);
    auto ocw_group = Group::From(key.ocw);
    for (size_t j = 0; j < n; ++j) {
      int4 sj = ys[j];
      bool tj = util::GetLsb(sj);
      sj = util::SetLsb(sj, false);

      // Output share
      auto g = Group::From(sj);
      if (tj) g = g + ocw_group;
      if (b) g = -g;
      ys[j] = g.Into();

      // Proof accumulation
      int4 x_buf = util::Pack(static_cast<In>(j));

      auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, sj});
      if (tj) {
        pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(key.cs));
      }

      cuda::std::array<int4, 4> h_input =
          util::Xor(cuda::std::span<const int4, 4>(pi), cuda::std::span<const int4, 4>(pi_tilde));
      auto h_out = hash.Hash(cuda::std::span<const int4, 4>(h_input));
      pi[0] = util::Xor(pi[0], h_out[0]);
      pi[1] = util::Xor(pi[1], h_out[1]);
    }
  }

  /**
   * Proof accumulation method (Merkle fold).
   *
   * Order-independent alternative to Prove(): hashes the corrected per-point hashes into the
   * leaves of a binary Merkle tree over the given sequence and folds adjacent nodes with the
   * same hash level by level (an odd trailing node carries to the next level unchanged). The
   * proof is {root, cs[2], cs[3]}, the same 64B shape as the chain's, so Verify() is shared.
   * Every level is an OpenMP parallel batch. See doc/vdpf_merkle_fold_security.md.
   *
   * @param pi_tildes Corrected per-point hashes returned by Eval().
   * @param cs Returned by Gen().
   * @param pi Proof output.
   */
  void ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>> pi_tildes, cuda::std::span<const int4, 4> cs,
      cuda::std::array<int4, 4> &pi) {
    size_t count = pi_tildes.size();
    assert(count >= 1);

    std::vector<cuda::std::array<int4, 2>> nodes(count);
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      nodes[i] = hash.Hash(cuda::std::span<const int4, 4>(pi_tildes[i]));
    }

    while (count > 1) {
      size_t pairs = count / 2;
      std::vector<cuda::std::array<int4, 2>> next(pairs + count % 2);
#pragma omp parallel for
      for (size_t i = 0; i < pairs; ++i) {
        int4 msg[4] = {nodes[2 * i][0], nodes[2 * i][1], nodes[2 * i + 1][0], nodes[2 * i + 1][1]};
        next[i] = hash.Hash(cuda::std::span<const int4, 4>(msg));
      }
      if (count % 2 != 0) next[pairs] = nodes[count - 1];
      nodes = std::move(next);
      count = nodes.size();
    }

    pi = {nodes[0][0], nodes[0][1], cs[2], cs[3]};
  }

  /**
   * Full domain evaluation method with the Merkle-fold proof.
   *
   * Same output shares as EvalAll(); the proof comes from ProveMerkle() over the corrected
   * per-point hashes. The conversion phase and every fold level are OpenMP parallel, unlike
   * EvalAll()'s sequential phase 2. See doc/vdpf_merkle_fold_security.md.
   *
   * @param b Party index. False for 0 and true for 1. $i$.
   * @param key The party's key returned by Gen().
   * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
   * @param pi Proof output.
   */
  void EvalAllMerkle(bool b, const Key &key, cuda::std::span<int4> ys, cuda::std::array<int4, 4> &pi) {
    int4 st = key.s0;
    bool t = b;
    st = util::SetLsb(st, t);

    assert(in_bits < sizeof(size_t) * 8);
    size_t l = 0;
    size_t r = 1ULL << in_bits;
    int i = 0;

    int par_depth_ = util::ResolveParDepth(par_depth);

    // Phase 1: tree traversal, store (s, t) packed into ys temporarily
#pragma omp parallel
#pragma omp single
    EvalTree(st, key.cws, ys, l, r, i, par_depth_);

    // Phase 2: parallel output conversion and leaf hashing, then the Merkle fold
    pi = {key.cs[0], key.cs[1], key.cs[2], key.cs[3]};
    size_t n = 1ULL << in_bits;
    assert((key.ocw.w & 1) == 0);
    auto ocw_group = Group::From(key.ocw);
    std::vector<cuda::std::array<int4, 4>> leaves(n);
#pragma omp parallel for
    for (size_t j = 0; j < n; ++j) {
      int4 sj = ys[j];
      bool tj = util::GetLsb(sj);
      sj = util::SetLsb(sj, false);

      auto g = Group::From(sj);
      if (tj) g = g + ocw_group;
      if (b) g = -g;
      ys[j] = g.Into();

      int4 x_buf = util::Pack(static_cast<In>(j));

      auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, sj});
      if (tj) {
        leaves[j] = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(key.cs));
      } else {
        leaves[j] = pi_tilde;
      }
    }
    ProveMerkle(cuda::std::span<const cuda::std::array<int4, 4>>(leaves), cuda::std::span<const int4, 4>(key.cs), pi);
  }

private:
  /**
     * Key generation core on the correction-word form.
     *
     * @param cws Pre-allocated array of Cw as returns. The array size must be in_bits.
     * @param cs Correction seed output ($4\lambda$ bits).
     * @param ocw Output correction word output.
     * @param s0s 2 initial seeds. Users can randomly sample them.
     * @param a $a$.
     * @param b_buf $b$. Will be clamped and converted to the group element.
     * @return 0 on success, 1 if cs is all zeros or $t_0 = t_1$ at the end (caller should resample seeds and retry).
     *
     * The key for party i consists of cws + cs + ocw + s0s[i].
     */
  __host__ __device__ int Gen(
      Cw cws[], cuda::std::array<int4, 4> &cs, int4 &ocw, cuda::std::span<const int4, 2> s0s, In a, int4 b_buf) {
    int4 s0 = s0s[0];
    s0 = util::SetLsb(s0, false);
    int4 s1 = s0s[1];
    s1 = util::SetLsb(s1, false);
    bool t0 = false;
    bool t1 = true;
    b_buf = util::SetLsb(b_buf, false);

    for (int i = 0; i < in_bits; ++i) {
      auto [s0l, s0r] = prg.Gen(s0);
      auto [s1l, s1r] = prg.Gen(s1);

      bool t0l = util::GetLsb(s0l);
      s0l = util::SetLsb(s0l, false);
      bool t0r = util::GetLsb(s0r);
      s0r = util::SetLsb(s0r, false);
      bool t1l = util::GetLsb(s1l);
      s1l = util::SetLsb(s1l, false);
      bool t1r = util::GetLsb(s1r);
      s1r = util::SetLsb(s1r, false);

      bool a_bit = (a >> (in_bits - 1 - i)) & 1;

      int4 s_cw;
      if (!a_bit) s_cw = util::Xor(s0r, s1r);
      else s_cw = util::Xor(s0l, s1l);

      bool tl_cw = t0l ^ t1l ^ a_bit ^ 1;
      bool tr_cw = t0r ^ t1r ^ a_bit;

      if (!a_bit) {
        s0 = s0l;
        if (t0) s0 = util::Xor(s0, s_cw);
        s1 = s1l;
        if (t1) s1 = util::Xor(s1, s_cw);

        if (t0) t0 = t0l ^ tl_cw;
        else t0 = t0l;
        if (t1) t1 = t1l ^ tl_cw;
        else t1 = t1l;
      } else {
        s0 = s0r;
        if (t0) s0 = util::Xor(s0, s_cw);
        s1 = s1r;
        if (t1) s1 = util::Xor(s1, s_cw);

        if (t0) t0 = t0r ^ tr_cw;
        else t0 = t0r;
        if (t1) t1 = t1r ^ tr_cw;
        else t1 = t1r;
      }

      s_cw = util::SetLsb(s_cw, tl_cw);
      // Both 16B halves written so the full 32B slot gets unmasked stores.
      // {s_cw, tr_cw} alone leaves padding undefined: NVCC emits a masked
      // 256-bit store, and the partial-sector write forces a write-allocate
      // round trip per level on GPU (measured ~2x Gen time).
      int4 *cw_slot = reinterpret_cast<int4 *>(&cws[i]);
      cw_slot[0] = s_cw;
      cw_slot[1] = make_int4(tr_cw, 0, 0, 0);
    }

    // Verification hash
    int4 a_buf = util::Pack(a);

    auto pi_tilde_0 = xor_hash.Hash(cuda::std::tuple<int4, const int4>{a_buf, s0});
    auto pi_tilde_1 = xor_hash.Hash(cuda::std::tuple<int4, const int4>{a_buf, s1});
    cs = util::Xor(cuda::std::span<const int4, 4>(pi_tilde_0), cuda::std::span<const int4, 4>(pi_tilde_1));

    // A zero cs would let adversarial keys keep both parties' per-point hashes equal while
    // their control bits differ everywhere (see Assumption 2 of doc/vdpf_merkle_fold_security.md).
    if ((cs[0].x | cs[0].y | cs[0].z | cs[0].w | cs[1].x | cs[1].y | cs[1].z | cs[1].w | cs[2].x | cs[2].y |
            cs[2].z | cs[2].w | cs[3].x | cs[3].y | cs[3].z | cs[3].w) == 0) {
      return 1;
    }

    // Check retry condition
    if (t0 == t1) return 1;

    // Output correction word
    auto v_cw = Group::From(b_buf) + (-Group::From(s0)) + Group::From(s1);
    if (t1) v_cw = -v_cw;
    ocw = v_cw.Into();

    return 0;
  }

  void EvalTree(
      int4 st, cuda::std::span<const Cw> cws, cuda::std::span<int4> ys, size_t l, size_t r, int i, int par_depth_) {
    if (i == in_bits) {
      assert(l + 1 == r);
      ys[l] = st;
      return;
    }

    bool t = util::GetLsb(st);
    int4 s = st;
    s = util::SetLsb(s, false);

    Cw cw = cws[i];
    int4 s_cw = cw.s;
    bool tl_cw = util::GetLsb(s_cw);
    s_cw = util::SetLsb(s_cw, false);
    bool tr_cw = cw.tr;

    auto [sl, sr] = prg.Gen(s);

    bool tl = util::GetLsb(sl);
    sl = util::SetLsb(sl, false);
    bool tr = util::GetLsb(sr);
    sr = util::SetLsb(sr, false);

    if (t) {
      sl = util::Xor(sl, s_cw);
      sr = util::Xor(sr, s_cw);
      tl = tl ^ tl_cw;
      tr = tr ^ tr_cw;
    }

    int4 stl = sl;
    stl = util::SetLsb(stl, tl);
    int4 str = sr;
    str = util::SetLsb(str, tr);

    size_t mid = (l + r) / 2;

    if (i < par_depth_) {
#pragma omp task
      EvalTree(stl, cws, ys, l, mid, i + 1, par_depth_);
#pragma omp task
      EvalTree(str, cws, ys, mid, r, i + 1, par_depth_);
#pragma omp taskwait
    } else {
      EvalTree(stl, cws, ys, l, mid, i + 1, par_depth_);
      EvalTree(str, cws, ys, mid, r, i + 1, par_depth_);
    }
  }
};

}  // namespace fss
