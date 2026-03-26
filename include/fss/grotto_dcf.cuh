// SPDX-License-Identifier: Apache-2.0
/**
 * @file grotto_dcf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed comparison function (DCF) over F2 from standard DPF.
 *
 * The scheme is from the paper, [_Grotto: Screaming fast (2+1)-PC for Z_{2^n} via
 * (2,2)-DPFs_](https://eprint.iacr.org/2023/108) (@ref grotto_dcf "1: the published version").
 *
 * Key generation is identical to standard BGI DPF. The comparison functionality
 * emerges from prefix-parity of the DPF control bits.
 *
 * Output shares are in F2 (XOR sharing). Each party holds a single bool per query.
 * For inputs x: share_0 XOR share_1 = 1[alpha <= x].
 *
 * ## References
 *
 * 1. Kyle Storrier, Adithya Vadapalli, Allan Lyons, Ryan Henry: Grotto: Screaming fast (2+1)-PC for Z_{2^n} via (2, 2)-DPFs. CCS 2023: 2143-2157. <https://doi.org/10.1145/3576915.3623147>. <https://eprint.iacr.org/2023/108>. @anchor grotto_dcf
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <omp.h>
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>

namespace fss {

/**
 * 2-party DCF scheme over F2 from standard DPF (Grotto construction).
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Prg See Prgable. Must satisfy Prgable<Prg, 2> (same as DPF).
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth -1 is to use ceil(log(num of threads)).
 * Only Preprocess() and EvalAll() use it.
 */
template <int in_bits, typename Prg, typename In = uint, int par_depth = -1>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Prgable<Prg, 2>)
class GrottoDcf {
    using DpfType = Dpf<in_bits, group::Bytes, Prg, In, par_depth>;

public:
    using Cw = typename DpfType::Cw;
    Prg prg;

    /**
     * Key generation method. Delegates to Dpf::Gen with beta=0.
     *
     * @param cws Pre-allocated array of Cw. Size must be in_bits + 1.
     * @param s0s 2 initial seeds. Users can randomly sample them.
     * @param a The secret comparison threshold.
     *
     * The key for party i consists of cws + s0s[i].
     */
    __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a) {
        DpfType dpf{prg};
        int4 beta = {0, 0, 0, 0};
        dpf.Gen(cws, s0s, a, beta);
    }

    /**
     * Parity segment tree over leaf control bits.
     *
     * p[0..2N-2]: level-order binary tree where N = 2^in_bits.
     * Root is p[0]. Leaf x has index p[x + N - 1].
     * Internal node j: p[j] = p[2j+1] XOR p[2j+2].
     *
     * b: party index, needed for reconstructing comparison results.
     */
    struct ParityTree {
        bool *p;
        bool b;
    };

    /**
     * Preprocess: expand DPF tree and build parity segment tree.
     *
     * Phase 1: O(N) PRG calls to expand the tree and extract leaf control bits.
     * Phase 2a: O(N) XOR operations to build the parity segment tree bottom-up.
     *
     * @param pt ParityTree with p pre-allocated to size 2*N-1 where N = 2^in_bits.
     *           pt.b must be set to the party index before calling.
     * @param s0 Initial seed of the party.
     * @param cws Correction words from Gen().
     */
    void Preprocess(ParityTree &pt, int4 s0, const Cw cws[]) {
        constexpr size_t N = 1ULL << in_bits;

        // Phase 1: expand tree, write leaf control bits to pt.p[N-1 .. 2N-2]
        ExpandTree(pt.b, s0, cws, pt.p + (N - 1));

        // Phase 2a: build parity segment tree bottom-up
        for (size_t j = N - 2; j < N - 1; --j) {
            pt.p[j] = pt.p[2 * j + 1] ^ pt.p[2 * j + 2];
        }
    }

    /**
     * Prefix-parity query on the parity segment tree.
     *
     * Returns party b's share of 1[alpha <= x].
     * Internally queries endpoint e = x + 1, computing prefix-parity of [0, e).
     *
     * @param pt ParityTree from Preprocess().
     * @param x Query point.
     * @return bool share such that share_0 XOR share_1 = 1[alpha <= x].
     */
    __host__ __device__ static bool Eval(const ParityTree &pt, In x) {
        constexpr size_t N = 1ULL << in_bits;
        In e = static_cast<In>(x) + 1;

        // e == 0 means x + 1 overflowed, i.e., e = N (entire domain)
        if (e == 0 || e == N) return pt.p[0];

        bool pi = false;
        size_t cur = 0;
        for (int i = 0; i < in_bits; ++i) {
            bool e_bit = (e >> (in_bits - 1 - i)) & 1;
            if (e_bit) {
                pi ^= pt.p[2 * cur + 1];
                cur = 2 * cur + 2;
            } else {
                cur = 2 * cur + 1;
            }
        }
        return pi;
    }

    /**
     * Full domain evaluation.
     *
     * Computes party b's share of 1[alpha <= x] for all x in [0, N).
     *
     * Phase 1: O(N) PRG calls to expand the tree.
     * Phase 2b: O(N) prefix-sum (running XOR) over leaf control bits.
     *
     * @param b Party index.
     * @param s0 Initial seed of the party.
     * @param cws Correction words from Gen().
     * @param ys Pre-allocated output array of size N = 2^in_bits.
     *           ys[x] = party b's share of 1[alpha <= x].
     */
    void EvalAll(bool b, int4 s0, const Cw cws[], bool ys[]) {
        constexpr size_t N = 1ULL << in_bits;

        // Phase 1: expand tree to get leaf control bits into ys[]
        ExpandTree(b, s0, cws, ys);

        // Phase 2b: prefix-sum scan (running XOR)
        // ys[x] currently holds leaf x's control bit.
        // Transform to: ys[x] = XOR of control bits [0..x] = share of 1[alpha <= x].
        for (size_t x = 1; x < N; ++x) {
            ys[x] = ys[x] ^ ys[x - 1];
        }
    }

private:
    /**
     * Expand the DPF tree and write leaf control bits.
     *
     * @param b Party index.
     * @param s0 Initial seed of the party.
     * @param cws Correction words from Gen().
     * @param t Output array of size N = 2^in_bits for leaf control bits.
     */
    void ExpandTree(bool b, int4 s0, const Cw cws[], bool t[]) {
        int4 st = s0;
        st = util::SetLsb(st, b);

        assert(in_bits < sizeof(size_t) * 8);
        size_t l = 0;
        size_t r = 1ULL << in_bits;
        int i = 0;

        int par_depth_ = util::ResolveParDepth(par_depth);

#pragma omp parallel
#pragma omp single
        ExpandTreeRec(st, cws, t, l, r, i, par_depth_);
    }

    void ExpandTreeRec(
        int4 st, const Cw cws[], bool t[], size_t l, size_t r, int i, int par_depth_) {
        bool tc = util::GetLsb(st);
        int4 s = st;
        s = util::SetLsb(s, false);

        if (i == in_bits) {
            assert(l + 1 == r);
            t[l] = tc;
            return;
        }

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

        if (tc) {
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
            ExpandTreeRec(stl, cws, t, l, mid, i + 1, par_depth_);
#pragma omp task
            ExpandTreeRec(str, cws, t, mid, r, i + 1, par_depth_);
#pragma omp taskwait
        } else {
            ExpandTreeRec(stl, cws, t, l, mid, i + 1, par_depth_);
            ExpandTreeRec(str, cws, t, mid, r, i + 1, par_depth_);
        }
    }
};

}  // namespace fss
