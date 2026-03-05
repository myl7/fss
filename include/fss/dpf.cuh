// SPDX-License-Identifier: Apache-2.0
/**
 * @file dpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * @brief 2-party distributed point function (DPF).
 *
 * The scheme is from the paper, [_Function Secret Sharing: Improvements and Extensions_](https://eprint.iacr.org/2018/707) (@ref dpf "1: the published version").
 *
 * ## Definitions
 *
 * **Point function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, and $b \in \sG_{out}$, a point function $f_{a, b}$ is a function that for any input $x$, the output $y$ has $y = b$ only when $x = a$, otherwise $y = 0$.
 *
 * **DPF**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, $b \in \sG_{out}$, and a security parameter $\lambda$, 2-party DPF is a scheme consisting of the methods:
 *
 * - Key generation: $Gen(1^\lambda, f_{a, b}) \rightarrow (k_0, k_1)$.
 * - Evaluation: $Eval(k_i, x) \rightarrow y_{i,x}$ for any $i \in \{0, 1\}$ and any $x \in \sG_{in}$.
 *
 * That satisfies:
 *
 * - Correctness: $y_{0, x} + y_{1, x} = b$ only when $x = a$, otherwise $y_{0, x} + y_{1, x} = 0$.
 * - Privacy: Neither $k_0$ nor $k_1$ reveals any information about $a$ or $b$.
 *   Formally speaking, there exists a probabilistic polynomial time (PPT) simulator $Sim$ that can generate output computationally indistinguishable from any strict subset of the keys output by $Gen$.
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
 * 1. Elette Boyle, Niv Gilboa, Yuval Ishai: Function Secret Sharing: Improvements and Extensions. CCS 2016: 1292-1303. <https://doi.org/10.1145/2976749.2978429>. @anchor dpf
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <omp.h>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>

namespace fss {

/**
 * 2-party DPF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Groupable<Group> && Prgable<Prg, 2>)
class Dpf {
public:
    Prg prg;

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
     * Key generation method.
     *
     * @param cws Pre-allocated array of Cw as returns. The array size must be `in_bits + 1`.
     * @param s0s 2 initial seeds. Users can randomly sample them.
     * @param a $a$.
     * @param b_buf $b$. Will be clamped and converted to the group element.
     *
     * The key for party i consists of cws + s0s[i].
     */
    __host__ __device__ void Gen(Cw cws[], const int4 s0s[2], In a, int4 b_buf) {
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

            // s_cw is updated here
            s_cw = util::SetLsb(s_cw, tl_cw);
            cws[i] = {s_cw, tr_cw};
        }

        auto v_cw_np1 = Group::From(b_buf) + (-Group::From(s0)) + Group::From(s1);
        if (t1) v_cw_np1 = -v_cw_np1;
        cws[in_bits] = {v_cw_np1.Into(), false};
    }

    /**
     * Evaluation method.
     *
     * @param b Party index. False for 0 and true for 1. $i$.
     * @param s0 Initial seed of the party.
     * @param cws Returned by Gen().
     * @param x Evaluated input. $x$.
     * @return Output share. $y_{i,x}$.
     */
    __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], In x) {
        int4 s = s0;
        s = util::SetLsb(s, false);
        bool t = b;

        for (int i = 0; i < in_bits; ++i) {
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

            bool x_bit = (x >> (in_bits - 1 - i)) & 1;

            if (!x_bit) {
                s = sl;
                t = tl;
            } else {
                s = sr;
                t = tr;
            }
        }

        auto y = Group::From(s);
        int4 v_cw_np1 = cws[in_bits].s;
        assert((v_cw_np1.w & 1) == 0);
        if (t) y = y + Group::From(v_cw_np1);
        if (b) y = -y;

        return y.Into();
    }

    /**
     * Full domain evaluation method.
     */
    __host__ void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[]) {
        int4 st = s0;
        bool t = b;
        st = util::SetLsb(st, t);

        assert(in_bits < sizeof(size_t) * 8);
        size_t l = 0;
        size_t r = 1ULL << in_bits;
        int i = 0;

        int par_depth = 0;
        int threads = omp_get_max_threads();
        while ((1 << par_depth) < threads) {
            par_depth++;
        }

#pragma omp parallel
#pragma omp single
        EvalTree(b, st, cws, ys, l, r, i, par_depth);
    }

private:
    __host__ void EvalTree(
        bool b, int4 st, const Cw cws[], int4 ys[], size_t l, size_t r, int i, int par_depth) {
        bool t = util::GetLsb(st);
        int4 s = st;
        s = util::SetLsb(s, false);

        if (i == in_bits) {
            auto y = Group::From(s);
            int4 v_cw_np1 = cws[in_bits].s;
            assert((v_cw_np1.w & 1) == 0);
            if (t) y = y + Group::From(v_cw_np1);
            if (b) y = -y;
            assert(l + 1 == r);
            ys[l] = y.Into();
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

        if (i < par_depth) {
#pragma omp task
            EvalTree(b, stl, cws, ys, l, mid, i + 1, par_depth);
#pragma omp task
            EvalTree(b, str, cws, ys, mid, r, i + 1, par_depth);
#pragma omp taskwait
        } else {
            EvalTree(b, stl, cws, ys, l, mid, i + 1, par_depth);
            EvalTree(b, str, cws, ys, mid, r, i + 1, par_depth);
        }
    }
};

}  // namespace fss
