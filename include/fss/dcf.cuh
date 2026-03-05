// SPDX-License-Identifier: Apache-2.0
/**
 * @file dcf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * @brief 2-party distributed comparison function (DCF).
 *
 * The scheme is from the paper, [_Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation_](https://eprint.iacr.org/2020/1392) (@ref dcf "1: the published version").
 *
 * ## Definitions
 *
 * **Comparison function**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, and $b \in \sG_{out}$, a comparison function $f^<_{a, b}$ is a function that for any input $x$, the output $y$ has $y = b$ only when $x < a$, otherwise $y = 0$.
 *
 * **DCF**: for the input domain $\sG_{in} = \{0, 1\}^n$, the output domain $(\sG_{out}, +)$ that is a group, $a \in \sG_{in}$, $b \in \sG_{out}$, and a security parameter $\lambda$, 2-party DCF is a scheme consisting of the methods:
 *
 * - Key generation: $Gen(1^\lambda, f^<_{a, b}) \rightarrow (k_0, k_1)$.
 * - Evaluation: $Eval(k_i, x) \rightarrow y_{i,x}$ for any $i \in \{0, 1\}$ and any $x \in \sG_{in}$.
 *
 * That satisfies:
 *
 * - Correctness: $y_{0, x} + y_{1, x} = b$ only when $x < a$, otherwise $y_{0, x} + y_{1, x} = 0$.
 * - Privacy: Neither $k_0$ nor $k_1$ reveals any information about $a$ or $b$.
 *   Formally speaking, there exists a probabilistic polynomial time (PPT) simulator $Sim$ that can generate output computationally indistinguishable from any strict subset of the keys output by $Gen$.
 *
 * Similarly, we have $f^>_{a, b}$ and DCF for it.
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
 * 1. Elette Boyle, Nishanth Chandran, Niv Gilboa, Divya Gupta, Yuval Ishai, Nishant Kumar, Mayank Rathee: Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation. EUROCRYPT (2) 2021: 871-900. <https://doi.org/10.1007/978-3-030-77886-6_30>. @anchor dcf
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
 * Comparison predicate.
 *
 * For the input domain $\sG_{in} = \{0, 1\}^n$ as bits and $x, a \in \sG_{in}$, $x < a$ is defined as that the unsigned integer represented by $x$ is less than that represented by $a$, i.e., comparison starts from the most significant bit (MSB).
 */
enum class DcfPred {
    kLt, /**< $y = b$ when $x < a$ */
    kGt, /**< $y = b$ when $x > a$ */
};

/**
 * 2-party DCF scheme.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam pred See DcfPred.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint,
    DcfPred pred = DcfPred::kLt, int par_depth = -1>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Groupable<Group> && Prgable<Prg, 4>)
class Dcf {
public:
    Prg prg;

    /**
     * Correction word.
     *
     * ## Layout
     *
     * According to the paper, there are s, v, tl, tr to be stored.
     * v is converted to the clamped 16B to be stored.
     * tl, tr is stored at the clamped bit of s, v, respectively.
     */
    struct __align__(32) Cw {
        int4 s;
        int4 v;
    };
    // For only 1 and aligned memory access on GPU
    static_assert(sizeof(Cw) == 32);

    /**
     * Key generation method.
     *
     * @param cws Pre-allocated array of Cw as returns. The array size must be in_bits + 1.
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
        Group v;
        b_buf = util::SetLsb(b_buf, false);

        for (int i = 0; i < in_bits; ++i) {
            auto [s0l, v0l_buf, s0r, v0r_buf] = prg.Gen(s0);
            auto [s1l, v1l_buf, s1r, v1r_buf] = prg.Gen(s1);

            bool t0l = util::GetLsb(s0l);
            s0l = util::SetLsb(s0l, false);
            v0l_buf = util::SetLsb(v0l_buf, false);
            auto v0l = Group::From(v0l_buf);
            bool t0r = util::GetLsb(s0r);
            s0r = util::SetLsb(s0r, false);
            v0r_buf = util::SetLsb(v0r_buf, false);
            auto v0r = Group::From(v0r_buf);
            bool t1l = util::GetLsb(s1l);
            s1l = util::SetLsb(s1l, false);
            v1l_buf = util::SetLsb(v1l_buf, false);
            auto v1l = Group::From(v1l_buf);
            bool t1r = util::GetLsb(s1r);
            s1r = util::SetLsb(s1r, false);
            v1r_buf = util::SetLsb(v1r_buf, false);
            auto v1r = Group::From(v1r_buf);

            bool a_bit = (a >> in_bits - 1 - i) & 1;

            int4 s_cw;
            if (!a_bit) s_cw = util::Xor(s0r, s1r);
            else s_cw = util::Xor(s0l, s1l);

            Group v_cw = (-v);
            if (!a_bit) {
                v_cw = v_cw + v1r + (-v0r);
                if constexpr (pred == DcfPred::kGt) v_cw = v_cw + Group::From(b_buf);
            } else {
                v_cw = v_cw + v1l + (-v0l);
                if constexpr (pred == DcfPred::kLt) v_cw = v_cw + Group::From(b_buf);
            }
            if (t1) v_cw = -v_cw;

            if (!a_bit) v = v + (-v1l) + v0l;
            else v = v + (-v1r) + v0r;
            if (t1) v = v + (-v_cw);
            else v = v + v_cw;

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
            int4 v_buf = v_cw.Into();
            v_buf = util::SetLsb(v_buf, tr_cw);
            cws[i] = {s_cw, v_buf};
        }

        auto v_cw_np1 = Group::From(s1) + (-Group::From(s0)) + (-v);
        if (t1) v_cw_np1 = -v_cw_np1;
        cws[in_bits] = {{0, 0, 0, 0}, v_cw_np1.Into()};
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
        Group v;
        bool t = b;

        for (int i = 0; i < in_bits; ++i) {
            auto cw = cws[i];

            int4 s_cw = cw.s;
            bool tl_cw = util::GetLsb(s_cw);
            s_cw = util::SetLsb(s_cw, false);

            int4 v_cw_buf = cw.v;
            bool tr_cw = util::GetLsb(v_cw_buf);
            v_cw_buf = util::SetLsb(v_cw_buf, false);
            auto v_cw = Group::From(v_cw_buf);

            auto [sl, vl_buf, sr, vr_buf] = prg.Gen(s);

            bool tl = util::GetLsb(sl);
            sl = util::SetLsb(sl, false);
            vl_buf = util::SetLsb(vl_buf, false);
            auto vl = Group::From(vl_buf);

            bool tr = util::GetLsb(sr);
            sr = util::SetLsb(sr, false);
            vr_buf = util::SetLsb(vr_buf, false);
            auto vr = Group::From(vr_buf);

            if (t) {
                sl = util::Xor(sl, s_cw);
                sr = util::Xor(sr, s_cw);
                tl = tl ^ tl_cw;
                tr = tr ^ tr_cw;
            }

            bool x_bit = (x >> in_bits - 1 - i) & 1;

            if (b) {
                if (!x_bit) v = v + (-vl);
                else v = v + (-vr);
                if (t) v = v + (-v_cw);
            } else {
                if (!x_bit) v = v + vl;
                else v = v + vr;
                if (t) v = v + v_cw;
            }

            if (!x_bit) {
                s = sl;
                t = tl;
            } else {
                s = sr;
                t = tr;
            }
        }

        int4 v_cw_np1_buf = cws[in_bits].v;
        assert((v_cw_np1_buf.w & 1) == 0);
        auto v_cw_np1 = Group::From(v_cw_np1_buf);

        if (b) {
            v = v + (-Group::From(s));
            if (t) v = v + (-v_cw_np1);
        } else {
            v = v + Group::From(s);
            if (t) v = v + v_cw_np1;
        }

        return v.Into();
    }

    /**
     * Full domain evaluation method.
     *
     * Evaluate the key on each input, i.e., 0b00...0 - 0b11...1.
     * Store the outputs sequentially.
     *
     * b, s0, cws are the same as the ones in Eval().
     *
     * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
     *
     * Support parallel using OpenMP.
     * The task is divided to 2 ** par_depth parallel sub-tasks with the equal workloads.
     * par_depth = -1: use ceil(log(num of threads)).
     * par_depth = 0: no parallelism, i.e., sequential execution.
     */
    void EvalAll(bool b, int4 s0, const Cw cws[], int4 ys[]) {
        int4 st = s0;
        bool t = b;
        st = util::SetLsb(st, t);

        assert(in_bits < sizeof(size_t) * 8);
        size_t l = 0;
        size_t r = 1ULL << in_bits;
        int i = 0;

        int par_depth_ = 0;
        if (par_depth == -1) {
            int threads = omp_get_max_threads();
            while ((1 << par_depth_) < threads) {
                par_depth_++;
            }
        } else par_depth_ = par_depth;

        Group v;

#pragma omp parallel
#pragma omp single
        EvalTree(b, st, cws, ys, l, r, i, par_depth_, v);
    }

private:
    void EvalTree(bool b, int4 st, const Cw cws[], int4 ys[], size_t l, size_t r, int i,
        int par_depth_, Group v) {
        bool t = util::GetLsb(st);
        int4 s = st;
        s = util::SetLsb(s, false);

        if (i == in_bits) {
            int4 v_cw_np1_buf = cws[in_bits].v;
            assert((v_cw_np1_buf.w & 1) == 0);
            auto term = Group::From(s);
            if (t) term = term + Group::From(v_cw_np1_buf);
            if (b) term = -term;
            v = v + term;
            assert(l + 1 == r);
            ys[l] = v.Into();
            return;
        }

        Cw cw = cws[i];
        int4 s_cw = cw.s;
        bool tl_cw = util::GetLsb(s_cw);
        s_cw = util::SetLsb(s_cw, false);
        int4 v_cw_buf = cw.v;
        bool tr_cw = util::GetLsb(v_cw_buf);
        v_cw_buf = util::SetLsb(v_cw_buf, false);
        auto v_cw = Group::From(v_cw_buf);

        auto [sl, vl_buf, sr, vr_buf] = prg.Gen(s);

        bool tl = util::GetLsb(sl);
        sl = util::SetLsb(sl, false);
        vl_buf = util::SetLsb(vl_buf, false);
        auto vl = Group::From(vl_buf);

        bool tr = util::GetLsb(sr);
        sr = util::SetLsb(sr, false);
        vr_buf = util::SetLsb(vr_buf, false);
        auto vr = Group::From(vr_buf);

        if (t) {
            sl = util::Xor(sl, s_cw);
            sr = util::Xor(sr, s_cw);
            tl = tl ^ tl_cw;
            tr = tr ^ tr_cw;
            vl = vl + v_cw;
            vr = vr + v_cw;
        }
        if (b) {
            vl = -vl;
            vr = -vr;
        }

        vl = vl + v;
        vr = vr + v;

        int4 stl = sl;
        stl = util::SetLsb(stl, tl);
        int4 str = sr;
        str = util::SetLsb(str, tr);

        size_t mid = (l + r) / 2;

        if (i < par_depth_) {
#pragma omp task
            EvalTree(b, stl, cws, ys, l, mid, i + 1, par_depth_, vl);
#pragma omp task
            EvalTree(b, str, cws, ys, mid, r, i + 1, par_depth_, vr);
#pragma omp taskwait
        } else {
            EvalTree(b, stl, cws, ys, l, mid, i + 1, par_depth_, vl);
            EvalTree(b, str, cws, ys, mid, r, i + 1, par_depth_, vr);
        }
    }
};

}  // namespace fss
