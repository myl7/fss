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
 * 1. Elette Boyle, Niv Gilboa, and Yuval Ishai. 2016. Function Secret Sharing: Improvements and Extensions. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16). Association for Computing Machinery, New York, NY, USA, 1292–1303. <https://doi.org/10.1145/2976749.2978429>. @anchor dpf
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cassert>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>
// #include <omp.h>

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

    // private:
    //     // Recursive helper passing s and t by value/register
    //     static void eval_full_domain_subtree(int depth, int4 *sbuf, size_t l, size_t r, uint8_t b,
    //         Key k, int x_bitlen, int par_depth, int4 s, uint8_t t) {
    //         if (depth == x_bitlen) {
    //             // Leaf
    //             Group g_s = *reinterpret_cast<Group *>(&s);
    //             Group cw_np1 = *reinterpret_cast<Group *>(k.cw_np1);

    //             if (t) g_s = g_s + cw_np1;
    //             if (b) g_s = -g_s;

    //             // Assume sbuf is array of Group (cast to int4 for API)
    //             *reinterpret_cast<Group *>(&sbuf[l]) = g_s;
    //             return;
    //         }

    //         const uint8_t *cw = k.cws + depth * (kLambda + 1);
    //         int4 s_cw = *reinterpret_cast<const int4 *>(cw);
    //         // get_cwt
    //         uint8_t tl_cw = cw[kLambda] >> 1;
    //         uint8_t tr_cw = cw[kLambda] & 1;

    //         int4 ss[2];  // sl, sr
    //         uint8_t tl, tr;

    //         Prg::gen(
    //             reinterpret_cast<uint8_t *>(ss), 2 * kLambda, reinterpret_cast<const uint8_t *>(&s));

    //         // load_sst: extract t from MSB and clear control bits
    //         uint8_t *ss0_bytes = reinterpret_cast<uint8_t *>(&ss[0]);
    //         uint8_t *ss1_bytes = reinterpret_cast<uint8_t *>(&ss[1]);
    //         tl = ss0_bytes[15] >> 7;
    //         ss0_bytes[15] &= 0x7F;
    //         ss0_bytes[8] &= 0xFE;
    //         tr = ss1_bytes[15] >> 7;
    //         ss1_bytes[15] &= 0x7F;
    //         ss1_bytes[8] &= 0xFE;

    //         if (t) {
    //             ss[0] = fss::util::Xor(ss[0], s_cw);
    //             ss[1] = fss::util::Xor(ss[1], s_cw);
    //             tl ^= tl_cw;
    //             tr ^= tr_cw;
    //         }

    //         size_t mid = (l + r) / 2;

    //         if (depth < par_depth) {
    // #pragma omp parallel
    // #pragma omp single
    //             {
    // #pragma omp task
    //                 {
    //                     eval_full_domain_subtree(
    //                         depth + 1, sbuf, l, mid, b, k, x_bitlen, par_depth, ss[0], tl);
    //                 }
    // #pragma omp task
    //                 {
    //                     eval_full_domain_subtree(
    //                         depth + 1, sbuf, mid, r, b, k, x_bitlen, par_depth, ss[1], tr);
    //                 }
    // #pragma omp taskwait
    //             }
    //         } else {
    //             eval_full_domain_subtree(depth + 1, sbuf, l, mid, b, k, x_bitlen, par_depth, ss[0], tl);
    //             eval_full_domain_subtree(depth + 1, sbuf, mid, r, b, k, x_bitlen, par_depth, ss[1], tr);
    //         }
    //     }

    // public:
    //     static void eval_full_domain(int4 *sbuf, uint8_t b, Key k, int x_bitlen) {
    //         int4 s = sbuf[0];
    //         uint8_t *s_bytes = reinterpret_cast<uint8_t *>(&s);

    //         // load_st: extract t from MSB and clear control bits
    //         uint8_t t = s_bytes[15] >> 7;
    //         s_bytes[15] &= 0x7F;
    //         s_bytes[8] &= 0xFE;
    //         t = b;

    //         int threads = omp_get_max_threads();
    //         int par_depth = 0;
    //         while ((1 << par_depth) <= threads) {
    //             par_depth++;
    //         }
    //         par_depth--;

    //         size_t sbuf_len = 1ULL << x_bitlen;
    //         eval_full_domain_subtree(0, sbuf, 0, sbuf_len, b, k, x_bitlen, par_depth, s, t);
    //     }
};

}  // namespace fss
