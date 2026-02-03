// SPDX-License-Identifier: Apache-2.0
/**
 * @file dcf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * 2-party Distributed Comparison Function (DCF) from the paper, [Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation](https://eprint.iacr.org/2020/1392).
 *
 * ## References
 *
 * - Boyle, E. et al. (2021). Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation. In: Canteaut, A., Standaert, FX. (eds) Advances in Cryptology – EUROCRYPT 2021. EUROCRYPT 2021. Lecture Notes in Computer Science(), vol 12697. Springer, Cham. <https://doi.org/10.1007/978-3-030-77886-6_30>.
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
 * Comparison predicate.
 *
 * TODO: Definition for comparison of the input domain
 */
enum class DcfPred {
    kLt, /**< \f$y = b\f$ when \f$x < a\f$ */
    kGt, /**< \f$y = b\f$ when \f$x > a\f$ */
};

/**
 * DCF scheme.
 *
 * TODO: lambda = 128
 */
template <int in_bits, typename Group, typename Prg, typename In = uint,
    DcfPred pred = DcfPred::kLt>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Groupable<Group> && Prgable<Prg, 4>)
class Dcf {
public:
    Prg prg;

    /**
     * Correction word.
     *
     * TODO: s, v, tl, tr
     */
    struct __align__(32) Cw {
        int4 s;
        int4 v;
    };
    static_assert(sizeof(Cw) == 32, "accessing a Cw results in more than 1 32B transaction");

    /**
     * Key generation method.
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

    // private:
    //     // Recursive helper
    //     static void eval_full_domain_subtree(int depth, int4 *sbuf, size_t l, size_t r, uint8_t b,
    //         Key k, int x_bitlen, int par_depth, int4 s, uint8_t t, Group v) {
    //         if (depth == x_bitlen) {
    //             Group g_s = *reinterpret_cast<Group *>(&s);
    //             Group term = g_s;
    //             if (t) term = term + *reinterpret_cast<Group *>(k.v_cw_np1);
    //             if (b) term = -term;
    //             v = v + term;
    //             *reinterpret_cast<Group *>(&sbuf[l]) = v;
    //             return;
    //         }

    //         const uint8_t *cw = k.cws + depth * (kLambda * 2 + 1);
    //         int4 s_cw = *reinterpret_cast<const int4 *>(cw);
    //         Group v_cw = *reinterpret_cast<const Group *>(cw + kLambda);
    //         uint8_t tl_cw, tr_cw;
    //         // get_cwt
    //         tl_cw = cw[kLambda * 2] >> 1;
    //         tr_cw = cw[kLambda * 2] & 1;

    //         Block svs[2];

    //         Prg::gen(
    //             reinterpret_cast<uint8_t *>(svs), 4 * kLambda, reinterpret_cast<const uint8_t *>(&s));

    //         uint8_t tl, tr;
    //         // load_svst(svs, &tl, &tr);
    //         {
    //             // svs[0]
    //             uint8_t *s_bytes = reinterpret_cast<uint8_t *>(&svs[0].s);
    //             tl = s_bytes[15] >> 7;
    //             s_bytes[15] &= 0x7F;
    //             s_bytes[8] &= 0xFE;

    //             uint8_t *v_bytes = reinterpret_cast<uint8_t *>(&svs[0].v);
    //             v_bytes[15] &= 0x7F;
    //             v_bytes[8] &= 0xFE;

    //             // svs[1]
    //             s_bytes = reinterpret_cast<uint8_t *>(&svs[1].s);
    //             tr = s_bytes[15] >> 7;
    //             s_bytes[15] &= 0x7F;
    //             s_bytes[8] &= 0xFE;

    //             v_bytes = reinterpret_cast<uint8_t *>(&svs[1].v);
    //             v_bytes[15] &= 0x7F;
    //             v_bytes[8] &= 0xFE;
    //         }

    //         if (t) {
    //             svs[0].s = util::Xor(svs[0].s, s_cw);
    //             svs[1].s = util::Xor(svs[1].s, s_cw);
    //             tl ^= tl_cw;
    //             tr ^= tr_cw;
    //         }

    //         Group vl = svs[0].v;
    //         Group vr = svs[1].v;

    //         if (t) {
    //             vl = vl + v_cw;
    //             vr = vr + v_cw;
    //         }
    //         if (b) {
    //             vl = -vl;
    //             vr = -vr;
    //         }

    //         vl = vl + v;
    //         vr = vr + v;

    //         size_t mid = (l + r) / 2;

    //         if (depth < par_depth) {
    // #pragma omp parallel
    // #pragma omp single
    //             {
    // #pragma omp task
    //                 {
    //                     eval_full_domain_subtree(
    //                         depth + 1, sbuf, l, mid, b, k, x_bitlen, par_depth, svs[0].s, tl, vl);
    //                 }
    // #pragma omp task
    //                 {
    //                     eval_full_domain_subtree(
    //                         depth + 1, sbuf, mid, r, b, k, x_bitlen, par_depth, svs[1].s, tr, vr);
    //                 }
    // #pragma omp taskwait
    //             }
    //         } else {
    //             eval_full_domain_subtree(
    //                 depth + 1, sbuf, l, mid, b, k, x_bitlen, par_depth, svs[0].s, tl, vl);
    //             eval_full_domain_subtree(
    //                 depth + 1, sbuf, mid, r, b, k, x_bitlen, par_depth, svs[1].s, tr, vr);
    //         }
    //     }

    // public:
    //     static void eval_full_domain(int4 *sbuf, uint8_t b, Key k, int x_bitlen) {
    //         int4 s = sbuf[0];
    //         uint8_t t = b;
    //         Group v = Group();
    //         // load_st(&s, &t); t=b;
    //         {
    //             uint8_t *s_bytes = reinterpret_cast<uint8_t *>(&s);
    //             t = s_bytes[15] >> 7;
    //             s_bytes[15] &= 0x7F;
    //             s_bytes[8] &= 0xFE;
    //         }
    //         t = b;

    //         int threads = omp_get_max_threads();
    //         int par_depth = 0;
    //         while ((1 << par_depth) <= threads) {
    //             par_depth++;
    //         }
    //         par_depth--;

    //         size_t sbuf_len = 1ULL << x_bitlen;
    //         eval_full_domain_subtree(0, sbuf, 0, sbuf_len, b, k, x_bitlen, par_depth, s, t, v);
    //     }
};

}  // namespace fss
