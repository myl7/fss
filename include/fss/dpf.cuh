// SPDX-License-Identifier: Apache-2.0
/**
 * @file dpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 *
 * 2-party Distributed Point Function (DPF) from the paper, [Function Secret Sharing: Improvements and Extensions](https://eprint.iacr.org/2018/707).
 *
 * ## References
 *
 * - Elette Boyle, Niv Gilboa, and Yuval Ishai. 2016. Function Secret Sharing: Improvements and Extensions. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16). Association for Computing Machinery, New York, NY, USA, 1292–1303. <https://doi.org/10.1145/2976749.2978429>.
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <fss/group.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>
// #include <omp.h>

namespace fss {

/**
 * DPF scheme.
 *
 * TODO: lambda = 128
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
     * TODO: s, tl, tr
     */
    struct __align__(32) Cw {
        int4 s;
        bool tr;
    };

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
