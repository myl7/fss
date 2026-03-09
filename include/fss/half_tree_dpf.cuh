// SPDX-License-Identifier: Apache-2.0
/**
 * @file half_tree_dpf.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief 2-party distributed point function (DPF) using the Half-Tree scheme.
 *
 * The scheme is from the paper, [_Half-Tree: Halving the Cost of Tree Expansion in COT and DPF_](https://eprint.iacr.org/2023/1044).
 * It reduces the number of hash calls from 2N to 1.5N for full-domain evaluation.
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
 * 2-party DPF scheme using the Half-Tree construction.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=1 (CCR hash, 128->128 bits).
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @tparam par_depth -1 is to use ceil(log(num of threads)), which should be good enough.
 * Only EvalAll() uses it. See EvalAll() for details.
 */
template <int in_bits, typename Group, typename Prg, typename In = uint, int par_depth = -1>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Groupable<Group> && Prgable<Prg, 1>)
class HalfTreeDpf {
public:
    Prg prg;
    int4 hash_key;

    /**
     * Correction word.
     *
     * For levels 1..n-1, s is the CW and extra is unused (false).
     * For level n, s stores SetLsb(HCW, LCW_0) and extra stores LCW_1.
     */
    struct __align__(32) Cw {
        int4 s;
        bool extra;
    };
    static_assert(sizeof(Cw) == 32);

    /**
     * Key generation method.
     *
     * @param cws Pre-allocated array of Cw as returns. The array size must be in_bits.
     * @param ocw Output correction word (for the Convert step).
     * @param s0s 2 initial seeds. Users can randomly sample them.
     * @param a The special point alpha.
     * @param b_buf The nonzero output value beta. Will be clamped and converted to the group element.
     */
    __host__ __device__ void Gen(Cw cws[], int4 &ocw, const int4 s0s[2], In a, int4 b_buf) {
        b_buf = util::SetLsb(b_buf, false);

        // Initialize: node0 has t=0, node1 has t=1
        int4 node0 = util::SetLsb(s0s[0], false);
        int4 node1 = util::SetLsb(s0s[1], true);
        int4 delta = util::Xor(node0, node1);  // LSB = 0^1 = 1

        // Levels 1 to n-1 (index i = 0 to in_bits-2)
        for (int i = 0; i < in_bits - 1; ++i) {
            int4 h0 = prg.Gen(util::Xor(hash_key, node0))[0];
            int4 h1 = prg.Gen(util::Xor(hash_key, node1))[0];

            bool a_bit = (a >> (in_bits - 1 - i)) & 1;

            // CW = h0 ^ h1 ^ (!a_bit ? delta : 0)
            // When a_bit=0 (go left): non-alpha is right, CW = h0^h1^delta makes right0=right1
            // When a_bit=1 (go right): non-alpha is left, CW = h0^h1 makes left0=left1
            int4 cw = util::Xor(h0, h1);
            if (!a_bit) cw = util::Xor(cw, delta);

            cws[i] = {cw, false};

            bool t0 = util::GetLsb(node0);
            bool t1 = util::GetLsb(node1);

            // node_b = h_b ^ (a_bit ? node_b : 0) ^ (t_b ? cw : 0)
            int4 zero4 = {0, 0, 0, 0};
            int4 ab_mask0 = a_bit ? node0 : zero4;
            int4 ab_mask1 = a_bit ? node1 : zero4;
            int4 t0_mask = t0 ? cw : zero4;
            int4 t1_mask = t1 ? cw : zero4;

            node0 = util::Xor(util::Xor(h0, ab_mask0), t0_mask);
            node1 = util::Xor(util::Xor(h1, ab_mask1), t1_mask);

            // delta = node0 ^ node1 for next level
            delta = util::Xor(node0, node1);
        }

        // Level n (last level, index i = in_bits-1)
        {
            bool a_n = (a >> 0) & 1;  // last bit of alpha
            bool t0 = util::GetLsb(node0);
            bool t1 = util::GetLsb(node1);

            // Hash with sigma in {0, 1}
            int4 h0_0 = prg.Gen(util::Xor(hash_key, util::SetLsb(node0, false)))[0];
            int4 h0_1 = prg.Gen(util::Xor(hash_key, util::SetLsb(node0, true)))[0];
            int4 h1_0 = prg.Gen(util::Xor(hash_key, util::SetLsb(node1, false)))[0];
            int4 h1_1 = prg.Gen(util::Xor(hash_key, util::SetLsb(node1, true)))[0];

            // Extract high (s) and low (t) parts
            int4 high0_0 = util::SetLsb(h0_0, false);
            bool low0_0 = util::GetLsb(h0_0);
            int4 high0_1 = util::SetLsb(h0_1, false);
            bool low0_1 = util::GetLsb(h0_1);
            int4 high1_0 = util::SetLsb(h1_0, false);
            bool low1_0 = util::GetLsb(h1_0);
            int4 high1_1 = util::SetLsb(h1_1, false);
            bool low1_1 = util::GetLsb(h1_1);

            // HCW corrects the non-alpha direction so both parties converge.
            // HCW = high{!a_n}_0 ^ high{!a_n}_1
            int4 HCW;
            if (a_n) HCW = util::Xor(high0_0, high1_0);
            else HCW = util::Xor(high0_1, high1_1);

            // LCW ensures:
            //   Alpha direction (sigma=a_n): low0 ^ low1 = 1 (exactly one adds ocw)
            //   Non-alpha direction (sigma=!a_n): low0 ^ low1 = 0 (cancels)
            // LCW_0 = low{0}_0 ^ low{0}_1 ^ !a_n
            // LCW_1 = low{1}_0 ^ low{1}_1 ^ a_n
            bool LCW_0 = low0_0 ^ low1_0 ^ !a_n;
            bool LCW_1 = low0_1 ^ low1_1 ^ a_n;

            // Store CW_n
            cws[in_bits - 1] = {util::SetLsb(HCW, LCW_0), LCW_1};

            // Compute leaf for each party
            // leaf_b = (a_n ? high{1}_b||low{1}_b : high{0}_b||low{0}_b)
            int4 leaf0, leaf1;
            if (a_n) {
                leaf0 = util::SetLsb(high0_1, low0_1);
                leaf1 = util::SetLsb(high1_1, low1_1);
            } else {
                leaf0 = util::SetLsb(high0_0, low0_0);
                leaf1 = util::SetLsb(high1_0, low1_0);
            }

            // Apply CW correction: if t_b: leaf_b ^= SetLsb(HCW, lcw_an)
            bool lcw_an = a_n ? LCW_1 : LCW_0;
            int4 leaf_cw = util::SetLsb(HCW, lcw_an);
            if (t0) leaf0 = util::Xor(leaf0, leaf_cw);
            if (t1) leaf1 = util::Xor(leaf1, leaf_cw);

            // Output CW: v_cw = Group::From(b_buf) + (-Group::From(SetLsb(leaf0,false))) + Group::From(SetLsb(leaf1,false))
            auto v_cw = Group::From(b_buf) + (-Group::From(util::SetLsb(leaf0, false))) +
                Group::From(util::SetLsb(leaf1, false));
            if (util::GetLsb(leaf1)) v_cw = -v_cw;
            ocw = v_cw.Into();
        }
    }

    /**
     * Evaluation method.
     *
     * @param b Party index. False for 0 and true for 1.
     * @param s0 Initial seed of the party.
     * @param cws Returned by Gen().
     * @param ocw Output correction word returned by Gen().
     * @param x Evaluated input.
     * @return Output share.
     */
    __host__ __device__ int4 Eval(bool b, int4 s0, const Cw cws[], int4 ocw, In x) {
        int4 node = util::SetLsb(s0, b);

        // Levels 1 to n-1 (index i = 0 to in_bits-2)
        for (int i = 0; i < in_bits - 1; ++i) {
            bool x_bit = (x >> (in_bits - 1 - i)) & 1;
            bool t = util::GetLsb(node);

            int4 h = prg.Gen(util::Xor(hash_key, node))[0];

            int4 zero4 = {0, 0, 0, 0};
            int4 xb_mask = x_bit ? node : zero4;
            int4 t_mask = t ? cws[i].s : zero4;

            node = util::Xor(util::Xor(h, xb_mask), t_mask);
        }

        // Level n (last level)
        {
            bool x_n = (x >> 0) & 1;
            bool t = util::GetLsb(node);

            int4 h = prg.Gen(util::Xor(hash_key, util::SetLsb(node, x_n)))[0];

            // Unpack CW_n
            int4 hcw = util::SetLsb(cws[in_bits - 1].s, false);
            bool lcw_xn;
            if (x_n) lcw_xn = cws[in_bits - 1].extra;
            else lcw_xn = util::GetLsb(cws[in_bits - 1].s);

            int4 high = util::SetLsb(h, false);
            bool low = util::GetLsb(h);

            if (t) {
                high = util::Xor(high, hcw);
                low = low ^ lcw_xn;
            }

            auto y = Group::From(high);
            if (low) y = y + Group::From(ocw);
            if (b) y = -y;

            return y.Into();
        }
    }

    /**
     * Full domain evaluation method.
     *
     * Evaluate the key on each input, i.e., 0b00...0 - 0b11...1.
     *
     * @param ys Pre-allocated output array. Its size must be at least 2 ** in_bits.
     *
     * Support parallel using OpenMP.
     */
    void EvalAll(bool b, int4 s0, const Cw cws[], int4 ocw, int4 ys[]) {
        int4 node = util::SetLsb(s0, b);

        assert(in_bits < sizeof(size_t) * 8);

        int par_depth_ = 0;
        if (par_depth == -1) {
            int threads = omp_get_max_threads();
            while ((1 << par_depth_) < threads) {
                par_depth_++;
            }
        } else par_depth_ = par_depth;

        if constexpr (in_bits == 1) {
            // Only level n (last level), no tree traversal
#pragma omp parallel
#pragma omp single
            EvalLastLevel(b, node, cws, ocw, ys);
            return;
        }

        // Phase 1: tree traversal for levels 1..n-1, stores nodes at level n-1
        // We use ys[] as scratch space for intermediate nodes.
        // After phase 1, ys[0..2^(in_bits-1)-1] hold the level n-1 nodes (packed s||t).
        size_t num_leaves = 1ULL << (in_bits - 1);

        // Recursive tree traversal
#pragma omp parallel
#pragma omp single
        EvalTree(node, cws, ys, 0, num_leaves, 0, par_depth_);

        // Phase 2: level n + output conversion
        // Unpack CW_n
        int4 hcw = util::SetLsb(cws[in_bits - 1].s, false);
        bool lcw_0 = util::GetLsb(cws[in_bits - 1].s);
        bool lcw_1 = cws[in_bits - 1].extra;

        // Iterate backward to avoid overwriting unprocessed parent nodes.
        // ys[j] holds parent, writes go to ys[2*j] and ys[2*j+1].
        // Processing j before j-1 ensures no read-after-write conflict.
        for (size_t j = num_leaves; j-- > 0;) {
            int4 parent = ys[j];
            bool t_parent = util::GetLsb(parent);

            int4 h0 = prg.Gen(util::Xor(hash_key, util::SetLsb(parent, false)))[0];
            int4 h1 = prg.Gen(util::Xor(hash_key, util::SetLsb(parent, true)))[0];

            int4 high0 = util::SetLsb(h0, false);
            bool low0 = util::GetLsb(h0);
            int4 high1 = util::SetLsb(h1, false);
            bool low1 = util::GetLsb(h1);

            if (t_parent) {
                high0 = util::Xor(high0, hcw);
                low0 = low0 ^ lcw_0;
                high1 = util::Xor(high1, hcw);
                low1 = low1 ^ lcw_1;
            }

            // Output: (-1)^b * (ConvertG(s) + t * ocw) for both children
            auto y0 = Group::From(high0);
            if (low0) y0 = y0 + Group::From(ocw);
            if (b) y0 = -y0;

            auto y1 = Group::From(high1);
            if (low1) y1 = y1 + Group::From(ocw);
            if (b) y1 = -y1;

            ys[2 * j] = y0.Into();
            ys[2 * j + 1] = y1.Into();
        }
    }

private:
    void EvalTree(int4 node, const Cw cws[], int4 ys[], size_t l, size_t r, int i, int par_depth_) {
        // i is the level index (0-based), we traverse levels 0..in_bits-2
        // At level in_bits-1, we store the node
        if (i == in_bits - 1) {
            assert(l + 1 == r);
            ys[l] = node;
            return;
        }

        bool t = util::GetLsb(node);
        int4 h = prg.Gen(util::Xor(hash_key, node))[0];

        int4 zero4 = {0, 0, 0, 0};
        int4 t_mask = t ? cws[i].s : zero4;

        // Left child: left = H_S(parent) ^ (t ? cw : 0)
        int4 left = util::Xor(h, t_mask);
        // Right child: right = left ^ parent
        int4 right = util::Xor(left, node);

        size_t mid = (l + r) / 2;

        if (i < par_depth_) {
#pragma omp task
            EvalTree(left, cws, ys, l, mid, i + 1, par_depth_);
#pragma omp task
            EvalTree(right, cws, ys, mid, r, i + 1, par_depth_);
#pragma omp taskwait
        } else {
            EvalTree(left, cws, ys, l, mid, i + 1, par_depth_);
            EvalTree(right, cws, ys, mid, r, i + 1, par_depth_);
        }
    }

    void EvalLastLevel(bool b, int4 node, const Cw cws[], int4 ocw, int4 ys[]) {
        // For in_bits == 1, the initial node goes directly to level n processing
        bool t_parent = util::GetLsb(node);

        int4 hcw = util::SetLsb(cws[0].s, false);
        bool lcw_0 = util::GetLsb(cws[0].s);
        bool lcw_1 = cws[0].extra;

        int4 h0 = prg.Gen(util::Xor(hash_key, util::SetLsb(node, false)))[0];
        int4 h1 = prg.Gen(util::Xor(hash_key, util::SetLsb(node, true)))[0];

        int4 high0 = util::SetLsb(h0, false);
        bool low0 = util::GetLsb(h0);
        int4 high1 = util::SetLsb(h1, false);
        bool low1 = util::GetLsb(h1);

        if (t_parent) {
            high0 = util::Xor(high0, hcw);
            low0 = low0 ^ lcw_0;
            high1 = util::Xor(high1, hcw);
            low1 = low1 ^ lcw_1;
        }

        auto y0 = Group::From(high0);
        if (low0) y0 = y0 + Group::From(ocw);
        if (b) y0 = -y0;

        auto y1 = Group::From(high1);
        if (low1) y1 = y1 + Group::From(ocw);
        if (b) y1 = -y1;

        ys[0] = y0.Into();
        ys[1] = y1.Into();
    }
};

}  // namespace fss
