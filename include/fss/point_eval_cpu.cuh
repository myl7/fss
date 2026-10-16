// SPDX-License-Identifier: Apache-2.0
/**
 * @file point_eval_cpu.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief CPU point evaluation of FSS keys in the lean packed-key layout.
 *
 * The scheme classes keep each correction word in a 32 B slot (16 B s plus
 * flag bytes), which the GPU needs for unmasked wide stores but which costs
 * the host evaluator a 32 B stride per level and two loads where one would
 * do. Run the matching *PackPoint() once per key after Gen(); it compacts a
 * key to aligned 16 B correction words per level, packs the per-level tr
 * bits into one word, and splits out the output correction word. The pack
 * cost amortizes over every later point evaluation, same as Gen itself sits
 * outside the timed loop.
 *
 * The *EvalPoint() walks keep the seed in __m128i end to end (the Prg int4
 * interface is wrapped at the Gen boundary), pre-extract the query bits, and
 * apply the control-bit corrections branchlessly via all-ones masks, so the
 * per-level work is one PRG expansion plus a handful of XOR/AND ops.
 *
 * The outputs are bit-exact with Dpf::Eval() / HalfTreeDpf::Eval(), for any
 * Groupable Group and Prgable Prg, and cover the same input-domain widths as
 * the host Eval, through __uint128_t.
 *
 * Only for host side (x86-64 with SSE4.1; the PRG may additionally require
 * AES-NI, e.g. Aes128MmoRaw).
 */

#ifndef FSS_POINT_EVAL_CPU_CUH_
#define FSS_POINT_EVAL_CPU_CUH_

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
  #include <emmintrin.h>
  #include <smmintrin.h>
#endif

#include <fss/dpf.cuh>
#include <fss/group.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/prg.cuh>
#include <fss/util.cuh>

// Unroll hint spelling per compiler: clang honors both spellings, GCC only
// the GCC one (it warns on bare #pragma unroll), and nvcc's frontend
// intercepts the GCC spelling with a warning while recognizing its own.
// The device pass never expands the macro: it only guards host-only bodies.
#if defined(__CUDACC__) || (!defined(__GNUC__) && !defined(__clang__))
#define FSS_POINT_EVAL_UNROLL_4 _Pragma("unroll 4")
#else
#define FSS_POINT_EVAL_UNROLL_4 _Pragma("GCC unroll 4")
#endif

namespace fss::cpu {

// The narrowest unsigned primitive holding one tr bit per level, in_bits bits
// in total. Mirrors fss::gpu::ExtBits so common in_bits <= 32 keys keep the
// uint32_t layout.
template <int bits>
using ExtBits = std::conditional_t<bits <= 32, uint32_t,
    std::conditional_t<bits <= 64, uint64_t, __uint128_t>>;

#ifndef __CUDA_ARCH__
namespace detail {

// The seed's LSB lives in int4.w's bit 0, i.e. bit 32 of the high 64-bit
// half in the little-endian SSE layout.

inline __m128i ToVec(int4 val) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i *>(&val));
}

inline int4 FromVec(__m128i val) {
  int4 out;
  _mm_storeu_si128(reinterpret_cast<__m128i *>(&out), val);
  return out;
}

inline int GetLsb(__m128i val) {
  return static_cast<int>((_mm_extract_epi64(val, 1) >> 32) & 1);
}

inline __m128i ClrLsb(__m128i val) {
  return _mm_and_si128(val, _mm_set_epi64x(~(int64_t{1} << 32), ~int64_t{0}));
}

inline __m128i SetLsb(__m128i val, bool bit) {
  const __m128i lsb = _mm_set_epi64x(int64_t{1} << 32, 0);
  if (bit) return _mm_or_si128(val, lsb);
  return _mm_andnot_si128(lsb, val);
}

// All-ones 128-bit mask when bit is 1, all-zero when bit is 0, for the
// branchless correction-word and child selects.
inline __m128i BitMask(int bit) {
  return _mm_set1_epi64x(-static_cast<int64_t>(bit));
}

template <typename T>
inline int GetBit(T x, int i) {
  return static_cast<int>((x >> i) & 1);
}

// Reverse the low in_bits bits of a <=64-bit value so that level i reads its
// input bit with a constant shift, whatever the loop unroller did.
template <int in_bits>
inline ExtBits<in_bits> ReverseBits(uint64_t x) {
  static_assert(in_bits >= 1 && in_bits <= 64);
  x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
  x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
  x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
  x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
  x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
  x = (x << 32) | (x >> 32);
  return static_cast<ExtBits<in_bits>>(x >> (64 - in_bits));
}

template <int in_bits>
inline ExtBits<in_bits> ReverseBits(__uint128_t x) {
  static_assert(in_bits >= 65 && in_bits <= 128);
  const uint64_t lo = ReverseBits<64>(static_cast<uint64_t>(x));
  const uint64_t hi = ReverseBits<64>(static_cast<uint64_t>(x >> 64));
  const __uint128_t r = static_cast<__uint128_t>(hi) | (static_cast<__uint128_t>(lo) << 64);
  return static_cast<ExtBits<in_bits>>(r >> (128 - in_bits));
}

template <int in_bits, typename In>
inline ExtBits<in_bits> ReverseDomainBits(In x) {
  if constexpr (in_bits <= 64) {
    return ReverseBits<in_bits>(static_cast<uint64_t>(x));
  } else {
    return ReverseBits<in_bits>(static_cast<__uint128_t>(x));
  }
}

}  // namespace detail
#endif  // !__CUDA_ARCH__

/**
 * A party's DPF key in the compact point-evaluation layout.
 *
 * cws[i] keeps level i's correction word with tl in its LSB, as in
 * Dpf::Cw::s. The per-level tr bits are packed into the single tr word,
 * bit i = level i. out_cw is the output correction word
 * cws[in_bits].s, whose LSB is 0.
 *
 * Both parties share the same correction words and differ in the seed.
 */
template <int in_bits>
struct DpfPointKey {
  int4 s0;
  int4 cws[in_bits];
  ExtBits<in_bits> tr;
  int4 out_cw;
};

/**
 * A party's HalfTree DPF key in the compact point-evaluation layout.
 *
 * cws[i] holds level i's correction word for i = 0 .. in_bits - 2. The
 * last level keeps HCW with LCW_0 in hcw_lcw0's LSB, LCW_1 in lcw1, and
 * the output correction word in ocw, as packed by HalfTreeDpf::Gen(). The
 * hash key stays on the scheme instance and is passed to
 * HalfTreeDpfEvalPoint().
 *
 * cws has size 1 and is unused when in_bits == 1.
 *
 * Both parties share the same correction words and differ in the seed.
 */
template <int in_bits>
struct HalfTreeDpfPointKey {
  int4 s0;
  int4 cws[in_bits > 1 ? in_bits - 1 : 1];
  int4 hcw_lcw0;
  bool lcw1;
  int4 ocw;
};

/**
 * One-time compaction of a Dpf key into the layout consumed by DpfEvalPoint().
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=2, same as Dpf.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param key A party's key returned by Dpf::Gen().
 * @return The packed key.
 */
template <int in_bits, typename Group, typename Prg, typename In>
DpfPointKey<in_bits> DpfPackPoint(const typename Dpf<in_bits, Group, Prg, In>::Key &key) {
  DpfPointKey<in_bits> packed = {};
  packed.s0 = key.s0;
  ExtBits<in_bits> tr = 0;
  for (int i = 0; i < in_bits; ++i) {
    packed.cws[i] = key.cws[i].s;
    if (key.cws[i].tr) tr |= static_cast<ExtBits<in_bits>>(1) << i;
  }
  packed.tr = tr;
  packed.out_cw = key.cws[in_bits].s;
  assert((packed.out_cw.w & 1) == 0);
  return packed;
}

/**
 * CPU point evaluation of a packed DPF key: the output is bit-exact with
 * Dpf::Eval(b, key, x) for the key DpfPackPoint() was called on.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=2, same as Dpf. Passed by non-const
 *   reference because Prg::Gen() is not const, same as Dpf::Eval().
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param b Party index. False for 0 and true for 1.
 * @param key The packed key from DpfPackPoint().
 * @param x Evaluated input.
 * @param prg The DPF instance's PRG.
 * @return Output share.
 */
template <int in_bits, typename Group, typename Prg, typename In>
int4 DpfEvalPoint(bool b, const DpfPointKey<in_bits> &key, In x, Prg &prg) {
#ifndef __CUDA_ARCH__
  // The control bit rides in the seed's LSB between levels: the level
  // correction XORs the same cw into both children, so it cancels in the
  // child select and the selected child's LSB is the next control bit up to
  // the cw's tl/tr difference, fixed branchlessly as xb & d & t below.
  __m128i s = detail::SetLsb(detail::ToVec(key.s0), b);
  int t = static_cast<int>(b);
  const ExtBits<in_bits> ex = key.tr;
  const ExtBits<in_bits> xbits = detail::ReverseDomainBits<in_bits>(x);

  FSS_POINT_EVAL_UNROLL_4
  for (int i = 0; i < in_bits; ++i) {
    const __m128i scw = detail::ToVec(key.cws[i]);
    const int d = detail::GetLsb(scw) ^ detail::GetBit(ex, i);

    const auto g = prg.Gen(detail::FromVec(detail::ClrLsb(s)));
    const __m128i g0 = detail::ToVec(g[0]);
    const __m128i g1 = detail::ToVec(g[1]);

    const __m128i cwm = _mm_and_si128(scw, detail::BitMask(t));
    const __m128i sl = _mm_xor_si128(g0, cwm);

    const int xb = detail::GetBit(xbits, i);
    s = _mm_xor_si128(sl, _mm_and_si128(_mm_xor_si128(g0, g1), detail::BitMask(xb)));
    t = detail::GetLsb(s) ^ (xb & d & t);
  }

  auto y = Group::From(detail::FromVec(detail::ClrLsb(s)));
  if (t) y = y + Group::From(key.out_cw);
  if (b) y = -y;
  return y.Into();
#else
  assert(false && "DpfEvalPoint is host-only");
  __trap();
  return make_int4(0, 0, 0, 0);
#endif
}

/**
 * One-time compaction of a HalfTree DPF key into the layout consumed by
 * HalfTreeDpfEvalPoint().
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=1, same as HalfTreeDpf.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param key A party's key returned by HalfTreeDpf::Gen().
 * @return The packed key.
 */
template <int in_bits, typename Group, typename Prg, typename In>
HalfTreeDpfPointKey<in_bits> HalfTreeDpfPackPoint(
    const typename HalfTreeDpf<in_bits, Group, Prg, In>::Key &key) {
  HalfTreeDpfPointKey<in_bits> packed = {};
  packed.s0 = key.s0;
  for (int i = 0; i < in_bits - 1; ++i) packed.cws[i] = key.cws[i].s;
  packed.hcw_lcw0 = key.cws[in_bits - 1].s;
  packed.lcw1 = key.cws[in_bits - 1].extra;
  packed.ocw = key.ocw;
  return packed;
}

/**
 * CPU point evaluation of a packed HalfTree DPF key: the output is bit-exact
 * with HalfTreeDpf::Eval(b, key, x) for the key HalfTreeDpfPackPoint() was
 * called on.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=1, same as HalfTreeDpf. Passed by
 *   non-const reference because Prg::Gen() is not const, same as
 *   HalfTreeDpf::Eval().
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param b Party index. False for 0 and true for 1.
 * @param key The packed key from HalfTreeDpfPackPoint().
 * @param x Evaluated input.
 * @param prg The scheme instance's PRG.
 * @param hash_key The scheme instance's hash key.
 * @return Output share.
 */
template <int in_bits, typename Group, typename Prg, typename In>
int4 HalfTreeDpfEvalPoint(bool b, const HalfTreeDpfPointKey<in_bits> &key, In x, Prg &prg, int4 hash_key) {
#ifndef __CUDA_ARCH__
  __m128i node = detail::SetLsb(detail::ToVec(key.s0), b);
  const __m128i hk = detail::ToVec(hash_key);
  const ExtBits<in_bits> xbits = detail::ReverseDomainBits<in_bits>(x);

  FSS_POINT_EVAL_UNROLL_4
  for (int i = 0; i < in_bits - 1; ++i) {
    const int xb = detail::GetBit(xbits, i);
    const int t = detail::GetLsb(node);

    const __m128i h = detail::ToVec(prg.Gen(detail::FromVec(_mm_xor_si128(hk, node)))[0]);
    const __m128i cw = detail::ToVec(key.cws[i]);

    node = _mm_xor_si128(_mm_xor_si128(h, _mm_and_si128(node, detail::BitMask(xb))),
        _mm_and_si128(cw, detail::BitMask(t)));
  }

  const int xn = detail::GetBit(xbits, in_bits - 1);
  const int t = detail::GetLsb(node);
  const __m128i h = detail::ToVec(prg.Gen(detail::FromVec(_mm_xor_si128(hk, detail::SetLsb(node, xn != 0))))[0]);

  const __m128i hcw_lcw0 = detail::ToVec(key.hcw_lcw0);
  const int lcw_xn = xn != 0 ? static_cast<int>(key.lcw1) : detail::GetLsb(hcw_lcw0);
  const __m128i hcw = detail::ClrLsb(hcw_lcw0);

  __m128i high = detail::ClrLsb(h);
  int low = detail::GetLsb(h);
  const __m128i tm = detail::BitMask(t);
  high = _mm_xor_si128(high, _mm_and_si128(hcw, tm));
  low = low ^ (lcw_xn & t);

  auto y = Group::From(detail::FromVec(high));
  if (low) y = y + Group::From(key.ocw);
  if (b) y = -y;
  return y.Into();
#else
  assert(false && "HalfTreeDpfEvalPoint is host-only");
  __trap();
  return make_int4(0, 0, 0, 0);
#endif
}

}  // namespace fss::cpu

#undef FSS_POINT_EVAL_UNROLL_4

#endif  // FSS_POINT_EVAL_CPU_CUH_
