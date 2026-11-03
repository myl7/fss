// SPDX-License-Identifier: Apache-2.0
/**
 * @file point_eval_cpu.cuh
 * @internal
 * CPU optimization candidates retained for benchmark comparison, outside the
 * supported public API.
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
 * apply the control bits of the correction words branchlessly via all-ones masks, so the
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

namespace fss::detail::cpu {

// The narrowest unsigned primitive holding one tr bit per level, in_bits bits
// in total, so common in_bits <= 32 keys keep the uint32_t layout.
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
// branchless selects of the correction word and the child.
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
 * bit i = level i. cws[in_bits] holds the output correction word, whose LSB is 0.
 *
 * Both parties share the same correction words and differ in the seed.
 */
template <int in_bits>
struct DpfPointKey {
  int4 s0;
  int4 cws[in_bits + 1];
  ExtBits<in_bits> tr;
};

/**
 * A party's HalfTree DPF key in the compact point-evaluation layout.
 *
 * cws[i] holds level i's correction word for i = 0 .. in_bits - 2. The
 * last level keeps HCW with LCW_0 in cws[in_bits - 1]'s LSB and LCW_1 in
 * lcw1. cws[in_bits] holds the output correction word from HalfTreeDpf::Gen(). The
 * hash key stays on the scheme instance and is passed to
 * HalfTreeDpfEvalPoint().
 *
 * Both parties share the same correction words and differ in the seed.
 */
template <int in_bits>
struct HalfTreeDpfPointKey {
  int4 s0;
  int4 cws[in_bits + 1];
  bool lcw1;
};

/**
 * One-time compaction of a Dpf key into the layout consumed by DpfEvalPoint().
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=2, same as Dpf.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param key An owning key assembled from Dpf::Gen() outputs.
 * @return The packed key.
 */
template <int in_bits, typename Group, typename Prg, typename In,
    typename Key = typename Dpf<in_bits, Group, Prg, In>::Key>
  requires(std::is_same_v<Key, typename Dpf<in_bits, Group, Prg, In>::Key>)
DpfPointKey<in_bits> DpfPackPoint(const Key &key) {
  DpfPointKey<in_bits> packed = {};
  packed.s0 = key.s0;
  ExtBits<in_bits> tr = 0;
  for (int i = 0; i < in_bits; ++i) {
    packed.cws[i] = key.cws[i].s;
    if (key.cws[i].tr) tr |= static_cast<ExtBits<in_bits>>(1) << i;
  }
  packed.tr = tr;
  packed.cws[in_bits] = key.cws[in_bits].s;
  assert((packed.cws[in_bits].w & 1) == 0);
  return packed;
}

/**
 * CPU point evaluation of a packed DPF key: the output is bit-exact with
 * raw Dpf::Eval() for the corresponding unpacked key material.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Prg See Prgable. Requires mul=2, same as Dpf. Passed by non-const
 *   reference because Prg::Gen() is not const, same as Dpf::Eval().
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Party seed from DpfPackPoint().
 * @param cws in_bits tree correction words followed by the output correction word.
 * @param tr Packed right-branch control bits.
 * @param x Evaluated input.
 * @param prg The DPF instance's PRG.
 * @return Output share.
 */
template <int in_bits, typename Group, typename Prg, typename In>
int4 DpfEvalPoint(bool b, int4 s0, const int4 cws[], ExtBits<in_bits> tr, In x, Prg &prg) {
#ifndef __CUDA_ARCH__
  // The control bit rides in the seed's LSB between levels: the level
  // correction word XORs the same cw into both children, so it cancels in the
  // child select and the selected child's LSB is the next control bit up to
  // the cw's tl/tr difference, fixed branchlessly as xb & d & t below.
  __m128i s = detail::SetLsb(detail::ToVec(s0), b);
  int t = static_cast<int>(b);
  const ExtBits<in_bits> ex = tr;
  const ExtBits<in_bits> xbits = detail::ReverseDomainBits<in_bits>(x);

  FSS_POINT_EVAL_UNROLL_4
  for (int i = 0; i < in_bits; ++i) {
    const __m128i scw = detail::ToVec(cws[i]);
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
  if (t) y = y + Group::From(cws[in_bits]);
  if (b) y = -y;
  return y.Into();
#else
  // This is a host function, so the device pass only parses it and never emits code for it.
  // The SSE path is unavailable there, so this stub stands in to keep the parse valid.
  return make_int4(0, 0, 0, 0);
#endif
}

/**
 * One-time compaction of a HalfTree DPF key into the layout consumed by
 * HalfTreeDpfEvalPoint().
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Hash See CcrHashable, as in HalfTreeDpf.
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param key An owning key assembled from HalfTreeDpf::Gen() outputs.
 * @return The packed key.
 */
template <int in_bits, typename Group, typename Hash, typename In,
    typename Key = typename HalfTreeDpf<in_bits, Group, Hash, In>::Key>
  requires(std::is_same_v<Key, typename HalfTreeDpf<in_bits, Group, Hash, In>::Key>)
HalfTreeDpfPointKey<in_bits> HalfTreeDpfPackPoint(const Key &key) {
  HalfTreeDpfPointKey<in_bits> packed = {};
  packed.s0 = key.s0;
  for (int i = 0; i < in_bits - 1; ++i) packed.cws[i] = key.cws[i].s;
  packed.cws[in_bits - 1] = key.cws[in_bits - 1].s;
  packed.lcw1 = key.cws[in_bits - 1].extra;
  packed.cws[in_bits] = key.cws[in_bits].s;
  return packed;
}

/**
 * CPU point evaluation of a packed HalfTree DPF key: the output is bit-exact
 * with raw HalfTreeDpf::Eval() for the corresponding unpacked key material.
 *
 * @tparam in_bits Input domain bit size.
 * @tparam Group Type for the output domain. See Groupable.
 * @tparam Hash See CcrHashable. Passed by non-const reference, as in HalfTreeDpf::Eval().
 * @tparam In Type for the input domain. From uint8_t to __uint128_t.
 * @param b Party index. False for 0 and true for 1.
 * @param s0 Party seed from HalfTreeDpfPackPoint().
 * @param cws in_bits tree correction words followed by the output correction word.
 * The last tree correction word stores LCW_0 in its least significant bit.
 * @param lcw1 Last tree-level LCW_1.
 * @param x Evaluated input.
 * @param hash The scheme instance's CCR hash.
 * @param instance_seed The scheme instance's public seed S.
 * @return Output share.
 */
template <int in_bits, typename Group, typename Hash, typename In>
int4 HalfTreeDpfEvalPoint(bool b, int4 s0, const int4 cws[], bool lcw1, In x, Hash &hash, int4 instance_seed) {
#ifndef __CUDA_ARCH__
  __m128i node = detail::SetLsb(detail::ToVec(s0), b);
  const __m128i hk = detail::ToVec(instance_seed);
  const ExtBits<in_bits> xbits = detail::ReverseDomainBits<in_bits>(x);

  FSS_POINT_EVAL_UNROLL_4
  for (int i = 0; i < in_bits - 1; ++i) {
    const int xb = detail::GetBit(xbits, i);
    const int t = detail::GetLsb(node);

    const __m128i h = detail::ToVec(hash.Hash(detail::FromVec(_mm_xor_si128(hk, node))));
    const __m128i cw = detail::ToVec(cws[i]);

    node = _mm_xor_si128(_mm_xor_si128(h, _mm_and_si128(node, detail::BitMask(xb))),
        _mm_and_si128(cw, detail::BitMask(t)));
  }

  const int xn = detail::GetBit(xbits, in_bits - 1);
  const int t = detail::GetLsb(node);
  const __m128i h = detail::ToVec(hash.Hash(detail::FromVec(_mm_xor_si128(hk, detail::SetLsb(node, xn != 0)))));

  const __m128i hcw_vector = detail::ToVec(cws[in_bits - 1]);
  const int lcw_xn = xn != 0 ? static_cast<int>(lcw1) : detail::GetLsb(hcw_vector);
  const __m128i hcw = detail::ClrLsb(hcw_vector);

  __m128i high = detail::ClrLsb(h);
  int low = detail::GetLsb(h);
  const __m128i tm = detail::BitMask(t);
  high = _mm_xor_si128(high, _mm_and_si128(hcw, tm));
  low = low ^ (lcw_xn & t);

  auto y = Group::From(detail::FromVec(high));
  if (low) y = y + Group::From(cws[in_bits]);
  if (b) y = -y;
  return y.Into();
#else
  // This is a host function, so the device pass only parses it and never emits code for it.
  // The SSE path is unavailable there, so this stub stands in to keep the parse valid.
  return make_int4(0, 0, 0, 0);
#endif
}

}  // namespace fss::detail::cpu

#undef FSS_POINT_EVAL_UNROLL_4

#endif  // FSS_POINT_EVAL_CPU_CUH_
