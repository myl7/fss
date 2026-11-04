// SPDX-License-Identifier: Apache-2.0
/**
 * @file hash/blake3.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <fss/hash.cuh>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cassert>
#include <fss/util.cuh>

// SSE2 is used only for the optional four-message host batch path.
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && \
    (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)) && \
    !defined(__CUDA_ARCH__)
#define FSS_BLAKE3_HAS_SSE2 1
#else
#define FSS_BLAKE3_HAS_SSE2 0
#endif

#if FSS_BLAKE3_HAS_SSE2 && defined(__has_include)
#if !__has_include(<emmintrin.h>)
#undef FSS_BLAKE3_HAS_SSE2
#define FSS_BLAKE3_HAS_SSE2 0
#endif
#endif

#if FSS_BLAKE3_HAS_SSE2
#include <emmintrin.h>
#endif

namespace fss::hash {

/**
 * Declared and never defined. Device code that reaches it fails to link with an
 * unresolved-extern error that names this symbol.
 */
__device__ void fss_host_only_Blake3_Hash4_used_in_device_code();

/**
 * BLAKE3 keyed hash (CPU+GPU).
 *
 * The IV replaces BLAKE3's standard IV in keyed-hash mode.
 * The KEYED_HASH flag is set on all compression calls.
 */
class Blake3 {
public:
  /**
     * Constructor.
     *
     * @param iv Custom IV (2 int4s = 32B) replacing BLAKE3's standard IV in keyed-hash mode.
     * Stored by value.
     */
  __host__ __device__ explicit Blake3(cuda::std::span<const int4, 2> iv) : iv_{iv[0], iv[1]} {}

  /**
     * Hash a 64B message.
     *
     * Single BLAKE3 compression: h=iv_, m=msg, counter=0, block_len=64.
     * flags = CHUNK_START | CHUNK_END | ROOT | KEYED_HASH (0x1B).
     *
     * @param msg 4 int4 blocks (64B).
     * @return 2 int4 blocks (32B), the first 32B of the compression output.
     */
  __host__ __device__ cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4> msg);

  /**
     * Hash a 128+127-bit logical message into two 32B digests.
     *
     * All 128 bits of a are message data. Bit 0 of b.w is reserved for
     * domain separation; its supplied value is ignored. The other 127 bits
     * of b are message data. Hash a followed by a copy of b with its reserved
     * bit set to 0, then to 1, and concatenate the digests in that order.
     * The input blocks are not modified.
     * Each digest uses a 32B message, counter 0, and flags
     * CHUNK_START | CHUNK_END | ROOT | KEYED_HASH under the configured IV.
     *
     * @param msg Tuple of (a, b), with the reserved bit in the second block.
     * @return 4 int4 blocks (64B).
     */
  __host__ __device__ cuda::std::array<int4, 4> Hash(cuda::std::tuple<int4, const int4> msg);

  /**
     * Hash four 64B messages.
     *
     * Produces the same 32B digest as Hash for each message on this instance.
     * The input is borrowed and the output is owned.
     * Only for host side; device use fails to build with an unresolved-extern error
     * naming `fss_host_only_Blake3_Hash4_used_in_device_code`.
     *
     * @param msgs Four 64B messages.
     * @return Four 32B digests.
     */
  __host__ __device__ cuda::std::array<cuda::std::array<int4, 2>, 4> Hash4(
      cuda::std::span<const cuda::std::array<int4, 4>, 4> msgs);

private:
  int4 iv_[2];

  template <int n>
  __host__ __device__ static int RotateRight(int a);

  __host__ __device__ static void G(int &a, int &b, int &c, int &d, int x, int y);

  __host__ __device__ static int GetWord(const int4 m[4], int i);

  __host__ __device__ static void SetWord(int4 m[4], int i, int val);

  __host__ __device__ static void Permute(int4 m[4]);

  __host__ __device__ static void Round(int4 v[4], const int4 m[4]);

#if FSS_BLAKE3_HAS_SSE2
  static int LaneWord(const int4 &v, int i);

  static void SetLaneWord(int4 &v, int i, int x);

  template <int n>
  static __m128i Rot(__m128i a);

  static void G4(__m128i &a, __m128i &b, __m128i &c, __m128i &d, __m128i x, __m128i y);

  static void Batch4(const int4 iv[2], const cuda::std::array<int4, 4> *in,
      cuda::std::array<int4, 2> *out);
#endif

  // First 4 standard BLAKE3 IV words
  constexpr static int4 kIv0 = {
      static_cast<int>(0x6A09E667),
      static_cast<int>(0xBB67AE85),
      static_cast<int>(0x3C6EF372),
      static_cast<int>(0xA54FF53A),
  };

  constexpr static int kChunkStart = 1;
  constexpr static int kChunkEnd = 2;
  constexpr static int kRoot = 8;
  constexpr static int kKeyedHash = 16;

  // Compress one message block using the supplied chaining value and flags.
  __host__ __device__ static cuda::std::array<int4, 4> Compress(
      const int4 h[2], const int4 msg[4], unsigned long long counter, int block_len, int flags);
};
// -----------------------------------------------------------------------------

template <int n>
__host__ __device__ inline int Blake3::RotateRight(int a) {
  auto u = static_cast<unsigned>(a);
  return static_cast<int>((u >> n) | (u << (32 - n)));
}

__host__ __device__ inline void Blake3::G(int &a, int &b, int &c, int &d, int x, int y) {
  a = static_cast<int>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b) + static_cast<uint32_t>(x));
  d = RotateRight<16>(d ^ a);
  c = static_cast<int>(static_cast<uint32_t>(c) + static_cast<uint32_t>(d));
  b = RotateRight<12>(b ^ c);
  a = static_cast<int>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b) + static_cast<uint32_t>(y));
  d = RotateRight<8>(d ^ a);
  c = static_cast<int>(static_cast<uint32_t>(c) + static_cast<uint32_t>(d));
  b = RotateRight<7>(b ^ c);
}

__host__ __device__ inline int Blake3::GetWord(const int4 m[4], int i) {
  return reinterpret_cast<const int *>(m)[i];
}

__host__ __device__ inline void Blake3::SetWord(int4 m[4], int i, int val) {
  reinterpret_cast<int *>(m)[i] = val;
}

__host__ __device__ inline void Blake3::Permute(int4 m[4]) {
  constexpr int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  int tmp[16];
  for (int i = 0; i < 16; ++i) {
    tmp[i] = GetWord(m, perm[i]);
  }
  for (int i = 0; i < 16; ++i) {
    SetWord(m, i, tmp[i]);
  }
}

__host__ __device__ inline void Blake3::Round(int4 v[4], const int4 m[4]) {
  // Column rounds: G calls use message words (0,1), (2,3), (4,5), (6,7)
  G(v[0].x, v[1].x, v[2].x, v[3].x, GetWord(m, 0), GetWord(m, 1));
  G(v[0].y, v[1].y, v[2].y, v[3].y, GetWord(m, 2), GetWord(m, 3));
  G(v[0].z, v[1].z, v[2].z, v[3].z, GetWord(m, 4), GetWord(m, 5));
  G(v[0].w, v[1].w, v[2].w, v[3].w, GetWord(m, 6), GetWord(m, 7));
  // Diagonal rounds: G calls use message words (8,9), (10,11), (12,13), (14,15)
  G(v[0].x, v[1].y, v[2].z, v[3].w, GetWord(m, 8), GetWord(m, 9));
  G(v[0].y, v[1].z, v[2].w, v[3].x, GetWord(m, 10), GetWord(m, 11));
  G(v[0].z, v[1].w, v[2].x, v[3].y, GetWord(m, 12), GetWord(m, 13));
  G(v[0].w, v[1].x, v[2].y, v[3].z, GetWord(m, 14), GetWord(m, 15));
}

__host__ __device__ inline cuda::std::array<int4, 4> Blake3::Compress(
    const int4 h[2], const int4 msg[4], unsigned long long counter, int block_len, int flags) {
  int4 v[4];
  v[0] = h[0];
  v[1] = h[1];
  v[2] = kIv0;
  v[3] = {static_cast<int>(counter & 0xFFFFFFFF), static_cast<int>((counter >> 32) & 0xFFFFFFFF), block_len, flags};

  int4 m[4] = {msg[0], msg[1], msg[2], msg[3]};

  // 7 rounds with permutation between rounds
  for (int i = 0; i < 7; ++i) {
    Round(v, m);
    if (i < 6) {
      Permute(m);
    }
  }

  // Finalization
  v[0] = util::Xor(v[0], v[2]);
  v[1] = util::Xor(v[1], v[3]);
  v[2] = util::Xor(v[2], h[0]);
  v[3] = util::Xor(v[3], h[1]);

  return {v[0], v[1], v[2], v[3]};
}

__host__ __device__ inline cuda::std::array<int4, 2> Blake3::Hash(cuda::std::span<const int4, 4> msg) {
  constexpr int flags = kChunkStart | kChunkEnd | kRoot | kKeyedHash;
  auto out = Compress(iv_, msg.data(), 0, 64, flags);
  return {out[0], out[1]};
}

__host__ __device__ inline cuda::std::array<int4, 4> Blake3::Hash(cuda::std::tuple<int4, const int4> msg) {
  constexpr int flags = kChunkStart | kChunkEnd | kRoot | kKeyedHash;
  auto [a, b] = msg;
  int4 padded[4] = {a, util::SetLsb(b, false), {0, 0, 0, 0}, {0, 0, 0, 0}};

  auto out0 = Compress(iv_, padded, 0, 32, flags);

  padded[1] = util::SetLsb(b, true);
  auto out1 = Compress(iv_, padded, 0, 32, flags);

  return {out0[0], out0[1], out1[0], out1[1]};
}

#if FSS_BLAKE3_HAS_SSE2
inline int Blake3::LaneWord(const int4 &v, int i) {
  switch (i) {
    case 0:
      return v.x;
    case 1:
      return v.y;
    case 2:
      return v.z;
    default:
      return v.w;
  }
}

inline void Blake3::SetLaneWord(int4 &v, int i, int x) {
  switch (i) {
    case 0:
      v.x = x;
      break;
    case 1:
      v.y = x;
      break;
    case 2:
      v.z = x;
      break;
    default:
      v.w = x;
      break;
  }
}

template <int n>
inline __m128i Blake3::Rot(__m128i a) {
  return _mm_or_si128(_mm_srli_epi32(a, n), _mm_slli_epi32(a, 32 - n));
}

inline void Blake3::G4(__m128i &a, __m128i &b, __m128i &c, __m128i &d, __m128i x, __m128i y) {
  a = _mm_add_epi32(_mm_add_epi32(a, b), x);
  d = Rot<16>(_mm_xor_si128(d, a));
  c = _mm_add_epi32(c, d);
  b = Rot<12>(_mm_xor_si128(b, c));
  a = _mm_add_epi32(_mm_add_epi32(a, b), y);
  d = Rot<8>(_mm_xor_si128(d, a));
  c = _mm_add_epi32(c, d);
  b = Rot<7>(_mm_xor_si128(b, c));
}

// Hashes four 64B messages in parallel, one message per SSE2 lane.
inline void Blake3::Batch4(const int4 iv[2], const cuda::std::array<int4, 4> *in,
    cuda::std::array<int4, 2> *out) {
  __m128i v[16];
  __m128i m[16];
  for (int w = 0; w < 16; ++w) {
    const int x0 = LaneWord(in[0][w / 4], w % 4);
    const int x1 = LaneWord(in[1][w / 4], w % 4);
    const int x2 = LaneWord(in[2][w / 4], w % 4);
    const int x3 = LaneWord(in[3][w / 4], w % 4);
    m[w] = _mm_set_epi32(x3, x2, x1, x0);
  }
  for (int w = 0; w < 8; ++w) {
    v[w] = _mm_set1_epi32(LaneWord(iv[w / 4], w % 4));
  }
  const int init[8] = {static_cast<int>(0x6A09E667u), static_cast<int>(0xBB67AE85u),
      static_cast<int>(0x3C6EF372u), static_cast<int>(0xA54FF53Au), 0, 0, 64, 0x1B};
  for (int w = 0; w < 8; ++w) {
    v[w + 8] = _mm_set1_epi32(init[w]);
  }
  constexpr int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  for (int r = 0; r < 7; ++r) {
    G4(v[0], v[4], v[8], v[12], m[0], m[1]);
    G4(v[1], v[5], v[9], v[13], m[2], m[3]);
    G4(v[2], v[6], v[10], v[14], m[4], m[5]);
    G4(v[3], v[7], v[11], v[15], m[6], m[7]);
    G4(v[0], v[5], v[10], v[15], m[8], m[9]);
    G4(v[1], v[6], v[11], v[12], m[10], m[11]);
    G4(v[2], v[7], v[8], v[13], m[12], m[13]);
    G4(v[3], v[4], v[9], v[14], m[14], m[15]);
    if (r < 6) {
      __m128i next[16];
      for (int w = 0; w < 16; ++w) {
        next[w] = m[perm[w]];
      }
      for (int w = 0; w < 16; ++w) {
        m[w] = next[w];
      }
    }
  }
  // Extract with defined SSE2 shifts rather than aliasing stores.
  for (int w = 0; w < 8; ++w) {
    const __m128i o = _mm_xor_si128(v[w], v[w + 8]);
    const int x0 = _mm_cvtsi128_si32(o);
    const int x1 = _mm_cvtsi128_si32(_mm_srli_si128(o, 4));
    const int x2 = _mm_cvtsi128_si32(_mm_srli_si128(o, 8));
    const int x3 = _mm_cvtsi128_si32(_mm_srli_si128(o, 12));
    SetLaneWord(out[0][w / 4], w % 4, x0);
    SetLaneWord(out[1][w / 4], w % 4, x1);
    SetLaneWord(out[2][w / 4], w % 4, x2);
    SetLaneWord(out[3][w / 4], w % 4, x3);
  }
}

#endif

__host__ __device__ inline cuda::std::array<cuda::std::array<int4, 2>, 4> Blake3::Hash4(
    cuda::std::span<const cuda::std::array<int4, 4>, 4> msgs) {
#ifdef __CUDA_ARCH__
  fss_host_only_Blake3_Hash4_used_in_device_code();
  return {};
#else
  cuda::std::array<cuda::std::array<int4, 2>, 4> out;
#if FSS_BLAKE3_HAS_SSE2
  Batch4(iv_, msgs.data(), out.data());
#else
  for (int i = 0; i < 4; ++i) {
    out[i] = Hash(cuda::std::span<const int4, 4>(msgs[i].data(), 4));
  }
#endif

  return out;
#endif
}

static_assert(Hashable<Blake3> && XorHashable<Blake3>);

}  // namespace fss::hash

#undef FSS_BLAKE3_HAS_SSE2
