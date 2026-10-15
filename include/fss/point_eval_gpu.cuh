/**
 * GPU point evaluation of FSS keys in level-major correction-word layout.
 *
 * The scheme headers store each key's correction words contiguously (per-key
 * layout, 32B stride per level), so a thread walking one key's path touches a
 * different 128B line at every level and the warp's 32 threads touch 32 lines.
 * The DRAM traffic is then ~4x the bytes the data needs, which caps 20-level
 * point evaluation well below the ChaCha ALU floor. These kernels read a
 * level-major layout instead (level i of every key adjacent), so each level
 * load is one coalesced 128B transaction per warp. The VDPF kernel keeps its
 * per-key cs/ocw: each is read once per evaluation, not once per level.
 *
 * Run the matching Repack*Gpu() once per key after Gen(); the repack cost
 * amortizes over every later point evaluation, same as Gen itself sits outside
 * the timed loop.
 *
 * The query points use the scheme's In type, so these kernels cover the same
 * input-domain widths as the host Eval, through __uint128_t. The packed tr
 * bits (extra) hold one bit per level and widen to uint64_t or __uint128_t
 * once in_bits passes 32 or 64.
 */
#ifndef FSS_POINT_EVAL_GPU_CUH_
#define FSS_POINT_EVAL_GPU_CUH_

#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <fss/cuckoo_hash.cuh>
#include <fss/dcf.cuh>
#include <fss/dmpf.cuh>
#include <fss/dpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/util.cuh>
#include <fss/vdpf.cuh>

namespace fss::gpu {

// The narrowest unsigned primitive holding one tr bit per level, in_bits bits
// in total. Common in_bits <= 32 keys keep the uint32_t layout, so existing
// callers see no change.
template <int bits>
using ExtBits = std::conditional_t<bits <= 32, uint32_t,
    std::conditional_t<bits <= 64, uint64_t, __uint128_t>>;

namespace detail {

__device__ __host__ inline int4 Zero4() { return {0, 0, 0, 0}; }

template <typename T>
__device__ __host__ inline bool GetBit(T x, int i) { return ((x >> i) & 1) != 0; }

// Pack per-key HalfTree DPF cws (stride in_bits) into level-major cw_s and the
// last-level extra bit (bit 0 of extra[k]).
template <int in_bits, typename Cw>
__global__ void HalfTreeDpfRepackKernel(const Cw *pk, int4 *cw_s, uint32_t *extra, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  uint32_t e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * in_bits + i].s;
    if (i == in_bits - 1 && pk[(size_t)k * in_bits + i].extra) e = 1;
  }
  extra[k] = e;
}

// Pack per-key DPF cws (stride in_bits + 1) into level-major cw_s, the tr bits
// (bit i of extra[k] = level i), and the output correction word.
template <int in_bits, typename Cw>
__global__ void DpfRepackKernel(const Cw *pk, int4 *cw_s, ExtBits<in_bits> *extra, int4 *out_cw, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  ExtBits<in_bits> e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].s;
    if (pk[(size_t)k * (in_bits + 1) + i].tr) e |= static_cast<ExtBits<in_bits>>(1) << i;
  }
  extra[k] = e;
  out_cw[k] = pk[(size_t)k * (in_bits + 1) + in_bits].s;
}

// Pack per-key VDPF cws (stride in_bits) into level-major cw_s and the tr
// bits (bit i of extra[k] = level i).
template <int in_bits, typename Cw>
__global__ void VdpfRepackKernel(const Cw *pk, int4 *cw_s, ExtBits<in_bits> *extra, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  ExtBits<in_bits> e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * in_bits + i].s;
    if (pk[(size_t)k * in_bits + i].tr) e |= static_cast<ExtBits<in_bits>>(1) << i;
  }
  extra[k] = e;
}

// Pack per-key DCF cws (stride in_bits + 1) into level-major cw_s and cw_v and
// the final correction word's v part.
template <int in_bits, typename Cw>
__global__ void DcfRepackKernel(const Cw *pk, int4 *cw_s, int4 *cw_v, int4 *out_cw, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].s;
    cw_v[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].v;
  }
  out_cw[k] = pk[(size_t)k * (in_bits + 1) + in_bits].v;
}

// Point-evaluate in_bits-level HalfTree DPF keys: one thread per key, one
// output value per key at xs[tid]. cw_s/extra come from HalfTreeDpfRepackKernel.
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename In>
__global__ void HalfTreeDpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s,
    const uint32_t *extra, const int4 *ocws, const In *xs, int nkeys, Prg prg, int4 hash_key) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = xs[tid];

#pragma unroll unroll
  for (int i = 0; i < in_bits - 1; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    bool t = fss::util::GetLsb(node);
    int4 h = prg.Gen(fss::util::Xor(hash_key, node))[0];
    int4 cw = cw_s[(size_t)i * nkeys + tid];
    node = fss::util::Xor(fss::util::Xor(h, x_bit ? node : Zero4()), t ? cw : Zero4());
  }

  bool x_n = GetBit(x, 0);
  bool t = fss::util::GetLsb(node);
  int4 h = prg.Gen(fss::util::Xor(hash_key, fss::util::SetLsb(node, x_n)))[0];
  int4 cw = cw_s[(size_t)(in_bits - 1) * nkeys + tid];
  int4 hcw = fss::util::SetLsb(cw, false);
  bool lcw_xn = x_n ? ((extra[tid] & 1) != 0) : fss::util::GetLsb(cw);
  int4 high = fss::util::SetLsb(h, false);
  bool low = fss::util::GetLsb(h);
  if (t) {
    high = fss::util::Xor(high, hcw);
    low = low ^ lcw_xn;
  }
  auto y = Group::From(high);
  if (low) y = y + Group::From(ocws[tid]);
  if constexpr (b) y = -y;
  ys[tid] = y.Into();
}

// Point-evaluate in_bits-level DPF keys, one thread per key. cw_s/extra/out_cw
// come from DpfRepackKernel.
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename In>
__global__ void DpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const int4 *out_cw, const In *xs, int nkeys, Prg prg) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  bool t = b;
  In x = xs[tid];
  ExtBits<in_bits> ex = extra[tid];

#pragma unroll unroll
  for (int i = 0; i < in_bits; ++i) {
    int4 s_cw = cw_s[(size_t)i * nkeys + tid];
    bool tr_cw = GetBit(ex, i);
    bool tl_cw = fss::util::GetLsb(s_cw);
    s_cw = fss::util::SetLsb(s_cw, false);

    auto g = prg.Gen(s);
    int4 sl = g[0], sr = g[1];
    bool tl = fss::util::GetLsb(sl);
    sl = fss::util::SetLsb(sl, false);
    bool tr = fss::util::GetLsb(sr);
    sr = fss::util::SetLsb(sr, false);

    if (t) {
      sl = fss::util::Xor(sl, s_cw);
      sr = fss::util::Xor(sr, s_cw);
      tl = tl ^ tl_cw;
      tr = tr ^ tr_cw;
    }

    bool x_bit = GetBit(x, in_bits - 1 - i);
    if (x_bit) {
      s = sr;
      t = tr;
    } else {
      s = sl;
      t = tl;
    }
  }

  auto y = Group::From(s);
  if (t) y = y + Group::From(out_cw[tid]);
  if constexpr (b) y = -y;
  ys[tid] = y.Into();
}

// Point-evaluate in_bits-level DCF keys, one thread per key. cw_s/cw_v/out_cw
// come from DcfRepackKernel. v starts at zero, which is the value the
// library Eval relies on the compiler to produce for its uninitialized v.
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename In>
__global__ void DcfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const int4 *cw_v,
    const int4 *out_cw, const In *xs, int nkeys, Prg prg) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  Group v = Group::From(Zero4());
  bool t = b;
  In x = xs[tid];

#pragma unroll unroll
  for (int i = 0; i < in_bits; ++i) {
    int4 s_cw = cw_s[(size_t)i * nkeys + tid];
    bool tl_cw = fss::util::GetLsb(s_cw);
    s_cw = fss::util::SetLsb(s_cw, false);
    int4 v_cw_buf = cw_v[(size_t)i * nkeys + tid];
    bool tr_cw = fss::util::GetLsb(v_cw_buf);
    v_cw_buf = fss::util::SetLsb(v_cw_buf, false);
    auto v_cw = Group::From(v_cw_buf);

    auto g = prg.Gen(s);
    int4 sl = g[0], vl_buf = g[1], sr = g[2], vr_buf = g[3];

    bool tl = fss::util::GetLsb(sl);
    sl = fss::util::SetLsb(sl, false);
    vl_buf = fss::util::SetLsb(vl_buf, false);
    auto vl = Group::From(vl_buf);
    bool tr = fss::util::GetLsb(sr);
    sr = fss::util::SetLsb(sr, false);
    vr_buf = fss::util::SetLsb(vr_buf, false);
    auto vr = Group::From(vr_buf);

    if (t) {
      sl = fss::util::Xor(sl, s_cw);
      sr = fss::util::Xor(sr, s_cw);
      tl = tl ^ tl_cw;
      tr = tr ^ tr_cw;
    }

    bool x_bit = GetBit(x, in_bits - 1 - i);
    if constexpr (b) {
      v = v + (x_bit ? (-vr) : (-vl));
      if (t) v = v + (-v_cw);
    } else {
      v = v + (x_bit ? vr : vl);
      if (t) v = v + v_cw;
    }
    if (x_bit) {
      s = sr;
      t = tr;
    } else {
      s = sl;
      t = tl;
    }
  }

  auto v_out_cw = Group::From(out_cw[tid]);
  if constexpr (b) {
    v = v + (-Group::From(s));
    if (t) v = v + (-v_out_cw);
  } else {
    v = v + Group::From(s);
    if (t) v = v + v_out_cw;
  }
  ys[tid] = v.Into();
}

// Point-evaluate in_bits-level VDPF keys, one thread per key. Same seed walk
// as the DPF kernel plus the per-point verification hash; the walk carries no
// hash cost, the tail does one Blake3 and, when t, the cs correction.
// cw_s/extra come from VdpfRepackKernel. cs and ocws stay per-key: each is
// read once per evaluation, not once per level.
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename XorHash, typename In>
__global__ void VdpfEvalPointKernel(int4 *ys, int4 *pi, const int4 *seeds, const int4 *cw_s,
    const ExtBits<in_bits> *extra, const cuda::std::array<int4, 4> *cs, const int4 *ocws, const In *xs, int nkeys,
    Prg prg, XorHash xor_hash) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  bool t = b;
  In x = xs[tid];
  ExtBits<in_bits> ex = extra[tid];

#pragma unroll unroll
  for (int i = 0; i < in_bits; ++i) {
    int4 s_cw = cw_s[(size_t)i * nkeys + tid];
    bool tr_cw = GetBit(ex, i);
    bool tl_cw = fss::util::GetLsb(s_cw);
    s_cw = fss::util::SetLsb(s_cw, false);

    auto g = prg.Gen(s);
    int4 sl = g[0], sr = g[1];
    bool tl = fss::util::GetLsb(sl);
    sl = fss::util::SetLsb(sl, false);
    bool tr = fss::util::GetLsb(sr);
    sr = fss::util::SetLsb(sr, false);

    if (t) {
      sl = fss::util::Xor(sl, s_cw);
      sr = fss::util::Xor(sr, s_cw);
      tl = tl ^ tl_cw;
      tr = tr ^ tr_cw;
    }

    bool x_bit = GetBit(x, in_bits - 1 - i);
    if (x_bit) {
      s = sr;
      t = tr;
    } else {
      s = sl;
      t = tl;
    }
  }

  auto y = Group::From(s);
  if (t) y = y + Group::From(ocws[tid]);
  if constexpr (b) y = -y;
  ys[tid] = y.Into();

  int4 x_buf = util::Pack(x);
  auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, s});
  if (t) {
    pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs[tid]));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) pi[(size_t)tid * 4 + j] = pi_tilde[j];
}

// Batch-evaluate a DMPF key: one thread per input point. Each thread runs the
// kappa Cuckoo Locate calls on its own point and descends the located bucket
// keys itself, so the outputs need no atomics and no cross-thread
// synchronization. cw_s/extra/out_cw come from DpfRepackGpu run over the m_rt
// active bucket keys; a warp's bucket indices are pseudorandom, so per-level
// loads scatter inside the level's row instead of following the per-key
// layout's one-line-per-thread pattern.
template <int in_bits, int bucket_bits, int unroll, bool b, typename Group, typename Prg, typename Prp, typename In,
    int kappa>
__global__ void DmpfBatchEvalKernel(int4 *ys, const In *xs, int n, int4 sigma, int m_rt, int b_rt, const int4 *s0s,
    const int4 *cw_s, const ExtBits<bucket_bits> *extra, const int4 *out_cw, Prg prg, Prp prp) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  static_assert(bucket_bits <= 32, "the bucket point type is fixed to uint");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  In x = xs[tid];
  constexpr __uint128_t n128 = static_cast<__uint128_t>(1) << in_bits;
  cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};
  auto acc = Group::From(Zero4());
  for (int k = 0; k < kappa; ++k) {
    auto [bucket, j] = prp_hash.Locate(sigma, x, k, n128, b_rt);
    if (bucket >= m_rt) continue;

    int4 s = fss::util::SetLsb(s0s[bucket], false);
    bool t = b;
    ExtBits<bucket_bits> ex = extra[bucket];

#pragma unroll unroll
    for (int i = 0; i < bucket_bits; ++i) {
      int4 s_cw = cw_s[(size_t)i * m_rt + bucket];
      bool tr_cw = GetBit(ex, i);
      bool tl_cw = fss::util::GetLsb(s_cw);
      s_cw = fss::util::SetLsb(s_cw, false);

      auto g = prg.Gen(s);
      int4 sl = g[0], sr = g[1];
      bool tl = fss::util::GetLsb(sl);
      sl = fss::util::SetLsb(sl, false);
      bool tr = fss::util::GetLsb(sr);
      sr = fss::util::SetLsb(sr, false);

      if (t) {
        sl = fss::util::Xor(sl, s_cw);
        sr = fss::util::Xor(sr, s_cw);
        tl = tl ^ tl_cw;
        tr = tr ^ tr_cw;
      }

      bool x_bit = GetBit(j, bucket_bits - 1 - i);
      if (x_bit) {
        s = sr;
        t = tr;
      } else {
        s = sl;
        t = tl;
      }
    }

    auto y = Group::From(s);
    if (t) y = y + Group::From(out_cw[bucket]);
    if constexpr (b) y = -y;
    acc = acc + y;
  }
  ys[tid] = acc.Into();
}

}  // namespace detail

/**
 * One-time repack of per-key HalfTree DPF keys into the level-major layout
 * consumed by HalfTreeDpfEvalPointGpu. cw_s has in_bits * nkeys int4s,
 * extra and ocws have nkeys entries each.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed last-level extra bits.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename In>
void HalfTreeDpfRepackGpu(const typename HalfTreeDpf<in_bits, Group, Prg, In>::Cw *cws, int nkeys, int4 *cw_s,
    uint32_t *extra, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::HalfTreeDpfRepackKernel<in_bits, typename HalfTreeDpf<in_bits, Group, Prg, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nkeys);
}

/**
 * One-time repack of per-key DPF keys into the level-major layout consumed
 * by DpfEvalPointGpu. cw_s has in_bits * nkeys int4s; extra and out_cw have
 * nkeys entries each.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits + 1.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed tr bits, bit i of extra[k] = level i.
 * @param out_cw Output correction words cws[k][in_bits].s, one per key.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename In>
void DpfRepackGpu(const typename Dpf<in_bits, Group, Prg, In>::Cw *cws, int nkeys, int4 *cw_s,
    ExtBits<in_bits> *extra, int4 *out_cw, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DpfRepackKernel<in_bits, typename Dpf<in_bits, Group, Prg, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, out_cw, nkeys);
}

/**
 * One-time repack of per-key DCF keys into the level-major layout consumed
 * by DcfEvalPointGpu. cw_s and cw_v have in_bits * nkeys int4s each; out_cw
 * has nkeys entries.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @tparam pred DCF predicate of the keys, see DcfPred. Defaults to lt.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits + 1.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major s parts, cw_s[i * nkeys + k].
 * @param cw_v Output level-major v parts, cw_v[i * nkeys + k].
 * @param out_cw Output final correction words' v parts, one per key.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred = DcfPred::kLt>
void DcfRepackGpu(const typename Dcf<in_bits, Group, Prg, In, pred>::Cw *cws, int nkeys, int4 *cw_s, int4 *cw_v,
    int4 *out_cw, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DcfRepackKernel<in_bits, typename Dcf<in_bits, Group, Prg, In, pred>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, cw_v, out_cw, nkeys);
}

/**
 * One-time repack of per-key VDPF keys into the level-major layout consumed
 * by VdpfEvalPointGpu. cw_s has in_bits * nkeys int4s; extra has nkeys
 * entries. cs and ocws need no repack: each is read once per evaluation.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed tr bits, bit i of extra[k] = level i.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
void VdpfRepackGpu(const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw *cws, int nkeys, int4 *cw_s,
    ExtBits<in_bits> *extra, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::VdpfRepackKernel<in_bits, typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nkeys);
}

/**
 * GPU point evaluation of nkeys HalfTree DPF keys: ys[k] = Eval(b, seeds[k],
 * cws, ocws[k], xs[k]). One thread per key. cw_s/extra must come from
 * HalfTreeDpfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from HalfTreeDpfRepackGpu.
 * @param extra Packed last-level extra bits from HalfTreeDpfRepackGpu.
 * @param ocws Output correction words returned by Gen(), one per key.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A HalfTreeDpf instance (holds the PRG and hash key).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In>
void HalfTreeDpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const uint32_t *extra, const int4 *ocws,
    const In *xs, int4 *ys, int nkeys, const HalfTreeDpf<in_bits, Group, Prg, In> &dpf,
    cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, true, Group, Prg, In>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys, dpf.prg, dpf.hash_key);
  } else {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, false, Group, Prg, In>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys, dpf.prg, dpf.hash_key);
  }
}

/**
 * GPU point evaluation of nkeys DPF keys: ys[k] = Eval(b, seeds[k], cws[k],
 * xs[k]). One thread per key. cw_s/extra/out_cw must come from DpfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from DpfRepackGpu.
 * @param extra Packed tr bits from DpfRepackGpu.
 * @param out_cw Output correction words from DpfRepackGpu.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A Dpf instance (holds the PRG).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In>
void DpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra, const int4 *out_cw,
    const In *xs, int4 *ys, int nkeys, const Dpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::DpfEvalPointKernel<in_bits, unroll, true, Group, Prg, In>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, out_cw, xs, nkeys, dpf.prg);
  } else {
    detail::DpfEvalPointKernel<in_bits, unroll, false, Group, Prg, In>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, out_cw, xs, nkeys, dpf.prg);
  }
}

/**
 * GPU point evaluation of nkeys DCF keys: ys[k] = Eval(b, seeds[k], cws[k],
 * xs[k]). One thread per key. cw_s/cw_v/out_cw must come from DcfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major s parts from DcfRepackGpu.
 * @param cw_v Level-major v parts from DcfRepackGpu.
 * @param out_cw Final correction words' v parts from DcfRepackGpu.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dcf A Dcf instance (holds the PRG). Used on the host side. Any predicate and
 *   par_depth instantiation; both are deduced from the argument.
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, DcfPred pred, int par_depth>
void DcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *cw_v, const int4 *out_cw,
    const In *xs, int4 *ys, int nkeys, const Dcf<in_bits, Group, Prg, In, pred, par_depth> &dcf,
    cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::DcfEvalPointKernel<in_bits, unroll, true, Group, Prg, In>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, dcf.prg);
  } else {
    detail::DcfEvalPointKernel<in_bits, unroll, false, Group, Prg, In>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, dcf.prg);
  }
}

/**
 * GPU point evaluation of nkeys VDPF keys: ys[k] = Eval(b, seeds[k], cws[k],
 * cs[k], ocws[k], xs[k])[0] and pi[k] = its corrected verification hash. One
 * thread per key. cw_s/extra must come from VdpfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from VdpfRepackGpu.
 * @param extra Packed tr bits from VdpfRepackGpu.
 * @param cs Check seeds returned by Gen(), one per key.
 * @param ocws Output correction words returned by Gen(), one per key.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param pi Output corrected per-point hashes, four int4s per key.
 * @param nkeys Number of keys.
 * @param vdpf A Vdpf instance (holds the PRG and hashes).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
void VdpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws, const In *xs, int4 *ys, int4 *pi, int nkeys,
    const Vdpf<in_bits, Group, Prg, XorHash, Hash, In> &vdpf, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::VdpfEvalPointKernel<in_bits, unroll, true, Group, Prg, XorHash, In>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys, vdpf.prg, vdpf.xor_hash);
  } else {
    detail::VdpfEvalPointKernel<in_bits, unroll, false, Group, Prg, XorHash, In>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys, vdpf.prg, vdpf.xor_hash);
  }
}

/**
 * GPU batch evaluation of a DMPF key: ys[i] = BatchEval(b, key, xs)[i]. One
 * thread per input point; each thread runs its kappa Cuckoo Locate calls and
 * the located bucket descents itself, so no atomics and no synchronization.
 *
 * cw_s/extra/out_cw must come from DpfRepackGpu run over the m_rt active
 * bucket keys (nkeys = m_rt, in_bits = bucket_bits of the Dmpf) and s0s holds
 * party b's per-bucket seeds. The Prp backend must be device-callable:
 * ChaChaFeistel qualifies, Aes128Feistel does not.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param xs Evaluation points (device memory), n entries.
 * @param ys Output array (device memory), one int4 per point.
 * @param n Number of points.
 * @param sigma PRP seed from the key.
 * @param m_rt Runtime bucket count stored in the key.
 * @param b_rt Runtime bucket size stored in the key.
 * @param s0s Per-bucket party seeds (device memory), m_rt entries.
 * @param cw_s Level-major correction words from DpfRepackGpu.
 * @param extra Packed tr bits from DpfRepackGpu.
 * @param out_cw Output correction words from DpfRepackGpu.
 * @param dmpf A Dmpf instance (holds the PRG and PRP, passed to the kernel by
 *   value on the host side).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename Prp,
    typename In, int kappa>
void DmpfBatchEvalGpu(bool b, const In *xs, int4 *ys, int n, int4 sigma, int m_rt, int b_rt, const int4 *s0s,
    const int4 *cw_s, const ExtBits<bucket_bits> *extra, const int4 *out_cw,
    const Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, kappa> &dmpf, cudaStream_t stream = 0) {
  static_assert(std::is_trivially_copyable_v<Prp>, "the Prp backend must be device-callable, e.g. ChaChaFeistel");
  constexpr int kBs = 256;
  int blocks = (n + kBs - 1) / kBs;
  if (b) {
    detail::DmpfBatchEvalKernel<in_bits, bucket_bits, unroll, true, Group, Prg, Prp, In, kappa>
        <<<blocks, kBs, 0, stream>>>(ys, xs, n, sigma, m_rt, b_rt, s0s, cw_s, extra, out_cw, dmpf.prg, dmpf.prp);
  } else {
    detail::DmpfBatchEvalKernel<in_bits, bucket_bits, unroll, false, Group, Prg, Prp, In, kappa>
        <<<blocks, kBs, 0, stream>>>(ys, xs, n, sigma, m_rt, b_rt, s0s, cw_s, extra, out_cw, dmpf.prg, dmpf.prp);
  }
}

}  // namespace fss::gpu

#endif  // FSS_POINT_EVAL_GPU_CUH_
