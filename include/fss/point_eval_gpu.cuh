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
 * Run the matching Relayout*Gpu() once per key after Gen(); the relayout cost
 * amortizes over every later point evaluation, same as Gen itself sits outside
 * the timed loop.
 */
#ifndef FSS_POINT_EVAL_GPU_CUH_
#define FSS_POINT_EVAL_GPU_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <fss/dcf.cuh>
#include <fss/dpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/util.cuh>
#include <fss/vdpf.cuh>

namespace fss::gpu {

namespace detail {

__device__ __host__ inline int4 Zero4() { return {0, 0, 0, 0}; }

__device__ __host__ inline bool GetBit(uint32_t x, int i) { return ((x >> i) & 1) != 0; }

// Pack per-key HalfTree DPF cws (stride in_bits) into level-major cw_s and the
// last-level extra bit (bit 0 of extra[k]).
template <int in_bits, typename Cw>
__global__ void HalfTreeDpfRelayoutKernel(const Cw *pk, int4 *cw_s, uint32_t *extra, int nkeys) {
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
__global__ void DpfRelayoutKernel(const Cw *pk, int4 *cw_s, uint32_t *extra, int4 *out_cw, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  uint32_t e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].s;
    if (pk[(size_t)k * (in_bits + 1) + i].tr) e |= 1u << i;
  }
  extra[k] = e;
  out_cw[k] = pk[(size_t)k * (in_bits + 1) + in_bits].s;
}

// Pack per-key VDPF cws (stride in_bits) into level-major cw_s and the tr
// bits (bit i of extra[k] = level i).
template <int in_bits, typename Cw>
__global__ void VdpfRelayoutKernel(const Cw *pk, int4 *cw_s, uint32_t *extra, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  uint32_t e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * in_bits + i].s;
    if (pk[(size_t)k * in_bits + i].tr) e |= 1u << i;
  }
  extra[k] = e;
}

// Pack per-key DCF cws (stride in_bits + 1) into level-major cw_s and cw_v and
// the final correction word's v part.
template <int in_bits, typename Cw>
__global__ void DcfRelayoutKernel(const Cw *pk, int4 *cw_s, int4 *cw_v, int4 *out_cw, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].s;
    cw_v[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].v;
  }
  out_cw[k] = pk[(size_t)k * (in_bits + 1) + in_bits].v;
}

// Point-evaluate in_bits-level HalfTree DPF keys: one thread per key, one
// output value per key at xs[tid]. cw_s/extra come from HalfTreeDpfRelayoutKernel.
template <int in_bits, int unroll, bool b, typename Group, typename Prg>
__global__ void HalfTreeDpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s,
    const uint32_t *extra, const int4 *ocws, const uint32_t *xs, int nkeys, Prg prg, int4 hash_key) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  uint32_t x = xs[tid];

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
// come from DpfRelayoutKernel.
template <int in_bits, int unroll, bool b, typename Group, typename Prg>
__global__ void DpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const uint32_t *extra,
    const int4 *out_cw, const uint32_t *xs, int nkeys, Prg prg) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  bool t = b;
  uint32_t x = xs[tid];
  uint32_t ex = extra[tid];

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
// come from DcfRelayoutKernel. v starts at zero, which is the value the
// library Eval relies on the compiler to produce for its uninitialized v.
template <int in_bits, int unroll, bool b, typename Group, typename Prg>
__global__ void DcfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const int4 *cw_v,
    const int4 *out_cw, const uint32_t *xs, int nkeys, Prg prg) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  Group v = Group::From(Zero4());
  bool t = b;
  uint32_t x = xs[tid];

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
// cw_s/extra come from VdpfRelayoutKernel. cs and ocws stay per-key: each is
// read once per evaluation, not once per level.
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename XorHash>
__global__ void VdpfEvalPointKernel(int4 *ys, int4 *pi, const int4 *seeds, const int4 *cw_s, const uint32_t *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws, const uint32_t *xs, int nkeys, Prg prg,
    XorHash xor_hash) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  bool t = b;
  uint32_t x = xs[tid];
  uint32_t ex = extra[tid];

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

}  // namespace detail

/**
 * One-time relayout of per-key HalfTree DPF keys into the level-major layout
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
void HalfTreeDpfRelayoutGpu(const typename HalfTreeDpf<in_bits, Group, Prg, In>::Cw *cws, int nkeys, int4 *cw_s,
    uint32_t *extra, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::HalfTreeDpfRelayoutKernel<in_bits, typename HalfTreeDpf<in_bits, Group, Prg, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nkeys);
}

/**
 * One-time relayout of per-key DPF keys into the level-major layout consumed
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
void DpfRelayoutGpu(const typename Dpf<in_bits, Group, Prg, In>::Cw *cws, int nkeys, int4 *cw_s, uint32_t *extra,
    int4 *out_cw, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DpfRelayoutKernel<in_bits, typename Dpf<in_bits, Group, Prg, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, out_cw, nkeys);
}

/**
 * One-time relayout of per-key DCF keys into the level-major layout consumed
 * by DcfEvalPointGpu. cw_s and cw_v have in_bits * nkeys int4s each; out_cw
 * has nkeys entries.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits + 1.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major s parts, cw_s[i * nkeys + k].
 * @param cw_v Output level-major v parts, cw_v[i * nkeys + k].
 * @param out_cw Output final correction words' v parts, one per key.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename In>
void DcfRelayoutGpu(const typename Dcf<in_bits, Group, Prg, In>::Cw *cws, int nkeys, int4 *cw_s, int4 *cw_v,
    int4 *out_cw, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DcfRelayoutKernel<in_bits, typename Dcf<in_bits, Group, Prg, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, cw_v, out_cw, nkeys);
}

/**
 * One-time relayout of per-key VDPF keys into the level-major layout consumed
 * by VdpfEvalPointGpu. cw_s has in_bits * nkeys int4s; extra has nkeys
 * entries. cs and ocws need no relayout: each is read once per evaluation.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed tr bits, bit i of extra[k] = level i.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
void VdpfRelayoutGpu(const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw *cws, int nkeys, int4 *cw_s,
    uint32_t *extra, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::VdpfRelayoutKernel<in_bits, typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nkeys);
}

/**
 * GPU point evaluation of nkeys HalfTree DPF keys: ys[k] = Eval(b, seeds[k],
 * cws, ocws[k], xs[k]). One thread per key. cw_s/extra must come from
 * HalfTreeDpfRelayoutGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from HalfTreeDpfRelayoutGpu.
 * @param extra Packed last-level extra bits from HalfTreeDpfRelayoutGpu.
 * @param ocws Output correction words returned by Gen(), one per key.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A HalfTreeDpf instance (holds the PRG and hash key).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In>
void HalfTreeDpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const uint32_t *extra, const int4 *ocws,
    const uint32_t *xs, int4 *ys, int nkeys, const HalfTreeDpf<in_bits, Group, Prg, In> &dpf,
    cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, true, Group, Prg>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys, dpf.prg, dpf.hash_key);
  } else {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, false, Group, Prg>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys, dpf.prg, dpf.hash_key);
  }
}

/**
 * GPU point evaluation of nkeys DPF keys: ys[k] = Eval(b, seeds[k], cws[k],
 * xs[k]). One thread per key. cw_s/extra/out_cw must come from DpfRelayoutGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from DpfRelayoutGpu.
 * @param extra Packed tr bits from DpfRelayoutGpu.
 * @param out_cw Output correction words from DpfRelayoutGpu.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A Dpf instance (holds the PRG).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In>
void DpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const uint32_t *extra, const int4 *out_cw,
    const uint32_t *xs, int4 *ys, int nkeys, const Dpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::DpfEvalPointKernel<in_bits, unroll, true, Group, Prg>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, out_cw, xs, nkeys, dpf.prg);
  } else {
    detail::DpfEvalPointKernel<in_bits, unroll, false, Group, Prg>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, out_cw, xs, nkeys, dpf.prg);
  }
}

/**
 * GPU point evaluation of nkeys DCF keys: ys[k] = Eval(b, seeds[k], cws[k],
 * xs[k]). One thread per key. cw_s/cw_v/out_cw must come from DcfRelayoutGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major s parts from DcfRelayoutGpu.
 * @param cw_v Level-major v parts from DcfRelayoutGpu.
 * @param out_cw Final correction words' v parts from DcfRelayoutGpu.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A Dcf instance (holds the PRG).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In>
void DcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *cw_v, const int4 *out_cw,
    const uint32_t *xs, int4 *ys, int nkeys, const Dcf<in_bits, Group, Prg, In> &dcf, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::DcfEvalPointKernel<in_bits, unroll, true, Group, Prg>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, dcf.prg);
  } else {
    detail::DcfEvalPointKernel<in_bits, unroll, false, Group, Prg>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, dcf.prg);
  }
}

/**
 * GPU point evaluation of nkeys VDPF keys: ys[k] = Eval(b, seeds[k], cws[k],
 * cs[k], ocws[k], xs[k])[0] and pi[k] = its corrected verification hash. One
 * thread per key. cw_s/extra must come from VdpfRelayoutGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from VdpfRelayoutGpu.
 * @param extra Packed tr bits from VdpfRelayoutGpu.
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
void VdpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const uint32_t *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws, const uint32_t *xs, int4 *ys, int4 *pi, int nkeys,
    const Vdpf<in_bits, Group, Prg, XorHash, Hash, In> &vdpf, cudaStream_t stream = 0) {
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::VdpfEvalPointKernel<in_bits, unroll, true, Group, Prg, XorHash>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys, vdpf.prg, vdpf.xor_hash);
  } else {
    detail::VdpfEvalPointKernel<in_bits, unroll, false, Group, Prg, XorHash>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys, vdpf.prg, vdpf.xor_hash);
  }
}

}  // namespace fss::gpu

#endif  // FSS_POINT_EVAL_GPU_CUH_
