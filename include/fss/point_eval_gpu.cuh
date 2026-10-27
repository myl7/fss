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
 * once in_bits passes 32 or 64. Released uint32_t query arrays remain
 * supported for domains of at most 32 bits, even when In is narrower. Each
 * uint32_t element is loaded at its own stride and converted to In.
 */
#ifndef FSS_POINT_EVAL_GPU_CUH_
#define FSS_POINT_EVAL_GPU_CUH_

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/array>
#include <fss/cuckoo_hash.cuh>
#include <fss/dcf.cuh>
#include <fss/dmpf.cuh>
#include <fss/dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/util.cuh>
#include <fss/vdpf.cuh>
#include <fss/vdmpf.cuh>
#include <fss/internal/gpu_workspace.cuh>

namespace fss::gpu {

// The narrowest unsigned primitive holding one tr bit per level, in_bits bits
// in total. Common in_bits <= 32 keys keep the uint32_t layout, so existing
// callers see no change.
template <int bits>
using ExtBits = std::conditional_t<bits <= 32, uint32_t, std::conditional_t<bits <= 64, uint64_t, __uint128_t>>;

namespace detail {

__device__ __host__ inline int4 Zero4() {
  return {0, 0, 0, 0};
}

template <typename T>
__device__ __host__ inline bool GetBit(T x, int i) {
  return ((x >> i) & 1) != 0;
}

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
// (bit i of extra[k] = level i), and the output correction word unless out_cw
// is null, as for Grotto DCF keys, which leave it unused.
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
  if (out_cw != nullptr) out_cw[k] = pk[(size_t)k * (in_bits + 1) + in_bits].s;
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
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename In, typename Query = In>
__global__ void HalfTreeDpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const uint32_t *extra,
    const int4 *ocws, const Query *xs, int nkeys, Prg prg, int4 hash_key) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = static_cast<In>(xs[tid]);

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

// Pack per-key Packed HalfTree DPF cws (stride depth) into level-major cw_s.
template <int depth, typename Cw>
__global__ void PackedHalfTreeDpfRepackKernel(const Cw *pk, int4 *cw_s, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  for (int i = 0; i < depth; ++i) cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * depth + i].s;
}

// Point-evaluate Packed HalfTree DPF keys, one thread per key. ys[tid] is x's output block. cw_s
// comes from PackedHalfTreeDpfRepackKernel.
template <int in_bits, int out_bits, int unroll, bool b, typename Prg, typename In, int par_depth, typename Query>
__global__ void PackedHalfTreeDpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const int4 *fcws,
    const Query *xs, int nkeys, Prg prg, int4 hash_key) {
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Prg, In, par_depth>;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = static_cast<In>(xs[tid]);

#pragma unroll unroll
  for (int i = 0; i < Scheme::kDepth; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    bool t = fss::util::GetLsb(node);
    int4 h = prg.Gen(fss::util::Xor(hash_key, node))[0];
    int4 cw = cw_s[(size_t)i * nkeys + tid];
    node = fss::util::Xor(fss::util::Xor(h, x_bit ? node : Zero4()), t ? cw : Zero4());
  }

  int4 leaf = prg.Gen(fss::util::Xor(hash_key, node))[0];
  if (fss::util::GetLsb(node)) leaf = fss::util::Xor(leaf, fcws[tid]);
  ys[tid] = leaf;
}

// Point-evaluate in_bits-level DPF keys, one thread per key. ys[tid] is the
// output, or with cmp the 1[alpha <= x] share of GrottoDcf::Eval(b, key, x).
// cw_s/extra/out_cw come from DpfRepackKernel. out_cw is unused with cmp.
template <int in_bits, int unroll, bool b, bool cmp, typename Group, typename Prg, typename In, typename Query,
    typename Y>
__global__ void DpfEvalPointKernel(Y *ys, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const int4 *out_cw, const Query *xs, int nkeys, Prg prg) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  bool t = b;
  In x = static_cast<In>(xs[tid]);
  ExtBits<in_bits> ex = extra[tid];
  fss::detail::GrottoTurns turns;

#pragma unroll unroll
  for (int i = 0; i < in_bits; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    if constexpr (cmp) turns(t, x_bit);

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

    if (x_bit) {
      s = sr;
      t = tr;
    } else {
      s = sl;
      t = tl;
    }
  }

  if constexpr (cmp) {
    ys[tid] = turns.PointLeaf(t);
  } else {
    auto y = Group::From(s);
    if (t) y = y + Group::From(out_cw[tid]);
    if constexpr (b) y = -y;
    ys[tid] = y.Into();
  }
}

template <int unroll, bool cmp, int in_bits, typename Group, typename Prg, typename In, typename Query, typename Y>
void DpfEvalPointLaunch(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const int4 *out_cw, const Query *xs, Y *ys, int nkeys, Prg prg, cudaStream_t stream) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    DpfEvalPointKernel<in_bits, unroll, true, cmp, Group, Prg, In, Query, Y>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, out_cw, xs, nkeys, prg);
  } else {
    DpfEvalPointKernel<in_bits, unroll, false, cmp, Group, Prg, In, Query, Y>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, out_cw, xs, nkeys, prg);
  }
}

// Point-evaluate in_bits-level DCF keys, one thread per key. cw_s/cw_v/out_cw
// come from DcfRepackKernel. v starts at zero, which is the value the
// library Eval relies on the compiler to produce for its uninitialized v.
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename In, typename Query = In>
__global__ void DcfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const int4 *cw_v, const int4 *out_cw,
    const Query *xs, int nkeys, Prg prg) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  Group v = Group::From(Zero4());
  bool t = b;
  In x = static_cast<In>(xs[tid]);

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
template <int in_bits, int unroll, bool b, typename Group, typename Prg, typename XorHash, typename In,
    typename Query = In>
__global__ void VdpfEvalPointKernel(int4 *ys, int4 *pi, const int4 *seeds, const int4 *cw_s,
    const ExtBits<in_bits> *extra, const cuda::std::array<int4, 4> *cs, const int4 *ocws, const Query *xs, int nkeys,
    Prg prg, XorHash xor_hash) {
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 s = fss::util::SetLsb(seeds[tid], false);
  bool t = b;
  In x = static_cast<In>(xs[tid]);
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

// Per-(point, hash) Cuckoo Locate records for VDMPF batch evaluation. keys
// holds the bucket and vals packs (contribution index << 32 | within-bucket
// point); a stable radix sort later groups the records by bucket while keeping
// the (point, hash) order inside each bucket, which is exactly the canonical
// order the host BatchEvalMerkle folds its per-bucket leaves in.
template <int in_bits, typename Prp, typename In, int kappa>
__global__ void VdmpfLocateKernel(
    const In *xs, int n, int4 sigma, int m_rt, int b_rt, uint32_t *keys, uint64_t *vals, uint32_t *counts, Prp prp) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  In x = xs[tid];
  constexpr __uint128_t n128 = static_cast<__uint128_t>(1) << in_bits;
  cuckoo_hash::PrpHash<Prp, In, kappa> prp_hash{prp};
  for (int k = 0; k < kappa; ++k) {
    auto [bucket, j] = prp_hash.Locate(sigma, x, k, n128, b_rt);
    size_t idx = static_cast<size_t>(tid) * kappa + k;
    keys[idx] = static_cast<uint32_t>(bucket);
    vals[idx] = (static_cast<uint64_t>(idx) << 32) | static_cast<uint32_t>(j);
    if (bucket < m_rt) atomicAdd(&counts[bucket], 1u);
  }
}

// Batch-evaluate a VDMPF key: one thread per input point computes its kappa
// contributions (from the Locate records) and writes each corrected per-point
// hash pi_tilde at its contribution index. cw_s/extra come from VdpfRepackGpu
// over the m_rt active bucket keys; cs and ocws are indexed by bucket.
template <int bucket_bits, int unroll, bool b, typename Group, typename Prg, typename XorHash, typename In, int kappa>
__global__ void VdmpfBatchEvalKernel(int4 *ys, const In *xs, int n, int m_rt, const int4 *s0s, const int4 *cw_s,
    const ExtBits<bucket_bits> *extra, const cuda::std::array<int4, 4> *cs, const int4 *ocws,
    cuda::std::array<int4, 4> *pi_tilde_buf, const uint32_t *keys, const uint64_t *vals, Prg prg, XorHash xor_hash) {
  static_assert(bucket_bits <= 32, "the bucket point type is fixed to uint");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  auto acc = Group::From(Zero4());
  for (int k = 0; k < kappa; ++k) {
    size_t idx = static_cast<size_t>(tid) * kappa + k;
    int bucket = static_cast<int>(keys[idx]);
    if (bucket >= m_rt) continue;
    uint j = static_cast<uint>(static_cast<uint32_t>(vals[idx]));

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
    if (t) y = y + Group::From(ocws[bucket]);
    if constexpr (b) y = -y;
    acc = acc + y;

    int4 x_buf = fss::util::Pack(j);
    auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, s});
    if (t) {
      cuda::std::array<int4, 4> corrected = pi_tilde;
      const cuda::std::array<int4, 4> &bucket_cs = cs[bucket];
#pragma unroll
      for (int w = 0; w < 4; ++w) corrected[w] = fss::util::Xor(corrected[w], bucket_cs[w]);
      pi_tilde = corrected;
    }
    pi_tilde_buf[idx] = pi_tilde;
  }
  ys[tid] = acc.Into();
}

// Prefix tables for the Merkle-fold kernels: per-bucket leaf offsets and the
// per-bucket chunk-count prefix, both with an m+1 sentinel end. One thread;
// m is tiny.
__global__ void VdmpfFoldSetupKernel(
    uint32_t *leaf_offsets, uint32_t *chunk_offsets, const uint32_t *counts, int m_buckets, int chunk_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t leaves = 0, chunks = 0;
  for (int i = 0; i < m_buckets; ++i) {
    leaf_offsets[i] = leaves;
    chunk_offsets[i] = chunks;
    uint32_t count = counts[i];
    leaves += count;
    chunks += (count + chunk_size - 1) / chunk_size;
  }
  leaf_offsets[m_buckets] = leaves;
  chunk_offsets[m_buckets] = chunks;
}

// Fold one kMerkleChunkSize-leaf chunk of the sorted Locate records into a
// chunk root. One block per chunk: the leaves (hashes of the corrected
// per-point hashes, gathered through the sorted records) fold pairwise level
// by level with an odd trailing node carrying unchanged - the same rule as the
// host FoldNodes - into double-buffered shared nodes. The launcher
// over-provisions blocks; the ones past the chunk table's end leave.
template <int chunk_size, typename Hash>
__global__ void VdmpfChunkFoldKernel(cuda::std::array<int4, 2> *chunk_roots, const uint32_t *chunk_offsets,
    const uint32_t *leaf_offsets, const uint32_t *counts, const uint64_t *sorted_vals,
    const cuda::std::array<int4, 4> *pi_tilde_buf, int m_buckets, Hash hash) {
  __shared__ cuda::std::array<int4, 2> s_a[chunk_size];
  __shared__ cuda::std::array<int4, 2> s_b[chunk_size];
  uint32_t chunk = blockIdx.x;
  if (chunk >= chunk_offsets[m_buckets]) return;

  // Binary search the bucket whose chunk range covers this chunk.
  int lo = 0, hi = m_buckets - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (chunk < chunk_offsets[mid + 1]) hi = mid;
    else lo = mid + 1;
  }
  int bucket = lo;
  uint32_t local = chunk - chunk_offsets[bucket];

  uint32_t leaf_base = leaf_offsets[bucket] + local * chunk_size;
  uint32_t take = counts[bucket] - local * chunk_size;
  if (take > static_cast<uint32_t>(chunk_size)) take = chunk_size;

  cuda::std::array<int4, 2> *cur = s_a;
  cuda::std::array<int4, 2> *next = s_b;
  if (threadIdx.x < take) {
    uint32_t p = leaf_base + threadIdx.x;
    uint32_t idx = static_cast<uint32_t>(sorted_vals[p] >> 32);
    cur[threadIdx.x] = hash.Hash(cuda::std::span<const int4, 4>(pi_tilde_buf[idx]));
  }
  __syncthreads();

  uint32_t count = take;
  while (count > 1) {
    uint32_t pairs = count / 2;
    if (threadIdx.x < pairs) {
      int4 msg[4] = {
          cur[2 * threadIdx.x][0], cur[2 * threadIdx.x][1], cur[2 * threadIdx.x + 1][0], cur[2 * threadIdx.x + 1][1]};
      next[threadIdx.x] = hash.Hash(cuda::std::span<const int4, 4>(msg));
    }
    if (count % 2 != 0 && threadIdx.x == pairs) {
      next[pairs] = cur[count - 1];
    }
    __syncthreads();
    cuda::std::array<int4, 2> *swap = cur;
    cur = next;
    next = swap;
    count = pairs + count % 2;
  }
  if (threadIdx.x == 0) chunk_roots[chunk] = cur[0];
}

// Combine the chunk roots into per-bucket nodes and the bucket nodes into the
// final proof. One block: every bucket thread left-chains its chunk roots
// starting from the bucket's hashed cs, then the m bucket nodes fold level by
// level with the same odd-carry rule as the chunks.
template <int m_buckets, typename Hash>
__global__ void VdmpfBucketFoldKernel(cuda::std::array<int4, 4> *pi, const cuda::std::array<int4, 2> *chunk_roots,
    const uint32_t *chunk_offsets, const cuda::std::array<int4, 4> *cs, Hash hash) {
  __shared__ cuda::std::array<int4, 2> s_a[m_buckets];
  __shared__ cuda::std::array<int4, 2> s_b[m_buckets];
  int i = threadIdx.x;
  if (i < m_buckets) {
    cuda::std::array<int4, 2> node = hash.Hash(cuda::std::span<const int4, 4>(cs[i]));
    for (uint32_t q = chunk_offsets == nullptr ? 0 : chunk_offsets[i];
        chunk_offsets != nullptr && q < chunk_offsets[i + 1]; ++q) {
      int4 msg[4] = {node[0], node[1], chunk_roots[q][0], chunk_roots[q][1]};
      node = hash.Hash(cuda::std::span<const int4, 4>(msg));
    }
    s_a[i] = node;
  }
  __syncthreads();

  cuda::std::array<int4, 2> *cur = s_a;
  cuda::std::array<int4, 2> *next = s_b;
  int count = m_buckets;
  while (count > 1) {
    int pairs = count / 2;
    if (i < pairs) {
      int4 msg[4] = {cur[2 * i][0], cur[2 * i][1], cur[2 * i + 1][0], cur[2 * i + 1][1]};
      next[i] = hash.Hash(cuda::std::span<const int4, 4>(msg));
    }
    if (count % 2 != 0 && i == pairs) {
      next[pairs] = cur[count - 1];
    }
    __syncthreads();
    cuda::std::array<int4, 2> *swap = cur;
    cur = next;
    next = swap;
    count = pairs + count % 2;
  }
  if (i == 0) *pi = {cur[0][0], cur[0][1], int4{0, 0, 0, 0}, int4{0, 0, 0, 0}};
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
template <int in_bits, typename Group, typename Prg, typename In, int par_depth = -1>
void HalfTreeDpfRepackGpu(const typename HalfTreeDpf<in_bits, Group, Prg, In, par_depth>::Cw *cws, int nkeys,
    int4 *cw_s, uint32_t *extra, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::HalfTreeDpfRepackKernel<in_bits, typename HalfTreeDpf<in_bits, Group, Prg, In, par_depth>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nkeys);
}

/**
 * One-time repack of per-key Packed HalfTree DPF keys into the level-major
 * layout consumed by PackedHalfTreeDpfEvalPointGpu. cw_s has kDepth * nkeys
 * int4s. fcws need no repack: each is read once per evaluation.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @tparam out_bits Output width of the keys.
 * @param cws Correction words returned by Gen(), key-major, stride kDepth.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, int out_bits, typename Prg, typename In, int par_depth = -1>
void PackedHalfTreeDpfRepackGpu(const typename PackedHalfTreeDpf<in_bits, out_bits, Prg, In, par_depth>::Cw *cws,
    int nkeys, int4 *cw_s, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Prg, In, par_depth>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::PackedHalfTreeDpfRepackKernel<Scheme::kDepth, typename Scheme::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, nkeys);
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
template <int in_bits, typename Group, typename Prg, typename In, int par_depth = -1>
void DpfRepackGpu(const typename Dpf<in_bits, Group, Prg, In, par_depth>::Cw *cws, int nkeys, int4 *cw_s,
    ExtBits<in_bits> *extra, int4 *out_cw, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DpfRepackKernel<in_bits, typename Dpf<in_bits, Group, Prg, In, par_depth>::Cw>
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
template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred = DcfPred::kLt, int par_depth = -1>
void DcfRepackGpu(const typename Dcf<in_bits, Group, Prg, In, pred, par_depth>::Cw *cws, int nkeys, int4 *cw_s,
    int4 *cw_v, int4 *out_cw, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DcfRepackKernel<in_bits, typename Dcf<in_bits, Group, Prg, In, pred, par_depth>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, cw_v, out_cw, nkeys);
}

/**
 * One-time repack of per-key Grotto DCF keys into the level-major layout
 * consumed by GrottoDcfEvalPointGpu. cw_s has in_bits * nkeys int4s and extra
 * has nkeys entries.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits + 1.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed tr bits, bit i of extra[k] = level i.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Prg, typename In, int par_depth = -1>
void GrottoDcfRepackGpu(const typename GrottoDcf<in_bits, Prg, In, par_depth>::Cw *cws, int nkeys, int4 *cw_s,
    ExtBits<in_bits> *extra, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DpfRepackKernel<in_bits, typename GrottoDcf<in_bits, Prg, In, par_depth>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nullptr, nkeys);
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
template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In, int par_depth = -1>
void VdpfRepackGpu(const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In, par_depth>::Cw *cws, int nkeys,
    int4 *cw_s, ExtBits<in_bits> *extra, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::VdpfRepackKernel<in_bits, typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In, par_depth>::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nkeys);
}

// Released names remain available with their original packed-control layout.
template <int in_bits, typename Group, typename Prg, typename In, int par_depth = -1>
void HalfTreeDpfRelayoutGpu(const typename HalfTreeDpf<in_bits, Group, Prg, In, par_depth>::Cw *cws, int nkeys,
    int4 *cw_s, uint32_t *extra, cudaStream_t stream = 0) {
  HalfTreeDpfRepackGpu<in_bits, Group, Prg, In, par_depth>(cws, nkeys, cw_s, extra, stream);
}

template <int in_bits, typename Group, typename Prg, typename In, int par_depth = -1>
void DpfRelayoutGpu(const typename Dpf<in_bits, Group, Prg, In, par_depth>::Cw *cws, int nkeys, int4 *cw_s,
    uint32_t *extra, int4 *out_cw, cudaStream_t stream = 0) {
  static_assert(in_bits <= 32, "released packed control words support at most 32 levels");
  DpfRepackGpu<in_bits, Group, Prg, In, par_depth>(cws, nkeys, cw_s, extra, out_cw, stream);
}

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred = DcfPred::kLt, int par_depth = -1>
void DcfRelayoutGpu(const typename Dcf<in_bits, Group, Prg, In, pred, par_depth>::Cw *cws, int nkeys, int4 *cw_s,
    int4 *cw_v, int4 *out_cw, cudaStream_t stream = 0) {
  DcfRepackGpu<in_bits, Group, Prg, In, pred, par_depth>(cws, nkeys, cw_s, cw_v, out_cw, stream);
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In, int par_depth = -1>
void VdpfRelayoutGpu(const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In, par_depth>::Cw *cws, int nkeys,
    int4 *cw_s, uint32_t *extra, cudaStream_t stream = 0) {
  static_assert(in_bits <= 32, "released packed control words support at most 32 levels");
  VdpfRepackGpu<in_bits, Group, Prg, XorHash, Hash, In, par_depth>(cws, nkeys, cw_s, extra, stream);
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
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, int par_depth = -1,
    typename Query = In>
  requires(std::is_same_v<Query, In> || (in_bits <= 32 && std::is_same_v<Query, uint32_t>))
void HalfTreeDpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const uint32_t *extra, const int4 *ocws,
    const Query *xs, int4 *ys, int nkeys, const HalfTreeDpf<in_bits, Group, Prg, In, par_depth> &dpf,
    cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, true, Group, Prg, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys, dpf.prg, dpf.hash_key);
  } else {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, false, Group, Prg, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys, dpf.prg, dpf.hash_key);
  }
}

/**
 * GPU point evaluation of nkeys Packed HalfTree DPF keys: ys[k] = Eval(b,
 * key[k], xs[k]), the output block containing xs[k]. Extract() reads the lane.
 * One thread per key. cw_s must come from PackedHalfTreeDpfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from PackedHalfTreeDpfRepackGpu.
 * @param fcws Final correction words returned by Gen(), one per key.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A PackedHalfTreeDpf instance (holds the PRG and hash key).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, int out_bits, typename Prg, typename In, int par_depth = -1, typename Query = In>
  requires(std::is_same_v<Query, In> || (in_bits <= 32 && std::is_same_v<Query, uint32_t>))
void PackedHalfTreeDpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *fcws, const Query *xs,
    int4 *ys, int nkeys, const PackedHalfTreeDpf<in_bits, out_bits, Prg, In, par_depth> &dpf, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::PackedHalfTreeDpfEvalPointKernel<in_bits, out_bits, unroll, true, Prg, In, par_depth, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, fcws, xs, nkeys, dpf.prg, dpf.hash_key);
  } else {
    detail::PackedHalfTreeDpfEvalPointKernel<in_bits, out_bits, unroll, false, Prg, In, par_depth, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, fcws, xs, nkeys, dpf.prg, dpf.hash_key);
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
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, int par_depth = -1,
    typename Query = In>
  requires(std::is_same_v<Query, In> || (in_bits <= 32 && std::is_same_v<Query, uint32_t>))
void DpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra, const int4 *out_cw,
    const Query *xs, int4 *ys, int nkeys, const Dpf<in_bits, Group, Prg, In, par_depth> &dpf, cudaStream_t stream = 0) {
  detail::DpfEvalPointLaunch<unroll, false, in_bits, Group, Prg, In>(
      b, seeds, cw_s, extra, out_cw, xs, ys, nkeys, dpf.prg, stream);
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
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, DcfPred pred, int par_depth,
    typename Query = In>
  requires(std::is_same_v<Query, In> || (in_bits <= 32 && std::is_same_v<Query, uint32_t>))
void DcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *cw_v, const int4 *out_cw, const Query *xs,
    int4 *ys, int nkeys, const Dcf<in_bits, Group, Prg, In, pred, par_depth> &dcf, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::DcfEvalPointKernel<in_bits, unroll, true, Group, Prg, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, dcf.prg);
  } else {
    detail::DcfEvalPointKernel<in_bits, unroll, false, Group, Prg, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, dcf.prg);
  }
}

/**
 * GPU single-path evaluation of nkeys Grotto DCF keys: ys[k] = Eval(b, key[k],
 * xs[k]), the party's share of 1[alpha <= xs[k]]. One thread per key, with
 * in_bits PRG calls and no preprocessing. cw_s/extra must come from
 * GrottoDcfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from GrottoDcfRepackGpu.
 * @param extra Packed tr bits from GrottoDcfRepackGpu.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one share bit per key.
 * @param nkeys Number of keys.
 * @param dcf A GrottoDcf instance (holds the PRG).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Prg, typename In, int par_depth = -1, typename Query = In>
  requires(std::is_same_v<Query, In> || (in_bits <= 32 && std::is_same_v<Query, uint32_t>))
void GrottoDcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const Query *xs, bool *ys, int nkeys, const GrottoDcf<in_bits, Prg, In, par_depth> &dcf, cudaStream_t stream = 0) {
  detail::DpfEvalPointLaunch<unroll, true, in_bits, fss::group::Bytes, Prg, In>(
      b, seeds, cw_s, extra, nullptr, xs, ys, nkeys, dcf.prg, stream);
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
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In,
    int par_depth = -1, typename Query = In>
  requires(std::is_same_v<Query, In> || (in_bits <= 32 && std::is_same_v<Query, uint32_t>))
void VdpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws, const Query *xs, int4 *ys, int4 *pi, int nkeys,
    const Vdpf<in_bits, Group, Prg, XorHash, Hash, In, par_depth> &vdpf, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::VdpfEvalPointKernel<in_bits, unroll, true, Group, Prg, XorHash, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys, vdpf.prg, vdpf.xor_hash);
  } else {
    detail::VdpfEvalPointKernel<in_bits, unroll, false, Group, Prg, XorHash, In, Query>
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
    typename In, int kappa, int ch_lambda, int par_depth>
void DmpfBatchEvalGpu(bool b, const In *xs, int4 *ys, int n, int4 sigma, int m_rt, int b_rt, const int4 *s0s,
    const int4 *cw_s, const ExtBits<bucket_bits> *extra, const int4 *out_cw,
    const Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, kappa, ch_lambda, par_depth> &dmpf,
    cudaStream_t stream = 0) {
  if (n == 0) return;
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

/**
 * GPU batch evaluation of a VDMPF key with the Merkle-fold proof:
 * ys[i] = BatchEvalMerkle(b, key, xs)[i] and d_pi receives the proof. Launches
 * on the stream: a Locate pass writes per-(point, hash) Cuckoo records, a
 * stable CUB radix sort groups them by bucket in the same canonical order the
 * host BatchEvalMerkle walks its per-bucket lists, an evaluation pass computes
 * every point's kappa contributions and corrected per-point hashes, and the
 * fold passes reduce the leaves chunk-wise, per bucket from the hashed cs, and
 * across the m buckets. The proof is bit-exact with the host
 * BatchEvalMerkle over the same construction.
 *
 * cw_s/extra must come from VdpfRepackGpu run over the m_rt active bucket keys
 * (nkeys = m_rt, in_bits = bucket_bits of the Vdmpf), s0s and ocws hold party
 * b's per-bucket seeds and output correction words (m_rt entries), and cs
 * holds the shared per-bucket check seeds for all m buckets - the fold covers
 * the buckets past m_rt as well. The Prp backend must be device-callable:
 * ChaChaFeistel qualifies, Aes128Feistel does not.
 *
 * The Locate records, sorted copies, bucket counts, per-contribution hashes,
 * chunk roots, and CUB temporary storage are cached per host thread and CUDA
 * device. Calls from one thread on different streams must not overlap.
 * ReleaseGpuWorkspaces completes outstanding work and releases this thread's
 * caches. Call it before cudaDeviceReset. Thread exit also releases scratch.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @param b Party index. False for 0 and true for 1.
 * @param xs Evaluation points (device memory), n entries.
 * @param ys Output array (device memory), one int4 per point.
 * @param d_pi Proof output (device memory), four int4s.
 * @param n Number of points.
 * @param sigma PRP seed from the key.
 * @param m_rt Runtime bucket count stored in the key.
 * @param b_rt Runtime bucket size stored in the key.
 * @param s0s Per-bucket party seeds (device memory), m_rt entries.
 * @param cw_s Level-major correction words from VdpfRepackGpu.
 * @param extra Packed tr bits from VdpfRepackGpu.
 * @param cs Per-bucket check seeds (device memory), m entries, shared by both
 *   parties.
 * @param ocws Per-bucket output correction words (device memory), m_rt entries.
 * @param vdmpf A Vdmpf instance (holds the PRG, hashes, and PRP).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash,
    typename Hash, typename Prp, typename In, int kappa, int ch_lambda>
void VdmpfBatchEvalMerkleGpu(bool b, const In *xs, int4 *ys, cuda::std::array<int4, 4> *d_pi, int n, int4 sigma,
    int m_rt, int b_rt, const int4 *s0s, const int4 *cw_s, const ExtBits<bucket_bits> *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws,
    const Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, kappa, ch_lambda> &vdmpf,
    cudaStream_t stream = 0) {
  using VdmpfT = Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, kappa, ch_lambda>;
  static_assert(std::is_trivially_copyable_v<Prp>, "the Prp backend must be device-callable, e.g. ChaChaFeistel");
  static_assert(VdmpfT::m < (1 << 16), "the radix sort covers the bucket keys");
  static_assert(2 * VdmpfT::m * 32 <= 46 * 1024, "the bucket-fold kernel double-buffers every bucket node");
  const size_t total = static_cast<size_t>(n) * kappa;
  constexpr int kChunk = VdmpfT::kMerkleChunkSize;

  constexpr int kFoldThreads = (VdmpfT::m + 31) / 32 * 32;
  if (n == 0) {
    detail::VdmpfBucketFoldKernel<VdmpfT::m, Hash>
        <<<1, kFoldThreads, 0, stream>>>(d_pi, nullptr, nullptr, cs, vdmpf.hash);
    return;
  }
  static const int kWorkspaceTag = 0;
  auto &workspace = detail::GetGpuWorkspaceCache().Get(&kWorkspaceTag);
  auto *d_keys = workspace.Ensure<uint32_t>(0, total);
  auto *d_keys_out = workspace.Ensure<uint32_t>(1, total);
  auto *d_vals = workspace.Ensure<uint64_t>(2, total);
  auto *d_vals_out = workspace.Ensure<uint64_t>(3, total);
  auto *d_pi_tilde = workspace.Ensure<cuda::std::array<int4, 4>>(4, total);
  auto *d_chunk_roots = workspace.Ensure<cuda::std::array<int4, 2>>(5, total / kChunk + VdmpfT::m);
  auto *d_counts = workspace.Ensure<uint32_t>(6, VdmpfT::m);
  auto *d_leaf_offsets = workspace.Ensure<uint32_t>(7, VdmpfT::m + 1);
  auto *d_chunk_offsets = workspace.Ensure<uint32_t>(8, VdmpfT::m + 1);
  size_t cub_bytes = 0;
  detail::CheckCudaWorkspace(cub::DeviceRadixSort::SortPairs(
      nullptr, cub_bytes, d_keys, d_keys_out, d_vals, d_vals_out, static_cast<int>(total), 0, 16, stream));
  auto *d_cub_temp = workspace.Ensure<unsigned char>(9, cub_bytes);
  detail::CheckCudaWorkspace(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * VdmpfT::m, stream));

  constexpr int kBs = 256;
  int blocks = (n + kBs - 1) / kBs;
  detail::VdmpfLocateKernel<in_bits, Prp, In, kappa>
      <<<blocks, kBs, 0, stream>>>(xs, n, sigma, m_rt, b_rt, d_keys, d_vals, d_counts, vdmpf.prp);
  cub::DeviceRadixSort::SortPairs(
      d_cub_temp, cub_bytes, d_keys, d_keys_out, d_vals, d_vals_out, static_cast<int>(total), 0, 16, stream);
  if (b) {
    detail::VdmpfBatchEvalKernel<bucket_bits, unroll, true, Group, Prg, XorHash, In, kappa><<<blocks, kBs, 0, stream>>>(
        ys, xs, n, m_rt, s0s, cw_s, extra, cs, ocws, d_pi_tilde, d_keys, d_vals, vdmpf.prg, vdmpf.xor_hash);
  } else {
    detail::VdmpfBatchEvalKernel<bucket_bits, unroll, false, Group, Prg, XorHash, In, kappa>
        <<<blocks, kBs, 0, stream>>>(
            ys, xs, n, m_rt, s0s, cw_s, extra, cs, ocws, d_pi_tilde, d_keys, d_vals, vdmpf.prg, vdmpf.xor_hash);
  }
  detail::VdmpfFoldSetupKernel<<<1, 1, 0, stream>>>(d_leaf_offsets, d_chunk_offsets, d_counts, VdmpfT::m, kChunk);
  detail::VdmpfChunkFoldKernel<kChunk, Hash><<<static_cast<int>(total / kChunk + VdmpfT::m), kChunk, 0, stream>>>(
      d_chunk_roots, d_chunk_offsets, d_leaf_offsets, d_counts, d_vals_out, d_pi_tilde, VdmpfT::m, vdmpf.hash);
  detail::VdmpfBucketFoldKernel<VdmpfT::m, Hash>
      <<<1, kFoldThreads, 0, stream>>>(d_pi, d_chunk_roots, d_chunk_offsets, cs, vdmpf.hash);
}

// Compatibility name for the existing Merkle batch evaluator.
template <int unroll = 4, int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash,
    typename Hash, typename Prp, typename In, int kappa, int ch_lambda>
void VdmpfBatchEvalGpu(bool b, const In *xs, int4 *ys, cuda::std::array<int4, 4> *d_pi, int n, int4 sigma, int m_rt,
    int b_rt, const int4 *s0s, const int4 *cw_s, const ExtBits<bucket_bits> *extra, const cuda::std::array<int4, 4> *cs,
    const int4 *ocws,
    const Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, kappa, ch_lambda> &vdmpf,
    cudaStream_t stream = 0) {
  VdmpfBatchEvalMerkleGpu<unroll>(b, xs, ys, d_pi, n, sigma, m_rt, b_rt, s0s, cw_s, extra, cs, ocws, vdmpf, stream);
}

}  // namespace fss::gpu

#endif  // FSS_POINT_EVAL_GPU_CUH_
