/**
 * GPU point evaluation of FSS keys in level-major correction word layout.
 *
 * Repacking places one level's correction words for adjacent keys next to each
 * other, so neighboring threads can read neighboring correction words. Separate
 * terminal and verification buffers carry values read once per evaluation.
 * Performance depends on the scheme, batch, backend, and GPU.
 *
 * Run the matching Repack*Gpu() once per key after Gen(); the repack cost
 * can be amortized over later evaluations. Benchmarks must report whether they
 * include generation and repacking.
 *
 * The query points use the scheme's In type, so these kernels cover the same
 * input-domain widths as the host Eval, through __uint128_t. The packed tr
 * bits (extra) hold one bit per level and widen to uint64_t or __uint128_t
 * once in_bits passes 32 or 64. Query element types may be unsigned integers wide enough for the domain.
 * Values must belong to that domain; each element is loaded at its own stride.
 */
#ifndef FSS_REPACK_GPU_CUH_
#define FSS_REPACK_GPU_CUH_

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
#include <fss/half_tree_grotto_dcf.cuh>
#include <fss/packed_half_tree_dpf.cuh>
#include <fss/util.cuh>
#include <fss/vdpf.cuh>
#include <fss/vdmpf.cuh>
#include <cuda/std/span>
#include <limits>
#include <stdexcept>

namespace fss::gpu {

namespace detail {

// Packed right-control bits for the prepared layout.
template <int bits>
using ExtBits = std::conditional_t<bits <= 32, uint32_t, std::conditional_t<bits <= 64, uint64_t, __uint128_t>>;

enum class RepackKind { kPoint, kPacked, kDcf, kVerifiable };
template <typename Scheme>
struct RepackTraits;

inline void CheckGpuArguments(bool valid) {
  if (!valid) throw std::invalid_argument("gpu buffer dimensions do not match the operation");
}
template <typename Query, int bits>
concept QueryFor = !std::is_same_v<Query, bool> &&
    (std::is_unsigned_v<Query> || std::is_same_v<Query, __uint128_t>) && sizeof(Query) * 8 >= bits;
}

/**
 * Repack Dpf, HalfTreeDpf or GrottoDcf correction words into caller-owned
 * buffers. Specify Scheme explicitly. words contains key_count complete rows
 * of correction words, each including the terminal correction word, in
 * key-major order. words has at least (in_bits + 1) * key_count entries.
 * cw_s has at least in_bits * key_count entries in [level * key_count + key] order.
 * control has at least key_count entries. Dpf and GrottoDcf store
 * right-control bit i at bit i and the left-control bit in cw_s's low bit.
 * HalfTreeDpf stores its last-level extra bit at control[key]'s low bit.
 * terminal has at least key_count entries for Dpf and HalfTreeDpf. GrottoDcf
 * ignores terminal, which may be empty.
 * All buffers are device-accessible and must outlive work queued on stream.
 * @ingroup fss_api
 */
template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kPoint)
void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s,
    cuda::std::span<typename detail::RepackTraits<Scheme>::Control> control,
    cuda::std::span<int4> terminal, cudaStream_t stream = 0);

/**
 * Repack packed HalfTree correction words into caller-owned device buffers.
 * Scheme is PackedHalfTreeDpf or HalfTreeGrottoDcf. words contains key_count
 * complete key-major rows of correction words, including the final
 * correction word. words has at least (kDepth + 1) * key_count entries.
 * cw_s has at least kDepth * key_count entries in
 * [level * key_count + key] order. terminal has at least key_count entries.
 * All storage must outlive work queued on stream.
 * @ingroup fss_api
 */
template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kPacked)
void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s, cuda::std::span<int4> terminal, cudaStream_t stream = 0);

/**
 * Repack Dcf correction words into separate caller-owned device buffers.
 * words has at least (in_bits + 1) * key_count entries in key-major order.
 * cw_s and cw_v each have at least in_bits * key_count entries in
 * [level * key_count + key] order. terminal has at least key_count entries.
 * All storage must outlive work queued on stream.
 * @ingroup fss_api
 */
template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kDcf)
void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s, cuda::std::span<int4> cw_v,
    cuda::std::span<int4> terminal, cudaStream_t stream = 0);

/**
 * Repack Vdpf correction words and copy one four-block check record per key.
 * words has at least (in_bits + 1) * key_count entries in key-major order.
 * cw_s has at least in_bits * key_count entries in
 * [level * key_count + key] order, with the left-control
 * bit in each block's low bit. control has at least key_count entries, with
 * right-control bit i stored at bit i of control[key].
 * terminal has at least key_count entries. input_cs and output_cs each
 * contain at least key_count records. All buffers are device-accessible and
 * must outlive work queued on stream.
 * @ingroup fss_api
 */
template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kVerifiable)
void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s,
    cuda::std::span<typename detail::RepackTraits<Scheme>::Control> control,
    cuda::std::span<int4> terminal, cuda::std::span<cuda::std::array<int4, 4>> output_cs,
    cuda::std::span<const cuda::std::array<int4, 4>> input_cs, cudaStream_t stream = 0);










// -----------------------------------------------------------------------------

namespace detail {

__global__ void CopyPointChecksKernel(const cuda::std::array<int4, 4> *in,
    cuda::std::array<int4, 4> *out, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = in[i];
}

// Exposes a scheme's exact correction word to a shared repack kernel without
// putting a wide group tag into the kernel's template arguments.
template <typename S>
struct PlainRepackKernelDescriptor {
  using Scheme = S;
};

// Pack per-key HalfTree DPF cws (stride in_bits + 1) into level-major cw_s and the
// last-level extra bit (bit 0 of extra[k]).
template <int in_bits, typename Descriptor>
__global__ void HalfTreeDpfRepackKernel(
    const void *packed, int4 *cw_s, uint32_t *extra, int4 *terminal, int nkeys) {
  using Scheme = typename Descriptor::Scheme;
  const auto *pk = static_cast<const typename Scheme::Cw *>(packed);
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  uint32_t e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].s;
    if (i == in_bits - 1 && pk[(size_t)k * (in_bits + 1) + i].extra) e = 1;
  }
  extra[k] = e;
  terminal[k] = pk[(size_t)k * (in_bits + 1) + in_bits].s;
}

// Pack per-key DPF cws (stride in_bits + 1) into level-major cw_s, the tr bits
// (bit i of extra[k] = level i), and the output correction word unless out_cw
// is null, as for Grotto DCF keys, which leave it unused.
template <int in_bits, typename Descriptor>
__global__ void DpfRepackKernel(
    const void *packed, int4 *cw_s, ExtBits<in_bits> *extra, int4 *out_cw, int nkeys) {
  using Scheme = typename Descriptor::Scheme;
  const auto *pk = static_cast<const typename Scheme::Cw *>(packed);
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

// Pack per-key VDPF cws (stride in_bits + 1) into level-major cw_s and the tr
// bits (bit i of extra[k] = level i).
template <int in_bits, typename Descriptor>
__global__ void VdpfRepackKernel(
    const void *packed, int4 *cw_s, ExtBits<in_bits> *extra, int4 *terminal, int nkeys) {
  using Scheme = typename Descriptor::Scheme;
  const auto *pk = static_cast<const typename Scheme::Cw *>(packed);
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  ExtBits<in_bits> e = 0;
  for (int i = 0; i < in_bits; ++i) {
    cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (in_bits + 1) + i].s;
    if (pk[(size_t)k * (in_bits + 1) + i].tr) e |= static_cast<ExtBits<in_bits>>(1) << i;
  }
  extra[k] = e;
  terminal[k] = pk[(size_t)k * (in_bits + 1) + in_bits].s;
}

// Pack per-key DCF cws (stride in_bits + 1) into level-major cw_s and cw_v and
// the final correction word's v part.
template <int in_bits, typename Descriptor>
__global__ void DcfRepackKernel(
    const void *packed, int4 *cw_s, int4 *cw_v, int4 *out_cw, int nkeys) {
  using Scheme = typename Descriptor::Scheme;
  const auto *pk = static_cast<const typename Scheme::Cw *>(packed);
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
template <int in_bits, int unroll, bool b, typename Descriptor, typename In, typename Query = In>
__global__ void HalfTreeDpfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const uint32_t *extra,
    const int4 *ocws, const Query *xs, int nkeys, typename Descriptor::Backend hash, int4 instance_seed) {
  using Scheme = typename Descriptor::Scheme;
  using Group = typename HalfTreeGpuTraits<Scheme>::OutputGroup;
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = static_cast<In>(xs[tid]);

#pragma unroll unroll
  for (int i = 0; i < in_bits - 1; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    bool t = fss::util::GetLsb(node);
    int4 h = hash.Hash(fss::util::Xor(instance_seed, node));
    int4 cw = cw_s[(size_t)i * nkeys + tid];
    node = fss::util::Xor(fss::util::Xor(h, x_bit ? node : Zero4()), t ? cw : Zero4());
  }

  bool x_n = GetBit(x, 0);
  bool t = fss::util::GetLsb(node);
  int4 h = hash.Hash(fss::util::Xor(instance_seed, fss::util::SetLsb(node, x_n)));
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
__global__ void PackedHalfTreeDpfRepackKernel(const Cw *pk, int4 *cw_s, int4 *terminal, int nkeys) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nkeys) return;
  for (int i = 0; i < depth; ++i) cw_s[(size_t)i * nkeys + k] = pk[(size_t)k * (depth + 1) + i].s;
  terminal[k] = pk[(size_t)k * (depth + 1) + depth].s;
}

// Point-evaluate Packed HalfTree DPF keys, one thread per key. ys[tid] is x's output block, or
// with cmp the 1[alpha <= x] share of HalfTreeGrottoDcf::Eval(). cw_s comes from
// PackedHalfTreeDpfRepackKernel.
template <int in_bits, int out_bits, int unroll, bool b, bool cmp, typename Hash, typename In,
    typename Query, typename Y>
__global__ void PackedHalfTreeDpfEvalPointKernel(
    Y *ys, const int4 *seeds, const int4 *cw_s, const int4 *fcws, const Query *xs, int nkeys, Hash hash, int4 instance_seed) {
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nkeys) return;
  int4 node = fss::util::SetLsb(seeds[tid], b);
  In x = static_cast<In>(xs[tid]);
  fss::detail::GrottoTurns turns;

#pragma unroll unroll
  for (int i = 0; i < Scheme::kDepth; ++i) {
    bool x_bit = GetBit(x, in_bits - 1 - i);
    bool t = fss::util::GetLsb(node);
    if constexpr (cmp) turns(t, x_bit);
    int4 h = hash.Hash(fss::util::Xor(instance_seed, node));
    int4 cw = cw_s[(size_t)i * nkeys + tid];
    node = fss::util::Xor(fss::util::Xor(h, x_bit ? node : Zero4()), t ? cw : Zero4());
  }

  int4 leaf = hash.Hash(fss::util::Xor(instance_seed, node));
  if (fss::util::GetLsb(node)) leaf = fss::util::Xor(leaf, fcws[tid]);
  if constexpr (cmp) {
    ys[tid] = turns.share ^ fss::detail::SchemeAccess::HalfTreeGrottoLeafParity<
        HalfTreeGrottoDcf<in_bits, Hash, In>>(leaf, x);
  } else {
    ys[tid] = leaf;
  }
}

template <int unroll, bool cmp, int in_bits, int out_bits, typename Hash, typename In, typename Query,
    typename Y>
void PackedHalfTreeDpfEvalPointLaunch(bool b, const int4 *seeds, const int4 *cw_s, const int4 *fcws, const Query *xs,
    Y *ys, int nkeys, Hash hash, int4 instance_seed, cudaStream_t stream) {
  if (nkeys == 0) return;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    PackedHalfTreeDpfEvalPointKernel<in_bits, out_bits, unroll, true, cmp, Hash, In, Query, Y>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, fcws, xs, nkeys, hash, instance_seed);
  } else {
    PackedHalfTreeDpfEvalPointKernel<in_bits, out_bits, unroll, false, cmp, Hash, In, Query, Y>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, fcws, xs, nkeys, hash, instance_seed);
  }
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
// come from DcfRepackKernel. Group default construction provides the additive identity.
template <int in_bits, int unroll, bool b, typename Descriptor, typename In, typename Query = In>
__global__ void DcfEvalPointKernel(int4 *ys, const int4 *seeds, const int4 *cw_s, const int4 *cw_v, const int4 *out_cw,
    const Query *xs, int nkeys, typename Descriptor::Backend prg) {
  using Scheme = typename Descriptor::Scheme;
  using Group = typename DcfGpuSchemeTraits<Scheme>::GroupType;
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
// hash cost, the tail applies the verification hash and, when t, folds in cs.
// cw_s/extra come from VdpfRepackKernel. cs and ocws stay per-key: each is
// read once per evaluation, not once per level.
template <int in_bits, int unroll, bool b, typename Descriptor, typename Query>
__global__ void VdpfEvalPointKernel(int4 *ys, cuda::std::array<int4, 4> *pi, const int4 *seeds, const int4 *cw_s,
    const ExtBits<in_bits> *extra, const cuda::std::array<int4, 4> *cs, const int4 *ocws, const Query *xs, int nkeys,
    typename Descriptor::PrgBackend prg, typename Descriptor::XorHashBackend xor_hash) {
  using Group = typename Descriptor::Group;
  using In = typename Descriptor::InputType;
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

  int4 x_buf = fss::detail::Pack(x);
  auto pi_tilde = xor_hash.Hash(cuda::std::tuple<int4, const int4>{x_buf, s});
  if (t) {
    pi_tilde = util::Xor(cuda::std::span<const int4, 4>(pi_tilde), cuda::std::span<const int4, 4>(cs[tid]));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) pi[tid][j] = pi_tilde[j];
}

// Decomposes HalfTreeDpf into its template arguments so the launch spells a
// descriptor instead of the scheme's correction word, whose wide group tag
// would otherwise reach the CUDA host stub.
template <typename Scheme>
struct HalfTreeDpfRepackLauncher;

template <int in_bits, typename Group, typename Hash, typename In>
struct HalfTreeDpfRepackLauncher<HalfTreeDpf<in_bits, Group, Hash, In>> {
  static void Launch(const typename HalfTreeDpf<in_bits, Group, Hash, In>::Cw *cws, int nkeys,
      int4 *cw_s, uint32_t *extra, int4 *terminal, cudaStream_t stream) {
    using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
    using Descriptor = HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
    constexpr int kBs = 256;
    int blocks = (nkeys + kBs - 1) / kBs;
    HalfTreeDpfRepackKernel<in_bits, Descriptor><<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, terminal, nkeys);
  }
};

/*
 * One-time repack of per-key HalfTree DPF keys into the level-major layout
 * consumed by HalfTreeDpfEvalPointGpu. cw_s has in_bits * nkeys int4s,
 * extra and ocws have nkeys entries each.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits + 1.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed last-level extra bits.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Hash, typename In>
void HalfTreeDpfRepackGpu(const typename HalfTreeDpf<in_bits, Group, Hash, In>::Cw *cws, int nkeys,
    int4 *cw_s, uint32_t *extra, int4 *terminal, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  HalfTreeDpfRepackLauncher<HalfTreeDpf<in_bits, Group, Hash, In>>::Launch(
      cws, nkeys, cw_s, extra, terminal, stream);
}

/*
 * One-time repack of per-key Packed HalfTree DPF keys into the level-major
 * layout consumed by PackedHalfTreeDpfEvalPointGpu. cw_s has kDepth * nkeys
 * int4s. The final correction word is extracted into terminal.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @tparam out_bits Output width of the keys.
 * @param cws Correction words returned by Gen(), key-major, stride kDepth.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, int out_bits, typename Hash, typename In>
void PackedHalfTreeDpfRepackGpu(const typename PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::Cw *cws,
    int nkeys, int4 *cw_s, int4 *terminal, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::PackedHalfTreeDpfRepackKernel<Scheme::kDepth, typename Scheme::Cw>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, terminal, nkeys);
}

/*
 * One-time repack of per-key HalfTree Grotto DCF keys into the level-major
 * layout consumed by HalfTreeGrottoDcfEvalPointGpu, the layout of
 * PackedHalfTreeDpfRepackGpu. The final correction word is extracted into terminal.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride kDepth.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Hash, typename In>
void HalfTreeGrottoDcfRepackGpu(const typename HalfTreeGrottoDcf<in_bits, Hash, In>::Cw *cws, int nkeys,
    int4 *cw_s, int4 *terminal, cudaStream_t stream = 0) {
  PackedHalfTreeDpfRepackGpu<in_bits, 1, Hash, In>(cws, nkeys, cw_s, terminal, stream);
}

// Decomposes Dpf into its template arguments so the launch spells a descriptor
// instead of the scheme's correction word, whose wide group tag would otherwise
// reach the CUDA host stub.
template <typename Scheme>
struct DpfRepackLauncher;

template <int in_bits, typename Group, typename Prg, typename In>
struct DpfRepackLauncher<Dpf<in_bits, Group, Prg, In>> {
  static void Launch(const typename Dpf<in_bits, Group, Prg, In>::Cw *cws, int nkeys, int4 *cw_s,
      ExtBits<in_bits> *extra, int4 *out_cw, cudaStream_t stream) {
    using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
    using Descriptor = DpfGpuKernelDescriptor<in_bits, GroupTag, Prg, In>;
    constexpr int kBs = 256;
    int blocks = (nkeys + kBs - 1) / kBs;
    DpfRepackKernel<in_bits, Descriptor><<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, out_cw, nkeys);
  }
};

/*
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
  if (nkeys == 0) return;
  DpfRepackLauncher<Dpf<in_bits, Group, Prg, In>>::Launch(
      cws, nkeys, cw_s, extra, out_cw, stream);
}

// Decomposes Dcf into its template arguments so the launch spells a descriptor
// instead of the scheme's correction word, whose wide group tag would otherwise
// reach the CUDA host stub.
template <typename Scheme>
struct DcfRepackLauncher;

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
struct DcfRepackLauncher<Dcf<in_bits, Group, Prg, In, pred>> {
  static void Launch(const typename Dcf<in_bits, Group, Prg, In, pred>::Cw *cws, int nkeys,
      int4 *cw_s, int4 *cw_v, int4 *out_cw, cudaStream_t stream) {
    using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
    using Descriptor = DcfGpuKernelDescriptor<in_bits, GroupTag, Prg, In, pred>;
    constexpr int kBs = 256;
    int blocks = (nkeys + kBs - 1) / kBs;
    DcfRepackKernel<in_bits, Descriptor><<<blocks, kBs, 0, stream>>>(cws, cw_s, cw_v, out_cw, nkeys);
  }
};

/*
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
void DcfRepackGpu(const typename Dcf<in_bits, Group, Prg, In, pred>::Cw *cws, int nkeys, int4 *cw_s,
    int4 *cw_v, int4 *out_cw, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  DcfRepackLauncher<Dcf<in_bits, Group, Prg, In, pred>>::Launch(
      cws, nkeys, cw_s, cw_v, out_cw, stream);
}

/*
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
template <int in_bits, typename Prg, typename In>
void GrottoDcfRepackGpu(const typename GrottoDcf<in_bits, Prg, In>::Cw *cws, int nkeys, int4 *cw_s,
    ExtBits<in_bits> *extra, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using Descriptor = PlainRepackKernelDescriptor<GrottoDcf<in_bits, Prg, In>>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  detail::DpfRepackKernel<in_bits, Descriptor>
      <<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, nullptr, nkeys);
}

// Decomposes Vdpf into its template arguments so the launch spells a descriptor
// instead of the scheme's correction word, whose wide group tag would otherwise
// reach the CUDA host stub.
template <typename Scheme>
struct VdpfRepackLauncher;

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
struct VdpfRepackLauncher<Vdpf<in_bits, Group, Prg, XorHash, Hash, In>> {
  static void Launch(const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw *cws, int nkeys,
      int4 *cw_s, ExtBits<in_bits> *extra, int4 *terminal, cudaStream_t stream) {
    using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
    using Descriptor = VdpfGpuKernelDescriptor<in_bits, GroupTag, Prg, XorHash, Hash, In>;
    constexpr int kBs = 256;
    int blocks = (nkeys + kBs - 1) / kBs;
    VdpfRepackKernel<in_bits, Descriptor><<<blocks, kBs, 0, stream>>>(cws, cw_s, extra, terminal, nkeys);
  }
};

/*
 * One-time repack of per-key VDPF keys into the level-major layout consumed
 * by VdpfEvalPointGpu. cw_s has in_bits * nkeys int4s; extra has nkeys
 * entries. The final correction word is extracted into terminal.
 *
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param cws Correction words returned by Gen(), key-major, stride in_bits.
 * @param nkeys Number of keys.
 * @param cw_s Output level-major correction words, cw_s[i * nkeys + k].
 * @param extra Output packed tr bits, bit i of extra[k] = level i.
 * @param stream CUDA stream to launch on.
 */
template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
void VdpfRepackGpu(const typename Vdpf<in_bits, Group, Prg, XorHash, Hash, In>::Cw *cws, int nkeys,
    int4 *cw_s, ExtBits<in_bits> *extra, int4 *terminal, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  VdpfRepackLauncher<Vdpf<in_bits, Group, Prg, XorHash, Hash, In>>::Launch(
      cws, nkeys, cw_s, extra, terminal, stream);
}









/*
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
 * @param ocws Output correction words cws[in_bits].s, one per key.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one int4 per key.
 * @param nkeys Number of keys.
 * @param dpf A HalfTreeDpf instance (holds the CCR hash and instance seed).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Hash, typename In,
    typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void HalfTreeDpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const uint32_t *extra, const int4 *ocws,
    const Query *xs, int4 *ys, int nkeys, const HalfTreeDpf<in_bits, Group, Hash, In> &dpf,
    cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = HalfTreeGpuKernelDescriptor<in_bits, GroupTag, Hash, In>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, true, Descriptor, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys,
        fss::detail::SchemeAccess::GetHash(dpf),
        fss::detail::SchemeAccess::GetInstanceSeed(dpf));
  } else {
    detail::HalfTreeDpfEvalPointKernel<in_bits, unroll, false, Descriptor, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, extra, ocws, xs, nkeys,
        fss::detail::SchemeAccess::GetHash(dpf),
        fss::detail::SchemeAccess::GetInstanceSeed(dpf));
  }
}

/*
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
 * @param dpf A PackedHalfTreeDpf instance (holds the CCR hash and instance seed).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, int out_bits, typename Hash, typename In, typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void PackedHalfTreeDpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *fcws, const Query *xs,
    int4 *ys, int nkeys, const PackedHalfTreeDpf<in_bits, out_bits, Hash, In> &dpf, cudaStream_t stream = 0) {
  detail::PackedHalfTreeDpfEvalPointLaunch<unroll, false, in_bits, out_bits, Hash, In>(
      b, seeds, cw_s, fcws, xs, ys, nkeys,
        fss::detail::SchemeAccess::GetHash(dpf),
        fss::detail::SchemeAccess::GetInstanceSeed(dpf), stream);
}

/*
 * GPU point evaluation of nkeys HalfTree Grotto DCF keys: ys[k] = Eval(b,
 * key[k], xs[k]), the party's share of 1[alpha <= xs[k]]. One thread per key.
 * cw_s must come from HalfTreeGrottoDcfRepackGpu.
 *
 * @tparam unroll Unroll factor of the level loop.
 * @tparam in_bits Domain size as 2 ** in_bits.
 * @param b Party index. False for 0 and true for 1.
 * @param seeds Party seeds, one per key.
 * @param cw_s Level-major correction words from HalfTreeGrottoDcfRepackGpu.
 * @param fcws Final correction words returned by Gen(), one per key.
 * @param xs Evaluation points, one per key.
 * @param ys Output array, one share bit per key.
 * @param nkeys Number of keys.
 * @param dcf A HalfTreeGrottoDcf instance (holds the CCR hash and instance seed).
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Hash, typename In, typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void HalfTreeGrottoDcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *fcws, const Query *xs,
    bool *ys, int nkeys, const HalfTreeGrottoDcf<in_bits, Hash, In> &dcf, cudaStream_t stream = 0) {
  detail::PackedHalfTreeDpfEvalPointLaunch<unroll, true, in_bits, 1, Hash, In>(
      b, seeds, cw_s, fcws, xs, ys, nkeys,
        fss::detail::SchemeAccess::GetHash(dcf),
        fss::detail::SchemeAccess::GetInstanceSeed(dcf), stream);
}

/*
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
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In,
    typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void DpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra, const int4 *out_cw,
    const Query *xs, int4 *ys, int nkeys, const Dpf<in_bits, Group, Prg, In> &dpf, cudaStream_t stream = 0) {
  detail::DpfCoreEvalPointLaunch<unroll, Dpf<in_bits, Group, Prg, In>>(
      b, seeds, cw_s, extra, out_cw, xs, ys, nkeys, fss::detail::SchemeAccess::GetPrg(dpf), stream);
}

/*
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
 * @param dcf A Dcf instance (holds the PRG). Used on the host side. The predicate is deduced from the argument.
 * @param stream CUDA stream to launch on.
 */
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename In, DcfPred pred,
    typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void DcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const int4 *cw_v, const int4 *out_cw, const Query *xs,
    int4 *ys, int nkeys, const Dcf<in_bits, Group, Prg, In, pred> &dcf, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = DcfGpuKernelDescriptor<in_bits, GroupTag, Prg, In, pred>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::DcfEvalPointKernel<in_bits, unroll, true, Descriptor, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, fss::detail::SchemeAccess::GetPrg(dcf));
  } else {
    detail::DcfEvalPointKernel<in_bits, unroll, false, Descriptor, In, Query>
        <<<blocks, kBs, 0, stream>>>(ys, seeds, cw_s, cw_v, out_cw, xs, nkeys, fss::detail::SchemeAccess::GetPrg(dcf));
  }
}

/*
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
template <int unroll = 4, int in_bits, typename Prg, typename In, typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void GrottoDcfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const Query *xs, bool *ys, int nkeys, const GrottoDcf<in_bits, Prg, In> &dcf, cudaStream_t stream = 0) {
  detail::DpfEvalPointLaunch<unroll, true, in_bits, fss::group::Bytes, Prg, In>(
      b, seeds, cw_s, extra, nullptr, xs, ys, nkeys, fss::detail::SchemeAccess::GetPrg(dcf), stream);
}

/*
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
template <int unroll = 4, int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In, typename Query = In>
  requires detail::QueryFor<Query, in_bits>
void VdpfEvalPointGpu(bool b, const int4 *seeds, const int4 *cw_s, const ExtBits<in_bits> *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws, const Query *xs, int4 *ys, cuda::std::array<int4, 4> *pi, int nkeys,
    const Vdpf<in_bits, Group, Prg, XorHash, Hash, In> &vdpf, cudaStream_t stream = 0) {
  if (nkeys == 0) return;
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = VdpfGpuKernelDescriptor<in_bits, GroupTag, Prg, XorHash, Hash, In>;
  constexpr int kBs = 256;
  int blocks = (nkeys + kBs - 1) / kBs;
  if (b) {
    detail::VdpfEvalPointKernel<in_bits, unroll, true, Descriptor, Query>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys,
        fss::detail::SchemeAccess::GetPrg(vdpf),
        fss::detail::SchemeAccess::GetXorHash(vdpf));
  } else {
    detail::VdpfEvalPointKernel<in_bits, unroll, false, Descriptor, Query>
        <<<blocks, kBs, 0, stream>>>(ys, pi, seeds, cw_s, extra, cs, ocws, xs, nkeys,
        fss::detail::SchemeAccess::GetPrg(vdpf),
        fss::detail::SchemeAccess::GetXorHash(vdpf));
  }
}

template <int in_bits, typename Group, typename Hash, typename In>
struct RepackTraits<HalfTreeDpf<in_bits, Group, Hash, In>> {
  using Scheme = HalfTreeDpf<in_bits, Group, Hash, In>;
  using Control = uint32_t;
  static constexpr RepackKind kKind = RepackKind::kPoint;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s,
      cuda::std::span<Control> control,
      cuda::std::span<int4> terminal, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(in_bits) * key_count);
    detail::CheckGpuArguments(control.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(terminal.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(in_bits + 1) * key_count);
    detail::HalfTreeDpfRepackGpu<in_bits, Group, Hash, In>(words.data(), key_count, cw_s.data(), control.data(), terminal.data(), stream);
  }
};

template <int in_bits, int out_bits, typename Hash, typename In>
struct RepackTraits<PackedHalfTreeDpf<in_bits, out_bits, Hash, In>> {
  using Scheme = PackedHalfTreeDpf<in_bits, out_bits, Hash, In>;
  static constexpr RepackKind kKind = RepackKind::kPacked;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s, cuda::std::span<int4> terminal, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::kDepth) * key_count);
    detail::CheckGpuArguments(terminal.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(PackedHalfTreeDpf<in_bits, out_bits, Hash, In>::kDepth + 1) * key_count);
    detail::PackedHalfTreeDpfRepackGpu<in_bits, out_bits, Hash, In>(words.data(), key_count, cw_s.data(), terminal.data(), stream);
  }
};

template <int in_bits, typename Hash, typename In>
struct RepackTraits<HalfTreeGrottoDcf<in_bits, Hash, In>> {
  using Scheme = HalfTreeGrottoDcf<in_bits, Hash, In>;
  static constexpr RepackKind kKind = RepackKind::kPacked;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s, cuda::std::span<int4> terminal, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(HalfTreeGrottoDcf<in_bits, Hash, In>::kDepth) * key_count);
    detail::CheckGpuArguments(terminal.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(HalfTreeGrottoDcf<in_bits, Hash, In>::kDepth + 1) * key_count);
    detail::HalfTreeGrottoDcfRepackGpu<in_bits, Hash, In>(words.data(), key_count, cw_s.data(), terminal.data(), stream);
  }
};

template <int in_bits, typename Group, typename Prg, typename In>
struct RepackTraits<Dpf<in_bits, Group, Prg, In>> {
  using Scheme = Dpf<in_bits, Group, Prg, In>;
  using Control = ExtBits<in_bits>;
  static constexpr RepackKind kKind = RepackKind::kPoint;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s,
      cuda::std::span<Control> control,
      cuda::std::span<int4> terminal, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(in_bits) * key_count);
    detail::CheckGpuArguments(control.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(terminal.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(in_bits + 1) * key_count);
    detail::DpfRepackGpu<in_bits, Group, Prg, In>(words.data(), key_count, cw_s.data(), control.data(), terminal.data(), stream);
  }
};

template <int in_bits, typename Group, typename Prg, typename In, DcfPred pred>
struct RepackTraits<Dcf<in_bits, Group, Prg, In, pred>> {
  using Scheme = Dcf<in_bits, Group, Prg, In, pred>;
  static constexpr RepackKind kKind = RepackKind::kDcf;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s, cuda::std::span<int4> cw_v,
      cuda::std::span<int4> terminal, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(in_bits) * key_count);
    detail::CheckGpuArguments(terminal.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(in_bits + 1) * key_count);
    detail::CheckGpuArguments(cw_v.size() >= static_cast<size_t>(in_bits) * key_count);
    detail::DcfRepackGpu<in_bits, Group, Prg, In, pred>(words.data(), key_count, cw_s.data(), cw_v.data(), terminal.data(), stream);
  }
};

template <int in_bits, typename Prg, typename In>
struct RepackTraits<GrottoDcf<in_bits, Prg, In>> {
  using Scheme = GrottoDcf<in_bits, Prg, In>;
  using Control = ExtBits<in_bits>;
  static constexpr RepackKind kKind = RepackKind::kPoint;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s,
      cuda::std::span<Control> control,
      cuda::std::span<int4> terminal, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(in_bits) * key_count);
    detail::CheckGpuArguments(control.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(in_bits + 1) * key_count);
    detail::GrottoDcfRepackGpu<in_bits, Prg, In>(words.data(), key_count, cw_s.data(), control.data(), stream);
  }
};

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash, typename In>
struct RepackTraits<Vdpf<in_bits, Group, Prg, XorHash, Hash, In>> {
  using Scheme = Vdpf<in_bits, Group, Prg, XorHash, Hash, In>;
  using Control = ExtBits<in_bits>;
  static constexpr RepackKind kKind = RepackKind::kVerifiable;
  static void Run(cuda::std::span<const typename Scheme::Cw> words, int key_count,
      cuda::std::span<int4> cw_s,
      cuda::std::span<Control> control,
      cuda::std::span<int4> terminal, cuda::std::span<cuda::std::array<int4, 4>> output_cs,
      cuda::std::span<const cuda::std::array<int4, 4>> input_cs, cudaStream_t stream) {
    detail::CheckGpuArguments(key_count >= 0);
    detail::CheckGpuArguments(cw_s.size() >= static_cast<size_t>(in_bits) * key_count);
    detail::CheckGpuArguments(control.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(terminal.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(output_cs.size() >= static_cast<size_t>(key_count));
    detail::CheckGpuArguments(words.size() >= static_cast<size_t>(in_bits + 1) * key_count);
    detail::CheckGpuArguments(input_cs.size() >= static_cast<size_t>(key_count));
    detail::VdpfRepackGpu<in_bits, Group, Prg, XorHash, Hash, In>(words.data(), key_count, cw_s.data(), control.data(), terminal.data(), stream);
    if (key_count != 0) detail::CopyPointChecksKernel<<<(key_count + 255) / 256, 256, 0, stream>>>(input_cs.data(), output_cs.data(), key_count);
  }
};

}  // namespace detail

template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kPoint)
inline void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s,
    cuda::std::span<typename detail::RepackTraits<Scheme>::Control> control,
    cuda::std::span<int4> terminal, cudaStream_t stream) {
  detail::RepackTraits<Scheme>::Run(words, key_count, cw_s, control, terminal, stream);
}

template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kPacked)
inline void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s, cuda::std::span<int4> terminal, cudaStream_t stream) {
  detail::RepackTraits<Scheme>::Run(words, key_count, cw_s, terminal, stream);
}

template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kDcf)
inline void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s, cuda::std::span<int4> cw_v,
    cuda::std::span<int4> terminal, cudaStream_t stream) {
  detail::RepackTraits<Scheme>::Run(words, key_count, cw_s, cw_v, terminal, stream);
}

template <typename Scheme>
  requires(detail::RepackTraits<Scheme>::kKind == detail::RepackKind::kVerifiable)
inline void RepackGpu(cuda::std::span<const typename Scheme::Cw> words, int key_count,
    cuda::std::span<int4> cw_s,
    cuda::std::span<typename detail::RepackTraits<Scheme>::Control> control,
    cuda::std::span<int4> terminal, cuda::std::span<cuda::std::array<int4, 4>> output_cs,
    cuda::std::span<const cuda::std::array<int4, 4>> input_cs, cudaStream_t stream) {
  detail::RepackTraits<Scheme>::Run(words, key_count, cw_s, control, terminal, output_cs, input_cs, stream);
}





}  // namespace fss::gpu

#endif  // FSS_REPACK_GPU_CUH_
