/**
 * GPU point evaluation of FSS keys in level-major correction-word layout.
 *
 * Repacking places one level's corrections for adjacent keys next to each
 * other, so neighboring threads can read neighboring corrections. Separate
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
 * Repack Dpf, HalfTreeDpf or GrottoDcf corrections into caller-owned buffers.
 * Specify Scheme explicitly. words contains key_count complete correction rows,
 * each including the terminal correction, in key-major order. words has at
 * least (in_bits + 1) * key_count entries.
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
 * Repack packed HalfTree corrections into caller-owned device buffers.
 * Scheme is PackedHalfTreeDpf or HalfTreeGrottoDcf. words contains key_count
 * complete key-major correction rows, including the final correction. words
 * has at least (kDepth + 1) * key_count entries.
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
 * Repack Dcf corrections into separate caller-owned device buffers.
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
 * Repack Vdpf corrections and copy one four-block check record per key.
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

__device__ __host__ inline int4 Zero4() {
  return {0, 0, 0, 0};
}

template <typename T>
__device__ __host__ inline bool GetBit(T x, int i) {
  return ((x >> i) & 1) != 0;
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
// hash cost, the tail applies the verification hash and, when t, the cs correction.
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

// Holds the DMPF group tag and backends so the batch kernel template arguments
// carry no wide Group type.
template <int in_bits, int bucket_bits, typename GroupTag, typename Prg, typename Prp, typename In>
struct DmpfBatchGpuKernelDescriptor {
  using Group = typename GroupTag::type;
  using PrgBackend = Prg;
  using PrpBackend = Prp;
  using InputType = In;
};

// Batch-evaluate a DMPF key: one thread per input point. Each thread runs the
// kappa Cuckoo Locate calls on its own point and descends the located bucket
// keys itself, so the outputs need no atomics and no cross-thread
// synchronization. cw_s/extra/out_cw come from DpfRepackGpu run over the m_rt
// active bucket keys; a warp's bucket indices are pseudorandom, so per-level
// loads scatter inside the level's row instead of following the per-key
// layout's one-line-per-thread pattern.
template <int in_bits, int bucket_bits, int unroll, bool b, typename Descriptor>
__global__ void DmpfBatchEvalKernel(int4 *ys, const typename Descriptor::InputType *xs, int n, int4 sigma, int m_rt,
    __uint128_t b_rt, const int4 *s0s, const int4 *cw_s, const ExtBits<bucket_bits> *extra, const int4 *out_cw,
    typename Descriptor::PrgBackend prg, typename Descriptor::PrpBackend prp) {
  using Group = typename Descriptor::Group;
  using In = typename Descriptor::InputType;
  static_assert(in_bits <= sizeof(In) * 8, "In too narrow for in_bits");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  In x = xs[tid];
  constexpr __uint128_t n128 = static_cast<__uint128_t>(1) << in_bits;
  cuckoo_hash::detail::PrpHash<typename Descriptor::PrpBackend, In> prp_hash{prp};
  auto acc = Group::From(Zero4());
  for (int k = 0; k < 3; ++k) {
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
template <int in_bits, typename Prp, typename In>
__global__ void VdmpfLocateKernel(
    const In *xs, int n, int4 sigma, int m_rt, __uint128_t b_rt, uint32_t *keys, uint64_t *vals, uint32_t *counts, Prp prp) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  In x = xs[tid];
  constexpr __uint128_t n128 = static_cast<__uint128_t>(1) << in_bits;
  cuckoo_hash::detail::PrpHash<Prp, In> prp_hash{prp};
  for (int k = 0; k < 3; ++k) {
    auto [bucket, j] = prp_hash.Locate(sigma, x, k, n128, b_rt);
    size_t idx = static_cast<size_t>(tid) * 3 + k;
    keys[idx] = static_cast<uint32_t>(bucket);
    vals[idx] = (static_cast<uint64_t>(idx) << 32) | static_cast<uint32_t>(j);
    if (bucket < m_rt) atomicAdd(&counts[bucket], 1u);
  }
}

// Holds the VDMPF group tag and backends so the batch kernel template arguments
// carry no wide Group type.
template <int bucket_bits, typename GroupTag, typename Prg, typename XorHash>
struct VdmpfBatchGpuKernelDescriptor {
  using Group = typename GroupTag::type;
  using PrgBackend = Prg;
  using XorHashBackend = XorHash;
};

// Batch-evaluate a VDMPF key: one thread per input point computes its kappa
// contributions (from the Locate records) and writes each corrected per-point
// hash pi_tilde at its contribution index. cw_s/extra come from VdpfRepackGpu
// over the m_rt active bucket keys; cs and ocws are indexed by bucket.
template <int bucket_bits, int unroll, bool b, typename Descriptor>
__global__ void VdmpfBatchEvalKernel(int4 *ys, int n, int m_rt, const int4 *s0s, const int4 *cw_s,
    const ExtBits<bucket_bits> *extra, const cuda::std::array<int4, 4> *cs, const int4 *ocws,
    cuda::std::array<int4, 4> *pi_tilde_buf, const uint32_t *keys, const uint64_t *vals,
    typename Descriptor::PrgBackend prg, typename Descriptor::XorHashBackend xor_hash) {
  using Group = typename Descriptor::Group;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  auto acc = Group::From(Zero4());
  for (int k = 0; k < 3; ++k) {
    size_t idx = static_cast<size_t>(tid) * 3 + k;
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

    int4 x_buf = fss::detail::Pack(j);
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

// Fold one execution chunk of the sorted Locate records into a
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
  for (uint32_t j = threadIdx.x; j < take; j += blockDim.x) {
    uint32_t p = leaf_base + j;
    uint32_t idx = static_cast<uint32_t>(sorted_vals[p] >> 32);
    cur[j] = hash.Hash(cuda::std::span<const int4, 4>(pi_tilde_buf[idx]));
  }
  __syncthreads();

  uint32_t count = take;
  while (count > 1) {
    uint32_t pairs = count / 2;
    for (uint32_t j = threadIdx.x; j < pairs; j += blockDim.x) {
      int4 msg[4] = {cur[2 * j][0], cur[2 * j][1], cur[2 * j + 1][0], cur[2 * j + 1][1]};
      next[j] = hash.Hash(cuda::std::span<const int4, 4>(msg));
    }
    if (count % 2 != 0 && threadIdx.x == 0) {
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

// Reduce one level of every bucket's chunk roots. Buffers are disjoint so
// no parent write can race another thread's child reads. Each bucket keeps its
// original chunk-offset segment, including after its root is already complete.
// Power-of-two chunks align with the canonical leaf tree. Its incomplete last
// chunk is carried unchanged through any missing levels before this reduction.
template <typename Hash>
__global__ void VdmpfChunkRootsFoldKernel(cuda::std::array<int4, 2> *out,
    const cuda::std::array<int4, 2> *in, const uint32_t *chunk_offsets,
    int m_buckets, uint32_t stride, Hash hash) {
  size_t slot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= chunk_offsets[m_buckets]) return;
  int lo = 0, hi = m_buckets - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (slot < chunk_offsets[mid + 1]) hi = mid;
    else lo = mid + 1;
  }
  int bucket = lo;
  uint32_t base = chunk_offsets[bucket];
  uint32_t local = static_cast<uint32_t>(slot) - base;
  uint32_t chunks = chunk_offsets[bucket + 1] - base;
  uint32_t count = (chunks + stride - 1) / stride;
  uint32_t parents = (count + 1) / 2;
  if (local >= parents) return;
  auto left = in[base + 2 * local];
  if (2 * local + 1 < count) {
    auto right = in[base + 2 * local + 1];
    int4 msg[4] = {left[0], left[1], right[0], right[1]};
    out[slot] = hash.Hash(cuda::std::span<const int4, 4>(msg));
  } else {
    out[slot] = left;
  }
}

// Bind each nonempty bucket's canonical leaf root to its hashed cs, then fold
// the bucket nodes level by level. Execution chunk sizes do not enter the proof.
template <int m_buckets, typename Hash>
__global__ void VdmpfBucketFoldKernel(cuda::std::array<int4, 4> *pi, const cuda::std::array<int4, 2> *chunk_roots,
    const uint32_t *chunk_offsets, const cuda::std::array<int4, 4> *cs, Hash hash) {
  __shared__ cuda::std::array<int4, 2> s_a[m_buckets];
  __shared__ cuda::std::array<int4, 2> s_b[m_buckets];
  int i = threadIdx.x;
  if (i < m_buckets) {
    cuda::std::array<int4, 2> node = hash.Hash(cuda::std::span<const int4, 4>(cs[i]));
    if (chunk_offsets != nullptr && chunk_offsets[i] != chunk_offsets[i + 1]) {
      auto root = chunk_roots[chunk_offsets[i]];
      int4 msg[4] = {node[0], node[1], root[0], root[1]};
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
 * int4s. The final correction is extracted into terminal.
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
 * PackedHalfTreeDpfRepackGpu. The final correction is extracted into terminal.
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
 * entries. The final correction is extracted into terminal.
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
 * @param ocws Output corrections cws[in_bits].s, one per key.
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

/*
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
    typename In, int ch_lambda>
void DmpfBatchEvalGpu(bool b, const In *xs, int4 *ys, int n, int4 sigma, int m_rt, __uint128_t b_rt, const int4 *s0s,
    const int4 *cw_s, const ExtBits<bucket_bits> *extra, const int4 *out_cw,
    const Dmpf<in_bits, max_points, bucket_bits, Group, Prg, Prp, In, ch_lambda> &dmpf,
    cudaStream_t stream = 0) {
  if (n == 0) return;
  static_assert(std::is_trivially_copyable_v<Prp>, "the Prp backend must be device-callable, e.g. ChaChaFeistel");
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = DmpfBatchGpuKernelDescriptor<in_bits, bucket_bits, GroupTag, Prg, Prp, In>;
  constexpr int kBs = 256;
  int blocks = (n + kBs - 1) / kBs;
  if (b) {
    detail::DmpfBatchEvalKernel<in_bits, bucket_bits, unroll, true, Descriptor>
        <<<blocks, kBs, 0, stream>>>(ys, xs, n, sigma, m_rt, b_rt, s0s, cw_s, extra, out_cw,
        fss::detail::SchemeAccess::GetPrg(dmpf),
        fss::detail::SchemeAccess::GetPrp(dmpf));
  } else {
    detail::DmpfBatchEvalKernel<in_bits, bucket_bits, unroll, false, Descriptor>
        <<<blocks, kBs, 0, stream>>>(ys, xs, n, sigma, m_rt, b_rt, s0s, cw_s, extra, out_cw,
        fss::detail::SchemeAccess::GetPrg(dmpf),
        fss::detail::SchemeAccess::GetPrp(dmpf));
  }
}

/*
 * GPU batch evaluation of a VDMPF key with the Merkle-fold proof:
 * ys matches host BatchEvalMerkle() for xs, and d_pi receives its proof. Launches
 * on the stream: a Locate pass writes per-(point, hash) Cuckoo records, a
 * stable CUB radix sort groups them by bucket in the same canonical order the
 * host BatchEvalMerkle walks its per-bucket lists, an evaluation pass computes
 * every point's kappa contributions and corrected per-point hashes, and the
 * fold passes compute each bucket's canonical Merkle root, bind it to the
 * bucket's hashed cs, and fold across all buckets. Execution chunk and block
 * sizes do not change the proof, which matches host BatchEvalMerkle().
 *
 * cw_s/extra must come from VdpfRepackGpu run over the m_rt active bucket keys
 * (nkeys = m_rt, in_bits = bucket_bits of the Vdmpf), s0s and ocws hold party
 * b's per-bucket seeds and output correction words (m_rt entries), and cs
 * holds the shared per-bucket check seeds for all m buckets - the fold covers
 * the buckets past m_rt as well. The Prp backend must be device-callable:
 * ChaChaFeistel qualifies, Aes128Feistel does not.
 *
 * The compact Locate records support bucket_bits <= 32.
 * The Locate records, sorted copies, bucket counts, per-contribution hashes,
 * chunk roots, reduction scratch, and CUB temporary storage are local to this call.
 * Temporary device storage is allocated and released on the supplied stream for this call.
 * @tparam unroll Unroll factor of the level loop.
 * @tparam chunk_size Power-of-two leaf count per execution chunk, in [1, 512].
 * Defaults to 256. Two shared node buffers use 64*chunk_size bytes, at most
 * 32 KiB. This implementation limit is independent of the proof definition.
 * @tparam threads Threads per execution block, a multiple of 32 in [32, 1024].
 * Defaults to 256. Chunk work is strided when chunk_size exceeds threads.
 * @param b Party index. False for 0 and true for 1.
 * @param xs Evaluation points (device memory), n entries.
 * @param ys Output array (device memory), one int4 per point.
 * @param d_pi Proof output (device memory), four int4s.
 * @param n Number of points in [0, INT_MAX / 3].
 * @throws std::invalid_argument If n is outside that range.
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
template <int unroll = 4, int chunk_size = 256, int threads = 256,
    int in_bits, int max_points, int bucket_bits, typename Group, typename Prg, typename XorHash,
    typename Hash, typename Prp, typename In, int ch_lambda>
  requires(bucket_bits <= 32 && chunk_size >= 1 && chunk_size <= 512 &&
      (chunk_size & (chunk_size - 1)) == 0 && threads >= 32 && threads <= 1024 && threads % 32 == 0)
void VdmpfBatchEvalMerkleGpu(bool b, const In *xs, int4 *ys, cuda::std::array<int4, 4> *d_pi, int n, int4 sigma,
    int m_rt, __uint128_t b_rt, const int4 *s0s, const int4 *cw_s, const ExtBits<bucket_bits> *extra,
    const cuda::std::array<int4, 4> *cs, const int4 *ocws,
    const Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda> &vdmpf,
    cudaStream_t stream = 0) {
  using VdmpfT = Vdmpf<in_bits, max_points, bucket_bits, Group, Prg, XorHash, Hash, Prp, In, ch_lambda>;
  static_assert(std::is_trivially_copyable_v<Prp>, "the Prp backend must be device-callable, e.g. ChaChaFeistel");
  static_assert(VdmpfT::kBucketCount < (1 << 16), "the radix sort covers the bucket keys");
  static_assert(2 * VdmpfT::kBucketCount * 32 <= 46 * 1024, "the bucket-fold kernel double-buffers every bucket node");
  if (n < 0 || n > std::numeric_limits<int>::max() / 3) {
    throw std::invalid_argument("n must be in [0, INT_MAX / 3]");
  }
  const size_t total = static_cast<size_t>(n) * 3;
  constexpr int kChunk = chunk_size;

  constexpr int kFoldThreads = (VdmpfT::kBucketCount + 31) / 32 * 32;
  if (n == 0) {
    detail::VdmpfBucketFoldKernel<VdmpfT::kBucketCount, Hash>
        <<<1, kFoldThreads, 0, stream>>>(d_pi, nullptr, nullptr, cs, fss::detail::SchemeAccess::GetHash(vdmpf));
    detail::CheckCudaStatus(cudaGetLastError());
    return;
  }
  detail::DeviceScratch d_keys_storage(stream);
  detail::CheckCudaStatus(d_keys_storage.Allocate<uint32_t>(total));
  auto *d_keys = d_keys_storage.Get<uint32_t>();
  detail::DeviceScratch d_keys_out_storage(stream);
  detail::CheckCudaStatus(d_keys_out_storage.Allocate<uint32_t>(total));
  auto *d_keys_out = d_keys_out_storage.Get<uint32_t>();
  detail::DeviceScratch d_vals_storage(stream);
  detail::CheckCudaStatus(d_vals_storage.Allocate<uint64_t>(total));
  auto *d_vals = d_vals_storage.Get<uint64_t>();
  detail::DeviceScratch d_vals_out_storage(stream);
  detail::CheckCudaStatus(d_vals_out_storage.Allocate<uint64_t>(total));
  auto *d_vals_out = d_vals_out_storage.Get<uint64_t>();
  detail::DeviceScratch d_pi_tilde_storage(stream);
  detail::CheckCudaStatus(d_pi_tilde_storage.Allocate<cuda::std::array<int4, 4>>(total));
  auto *d_pi_tilde = d_pi_tilde_storage.Get<cuda::std::array<int4, 4>>();
  size_t root_capacity = total / kChunk + VdmpfT::kBucketCount;
  if (root_capacity > total) root_capacity = total;
  detail::DeviceScratch d_chunk_roots_storage(stream);
  detail::CheckCudaStatus(d_chunk_roots_storage.Allocate<cuda::std::array<int4, 2>>(root_capacity));
  auto *d_chunk_roots = d_chunk_roots_storage.Get<cuda::std::array<int4, 2>>();
  detail::DeviceScratch d_root_scratch_storage(stream);
  detail::CheckCudaStatus(d_root_scratch_storage.Allocate<cuda::std::array<int4, 2>>(root_capacity));
  auto *d_root_scratch = d_root_scratch_storage.Get<cuda::std::array<int4, 2>>();
  detail::DeviceScratch d_counts_storage(stream);
  detail::CheckCudaStatus(d_counts_storage.Allocate<uint32_t>(VdmpfT::kBucketCount));
  auto *d_counts = d_counts_storage.Get<uint32_t>();
  detail::DeviceScratch d_leaf_offsets_storage(stream);
  detail::CheckCudaStatus(d_leaf_offsets_storage.Allocate<uint32_t>(VdmpfT::kBucketCount + 1));
  auto *d_leaf_offsets = d_leaf_offsets_storage.Get<uint32_t>();
  detail::DeviceScratch d_chunk_offsets_storage(stream);
  detail::CheckCudaStatus(d_chunk_offsets_storage.Allocate<uint32_t>(VdmpfT::kBucketCount + 1));
  auto *d_chunk_offsets = d_chunk_offsets_storage.Get<uint32_t>();
  size_t cub_bytes = 0;
  detail::CheckCudaStatus(cub::DeviceRadixSort::SortPairs(
      nullptr, cub_bytes, d_keys, d_keys_out, d_vals, d_vals_out, static_cast<int>(total), 0, 16, stream));
  detail::DeviceScratch d_cub_temp_storage(stream);
  detail::CheckCudaStatus(d_cub_temp_storage.Allocate<unsigned char>(cub_bytes == 0 ? 1 : cub_bytes));
  auto *d_cub_temp = d_cub_temp_storage.Get<unsigned char>();
  detail::CheckCudaStatus(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * VdmpfT::kBucketCount, stream));

  constexpr int kBs = threads;
  int blocks = (n + kBs - 1) / kBs;
  detail::VdmpfLocateKernel<in_bits, Prp, In>
      <<<blocks, kBs, 0, stream>>>(xs, n, sigma, m_rt, b_rt, d_keys, d_vals, d_counts,
        fss::detail::SchemeAccess::GetPrp(vdmpf));
  detail::CheckCudaStatus(cudaGetLastError());
  detail::CheckCudaStatus(cub::DeviceRadixSort::SortPairs(
      d_cub_temp, cub_bytes, d_keys, d_keys_out, d_vals, d_vals_out, static_cast<int>(total), 0, 16, stream));
  using GroupTag = typename fss::detail::GpuGroupTag<Group>::Type;
  using Descriptor = VdmpfBatchGpuKernelDescriptor<bucket_bits, GroupTag, Prg, XorHash>;
  if (b) {
    detail::VdmpfBatchEvalKernel<bucket_bits, unroll, true, Descriptor><<<blocks, kBs, 0, stream>>>(
        ys, n, m_rt, s0s, cw_s, extra, cs, ocws, d_pi_tilde, d_keys, d_vals,
        fss::detail::SchemeAccess::GetPrg(vdmpf),
        fss::detail::SchemeAccess::GetXorHash(vdmpf));
  } else {
    detail::VdmpfBatchEvalKernel<bucket_bits, unroll, false, Descriptor>
        <<<blocks, kBs, 0, stream>>>(
            ys, n, m_rt, s0s, cw_s, extra, cs, ocws, d_pi_tilde, d_keys, d_vals,
        fss::detail::SchemeAccess::GetPrg(vdmpf),
        fss::detail::SchemeAccess::GetXorHash(vdmpf));
  }
  detail::CheckCudaStatus(cudaGetLastError());
  detail::VdmpfFoldSetupKernel<<<1, 1, 0, stream>>>(d_leaf_offsets, d_chunk_offsets, d_counts, VdmpfT::kBucketCount, kChunk);
  detail::CheckCudaStatus(cudaGetLastError());
  detail::VdmpfChunkFoldKernel<kChunk, Hash><<<static_cast<int>(root_capacity), kBs, 0, stream>>>(
      d_chunk_roots, d_chunk_offsets, d_leaf_offsets, d_counts, d_vals_out, d_pi_tilde, VdmpfT::kBucketCount,
        fss::detail::SchemeAccess::GetHash(vdmpf));
  detail::CheckCudaStatus(cudaGetLastError());
  auto *roots = d_chunk_roots;
  auto *roots_scratch = d_root_scratch;
  size_t max_roots = (total + kChunk - 1) / kChunk;
  int root_blocks = static_cast<int>((root_capacity + kBs - 1) / kBs);
  for (uint32_t stride = 1; max_roots > 1; stride *= 2) {
    detail::VdmpfChunkRootsFoldKernel<Hash><<<root_blocks, kBs, 0, stream>>>(
        roots_scratch, roots, d_chunk_offsets, VdmpfT::kBucketCount, stride,
        fss::detail::SchemeAccess::GetHash(vdmpf));
    detail::CheckCudaStatus(cudaGetLastError());
    auto *swap = roots;
    roots = roots_scratch;
    roots_scratch = swap;
    max_roots = (max_roots + 1) / 2;
  }
  detail::VdmpfBucketFoldKernel<VdmpfT::kBucketCount, Hash>
      <<<1, kFoldThreads, 0, stream>>>(d_pi, roots, d_chunk_offsets, cs,
        fss::detail::SchemeAccess::GetHash(vdmpf));
  detail::CheckCudaStatus(detail::ReleaseScratch(cudaGetLastError(),
      {&d_keys_storage, &d_keys_out_storage, &d_vals_storage, &d_vals_out_storage, &d_pi_tilde_storage, &d_chunk_roots_storage, &d_root_scratch_storage, &d_counts_storage, &d_leaf_offsets_storage, &d_chunk_offsets_storage, &d_cub_temp_storage}));
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
