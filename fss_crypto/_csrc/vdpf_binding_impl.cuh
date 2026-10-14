// SPDX-License-Identifier: Apache-2.0
// VDPF binding implementation. Included by JIT-generated .cu files after type
// aliases (kInBits, InType, GroupType, PrgType, XorHashType, HashType, VdpfInst)
// are defined.

#include "binding_common.cuh"
#include <fss/eval_all_gpu.cuh>
#include <fss/hash/blake3.cuh>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <tuple>
#include <type_traits>

namespace {

// Device paths require both hashes to be Blake3: the Merkle fold calls
// Hash::Hash on the GPU, and Sha256 traps under __CUDA_ARCH__.
constexpr bool kGpuHashes = std::is_same_v<XorHashType, fss::hash::Blake3> &&
    std::is_same_v<HashType, fss::hash::Blake3>;

// Build a verifiable-scheme hash instance: Blake3 copies a 2x int4 IV by
// value, Sha256 takes a single int4 key.
template <typename H>
H MakeHash() {
  if constexpr (std::is_constructible_v<H, cuda::std::span<const int4, 2>>) {
    return H(cuda::std::span<const int4, 2>(kHashIv, 2));
  } else {
    return H(kHashIv[0]);
  }
}

VdpfInst MakeVdpf() {
  return VdpfInst{PrgType(g_nonce0, g_nonce1), MakeHash<XorHashType>(), MakeHash<HashType>()};
}

}  // namespace

// ---------------------------------------------------------------------------
// GPU eval kernel
// ---------------------------------------------------------------------------

// Templated so sha256-only builds never instantiate device code.
template <typename Inst>
__global__ void vdpf_eval_kernel(Inst vdpf, bool party, int4 s0, const typename Inst::Cw *cws,
    cuda::std::array<int4, 4> cs, int4 ocw, int4 *y_out, cuda::std::array<int4, 4> *pi_out, int64_t x_lo,
    int64_t x_hi) {
  typename Inst::Key key;
  key.s0 = s0;
  for (int i = 0; i < kInBits; ++i) key.cws[i] = cws[i];
  key.cs = cs;
  key.ocw = ocw;

  InType x = BuildInput<InType>(x_lo, x_hi);

  auto res = vdpf.Eval(party, key, x);
  y_out[0] = res.y;
  pi_out[0] = res.pi_tilde;
}

// ---------------------------------------------------------------------------
// vdpf_gen
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> vdpf_gen(
    torch::Tensor s0s, int64_t alpha_lo, int64_t alpha_hi, torch::Tensor beta, bool retry) {
  TORCH_CHECK(s0s.is_cpu(), "s0s must be on CPU for Gen");
  TORCH_CHECK(beta.is_cpu(), "beta must be on CPU for Gen");
  TORCH_CHECK(s0s.dtype() == torch::kInt32, "s0s must be int32");
  TORCH_CHECK(beta.dtype() == torch::kInt32, "beta must be int32");
  TORCH_CHECK(s0s.sizes() == torch::IntArrayRef({2, 4}), "s0s must have shape (2, 4)");
  TORCH_CHECK(beta.sizes() == torch::IntArrayRef({4}), "beta must have shape (4,)");

  EnsureNonce();

  InType alpha = BuildInput<InType>(alpha_lo, alpha_hi);

  auto *s0s_ptr = reinterpret_cast<int4 *>(s0s.data_ptr<int32_t>());
  auto *beta_ptr = reinterpret_cast<int4 *>(beta.data_ptr<int32_t>());
  int4 b_buf = beta_ptr[0];

  // Gen writes the effective seeds of the successful attempt back into s0s.
  int4 seeds[2] = {s0s_ptr[0], s0s_ptr[1]};

  // Allocate outputs: kInBits correction words, each 32B = 8 int32.
  auto cws_tensor = torch::zeros({kInBits, 8}, torch::kInt32);
  auto cs_tensor = torch::zeros({4, 4}, torch::kInt32);
  auto ocw_tensor = torch::zeros({4}, torch::kInt32);

  VdpfInst vdpf = MakeVdpf();
  typename VdpfInst::Key k0, k1;
  int ret = vdpf.Gen(k0, k1, cuda::std::span<int4, 2>(seeds, 2), alpha, b_buf, retry);
  TORCH_CHECK(
      ret == 0, "vdpf Gen failed: bad seeds (all-zero cs or equal control bits); "
                "pass retry=true or resample s0s");

  auto *cws_ptr = reinterpret_cast<typename VdpfInst::Cw *>(cws_tensor.data_ptr<int32_t>());
  for (int i = 0; i < kInBits; ++i) cws_ptr[i] = k0.cws[i];
  auto *cs_out = reinterpret_cast<int4 *>(cs_tensor.data_ptr<int32_t>());
  for (int i = 0; i < 4; ++i) cs_out[i] = k0.cs[i];
  auto *ocw_out = reinterpret_cast<int4 *>(ocw_tensor.data_ptr<int32_t>());
  ocw_out[0] = k0.ocw;

  auto s0s_eff = torch::zeros({2, 4}, torch::kInt32);
  auto *eff_ptr = reinterpret_cast<int4 *>(s0s_eff.data_ptr<int32_t>());
  eff_ptr[0] = seeds[0];
  eff_ptr[1] = seeds[1];

  return std::make_tuple(s0s_eff, cws_tensor, cs_tensor, ocw_tensor);
}

// ---------------------------------------------------------------------------
// vdpf_eval
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> vdpf_eval(int64_t party, torch::Tensor s0, torch::Tensor cws,
    torch::Tensor cs, torch::Tensor ocw, int64_t x_lo, int64_t x_hi) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(cs.dtype() == torch::kInt32, "cs must be int32");
  TORCH_CHECK(ocw.dtype() == torch::kInt32, "ocw must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits, 8}), "cws must have shape (kInBits, 8)");
  TORCH_CHECK(cs.sizes() == torch::IntArrayRef({4, 4}), "cs must have shape (4, 4)");
  TORCH_CHECK(ocw.sizes() == torch::IntArrayRef({4}), "ocw must have shape (4,)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");
  TORCH_CHECK(s0.device() == cs.device(), "s0 and cs must be on the same device");
  TORCH_CHECK(s0.device() == ocw.device(), "s0 and ocw must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  auto y = torch::zeros({4}, torch::dtype(torch::kInt32).device(s0.device()));
  auto pi_tilde = torch::zeros({4, 4}, torch::dtype(torch::kInt32).device(s0.device()));

  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = FetchInt4(s0_ptr, s0.is_cuda());
  auto *cws_ptr = reinterpret_cast<const typename VdpfInst::Cw *>(cws.data_ptr<int32_t>());
  auto *cs_ptr = reinterpret_cast<int4 *>(cs.data_ptr<int32_t>());
  cuda::std::array<int4, 4> cs_val = FetchCs(cs_ptr, s0.is_cuda());
  auto *ocw_ptr = reinterpret_cast<int4 *>(ocw.data_ptr<int32_t>());
  int4 ocw_val = FetchInt4(ocw_ptr, s0.is_cuda());

  auto *y_out = reinterpret_cast<int4 *>(y.data_ptr<int32_t>());
  auto *pi_out = reinterpret_cast<cuda::std::array<int4, 4> *>(pi_tilde.data_ptr<int32_t>());

  if (s0.is_cuda()) {
    if constexpr (kGpuHashes) {
      VdpfInst vdpf = MakeVdpf();
      vdpf_eval_kernel<<<1, 1, 0, CurrentStream()>>>(
          vdpf, b, s0_val, cws_ptr, cs_val, ocw_val, y_out, pi_out, x_lo, x_hi);
      auto err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "vdpf_eval_kernel launch failed: ", cudaGetErrorString(err));
    } else {
      TORCH_CHECK(false, "sha256 is CPU-only");
    }
    return std::make_tuple(y, pi_tilde);
  }

  // CPU path.
  VdpfInst vdpf = MakeVdpf();
  typename VdpfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i < kInBits; ++i) key.cws[i] = cws_ptr[i];
  key.cs = cs_val;
  key.ocw = ocw_val;
  InType x = BuildInput<InType>(x_lo, x_hi);
  auto res = vdpf.Eval(b, key, x);
  y_out[0] = res.y;
  pi_out[0] = res.pi_tilde;

  return std::make_tuple(y, pi_tilde);
}

// ---------------------------------------------------------------------------
// vdpf_eval_all (chain proof, CPU only)
// ---------------------------------------------------------------------------

static torch::Tensor vdpf_eval_all(int64_t party, torch::Tensor s0, torch::Tensor cws, torch::Tensor cs,
    torch::Tensor ocw) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.is_cpu(), "vdpf eval_all (chain proof) is CPU-only");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(cs.dtype() == torch::kInt32, "cs must be int32");
  TORCH_CHECK(ocw.dtype() == torch::kInt32, "ocw must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits, 8}), "cws must have shape (kInBits, 8)");
  TORCH_CHECK(cs.sizes() == torch::IntArrayRef({4, 4}), "cs must have shape (4, 4)");
  TORCH_CHECK(ocw.sizes() == torch::IntArrayRef({4}), "ocw must have shape (4,)");

  EnsureNonce();
  bool b = (party == 1);

  int64_t n = 1LL << kInBits;
  auto ys = torch::zeros({n, 4}, torch::kInt32);

  VdpfInst vdpf = MakeVdpf();
  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = s0_ptr[0];
  auto *cws_ptr = reinterpret_cast<const typename VdpfInst::Cw *>(cws.data_ptr<int32_t>());
  auto *cs_ptr = reinterpret_cast<int4 *>(cs.data_ptr<int32_t>());
  auto *ocw_ptr = reinterpret_cast<int4 *>(ocw.data_ptr<int32_t>());
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

  typename VdpfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i < kInBits; ++i) key.cws[i] = cws_ptr[i];
  for (int i = 0; i < 4; ++i) key.cs[i] = cs_ptr[i];
  key.ocw = ocw_ptr[0];

  cuda::std::array<int4, 4> pi;
  vdpf.EvalAll(b, key, cuda::std::span<int4>(ys_ptr, n), pi);

  return ys;
}

// ---------------------------------------------------------------------------
// vdpf_eval_all_merkle
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> vdpf_eval_all_merkle(int64_t party, torch::Tensor s0,
    torch::Tensor cws, torch::Tensor cs, torch::Tensor ocw) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(cs.dtype() == torch::kInt32, "cs must be int32");
  TORCH_CHECK(ocw.dtype() == torch::kInt32, "ocw must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits, 8}), "cws must have shape (kInBits, 8)");
  TORCH_CHECK(cs.sizes() == torch::IntArrayRef({4, 4}), "cs must have shape (4, 4)");
  TORCH_CHECK(ocw.sizes() == torch::IntArrayRef({4}), "ocw must have shape (4,)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");
  TORCH_CHECK(s0.device() == cs.device(), "s0 and cs must be on the same device");
  TORCH_CHECK(s0.device() == ocw.device(), "s0 and ocw must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  int64_t n = 1LL << kInBits;
  auto ys = torch::zeros({n, 4}, torch::dtype(torch::kInt32).device(s0.device()));
  auto pi = torch::zeros({4, 4}, torch::dtype(torch::kInt32).device(s0.device()));

  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = FetchInt4(s0_ptr, s0.is_cuda());
  auto *cws_ptr = reinterpret_cast<const typename VdpfInst::Cw *>(cws.data_ptr<int32_t>());
  auto *cs_ptr = reinterpret_cast<int4 *>(cs.data_ptr<int32_t>());
  cuda::std::array<int4, 4> cs_val = FetchCs(cs_ptr, s0.is_cuda());
  auto *ocw_ptr = reinterpret_cast<int4 *>(ocw.data_ptr<int32_t>());
  int4 ocw_val = FetchInt4(ocw_ptr, s0.is_cuda());

  if (s0.is_cuda()) {
    if constexpr (kGpuHashes) {
      if constexpr (kInBits <= 22) {
        // VDPF-specific tiling: the frontier depth caps at 14 so the single
        // block folding the per-block roots stays within (1 << Z) <= bs * bs.
        constexpr int kZ = kInBits < 14 ? kInBits : 14;
        constexpr int kB1 = kZ >= 7 ? kZ - 7 : 0;
        constexpr int kBS = 1 << (kZ - kB1);
        VdpfInst vdpf = MakeVdpf();
        fss::gpu::VdpfEvalAllMerkleGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, cs_val, ocw_val,
            reinterpret_cast<int4 *>(ys.data_ptr<int32_t>()),
            reinterpret_cast<cuda::std::array<int4, 4> *>(pi.data_ptr<int32_t>()), vdpf, CurrentStream());
        auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "vdpf eval_all_merkle launch failed: ", cudaGetErrorString(err));
      } else {
        TORCH_CHECK(false, "GPU eval_all_merkle supports in_bits <= 22, got ", kInBits);
      }
    } else {
      TORCH_CHECK(false, "sha256 is CPU-only");
    }
    return std::make_tuple(ys, pi);
  }

  // CPU path.
  VdpfInst vdpf = MakeVdpf();
  typename VdpfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i < kInBits; ++i) key.cws[i] = cws_ptr[i];
  key.cs = cs_val;
  key.ocw = ocw_val;

  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());
  auto *pi_out = reinterpret_cast<cuda::std::array<int4, 4> *>(pi.data_ptr<int32_t>());
  vdpf.EvalAllMerkle(b, key, cuda::std::span<int4>(ys_ptr, n), pi_out[0]);

  return std::make_tuple(ys, pi);
}

// ---------------------------------------------------------------------------
// vdpf_prove
// ---------------------------------------------------------------------------

static torch::Tensor vdpf_prove(torch::Tensor pi_tildes, torch::Tensor cs, bool merkle) {
  TORCH_CHECK(pi_tildes.is_cpu(), "pi_tildes must be on CPU for prove");
  TORCH_CHECK(cs.is_cpu(), "cs must be on CPU for prove");
  TORCH_CHECK(pi_tildes.dtype() == torch::kInt32, "pi_tildes must be int32");
  TORCH_CHECK(cs.dtype() == torch::kInt32, "cs must be int32");
  TORCH_CHECK(pi_tildes.dim() == 3 && pi_tildes.size(1) == 4 && pi_tildes.size(2) == 4,
      "pi_tildes must have shape (L, 4, 4)");
  TORCH_CHECK(pi_tildes.size(0) >= 1, "pi_tildes must contain at least one per-point hash");
  TORCH_CHECK(cs.sizes() == torch::IntArrayRef({4, 4}), "cs must have shape (4, 4)");

  EnsureNonce();

  auto pi = torch::zeros({4, 4}, torch::kInt32);
  auto *tildes_ptr = reinterpret_cast<const cuda::std::array<int4, 4> *>(pi_tildes.data_ptr<int32_t>());
  auto *cs_ptr = reinterpret_cast<const int4 *>(cs.data_ptr<int32_t>());
  auto *pi_out = reinterpret_cast<cuda::std::array<int4, 4> *>(pi.data_ptr<int32_t>());

  VdpfInst vdpf = MakeVdpf();
  cuda::std::span<const cuda::std::array<int4, 4>> tildes(tildes_ptr, pi_tildes.size(0));
  cuda::std::span<const int4, 4> cs_span(cs_ptr, 4);
  if (merkle) {
    vdpf.ProveMerkle(tildes, cs_span, pi_out[0]);
  } else {
    vdpf.Prove(tildes, cs_span, pi_out[0]);
  }

  return pi;
}

// ---------------------------------------------------------------------------
// vdpf_verify
// ---------------------------------------------------------------------------

static bool vdpf_verify(torch::Tensor pi0, torch::Tensor pi1) {
  TORCH_CHECK(pi0.is_cpu() && pi1.is_cpu(), "proofs must be on CPU for verify");
  TORCH_CHECK(pi0.dtype() == torch::kInt32, "pi0 must be int32");
  TORCH_CHECK(pi1.dtype() == torch::kInt32, "pi1 must be int32");
  TORCH_CHECK(pi0.sizes() == torch::IntArrayRef({4, 4}), "pi0 must have shape (4, 4)");
  TORCH_CHECK(pi1.sizes() == torch::IntArrayRef({4, 4}), "pi1 must have shape (4, 4)");

  auto *p0 = reinterpret_cast<const int4 *>(pi0.data_ptr<int32_t>());
  auto *p1 = reinterpret_cast<const int4 *>(pi1.data_ptr<int32_t>());
  return VdpfInst::Verify(cuda::std::span<const int4, 4>(p0, 4), cuda::std::span<const int4, 4>(p1, 4));
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &vdpf_gen, "VDPF key generation");
  m.def("eval", &vdpf_eval, "VDPF single-point evaluation");
  m.def("eval_all", &vdpf_eval_all, "VDPF full-domain evaluation (chain proof)");
  m.def("eval_all_merkle", &vdpf_eval_all_merkle, "VDPF full-domain evaluation (Merkle proof)");
  m.def("prove", &vdpf_prove, "VDPF proof accumulation");
  m.def("verify", &vdpf_verify, "VDPF proof verification");
}
