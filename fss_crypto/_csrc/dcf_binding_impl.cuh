// SPDX-License-Identifier: Apache-2.0
// DCF binding implementation. Included by JIT-generated .cu files after type
// aliases (kInBits, InType, GroupType, PrgType, kPred, DcfInst) are defined.

#include "binding_common.cuh"
#include <fss/eval_all_gpu.cuh>
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// GPU eval kernel
// ---------------------------------------------------------------------------

__global__ void dcf_eval_kernel(
    bool party, const int4 *s0, const typename DcfInst::Cw *cws, int4 *out, int nonce0, int nonce1, int64_t x_lo,
    int64_t x_hi) {
  DcfInst dpf{PrgType(nonce0, nonce1)};

  typename DcfInst::Key key;
  key.s0 = s0[0];
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws[i];

  InType x = BuildInput<InType>(x_lo, x_hi);

  int4 result = dpf.Eval(party, key, x);
  out[0] = result;
}

// ---------------------------------------------------------------------------
// dcf_gen
// ---------------------------------------------------------------------------

static torch::Tensor dcf_gen(torch::Tensor s0s, int64_t alpha_lo, int64_t alpha_hi, torch::Tensor beta) {
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

  // Allocate output: (kInBits+1) correction words, each 32B = 8 int32.
  auto cws_tensor = torch::zeros({kInBits + 1, 8}, torch::kInt32);
  auto *cws = reinterpret_cast<typename DcfInst::Cw *>(cws_tensor.data_ptr<int32_t>());

  DcfInst dpf{PrgType(g_nonce0, g_nonce1)};
  typename DcfInst::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s_ptr, 2), alpha, b_buf);
  for (int i = 0; i <= kInBits; ++i) cws[i] = k0.cws[i];

  return cws_tensor;
}

// ---------------------------------------------------------------------------
// dpf_eval
// ---------------------------------------------------------------------------

static torch::Tensor dcf_eval(int64_t party, torch::Tensor s0, torch::Tensor cws, int64_t x_lo, int64_t x_hi) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits + 1, 8}), "cws must have shape (kInBits+1, 8)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  if (s0.is_cuda()) {
    auto out = torch::zeros({4}, torch::dtype(torch::kInt32).device(s0.device()));

    auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());

    dcf_eval_kernel<<<1, 1, 0, CurrentStream()>>>(b, s0_ptr,
        reinterpret_cast<const typename DcfInst::Cw *>(cws.data_ptr<int32_t>()),
        reinterpret_cast<int4 *>(out.data_ptr<int32_t>()), g_nonce0, g_nonce1, x_lo, x_hi);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dcf_eval_kernel launch failed: ", cudaGetErrorString(err));

    return out;
  }

  // CPU path.
  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = s0_ptr[0];
  auto *cws_ptr = reinterpret_cast<const typename DcfInst::Cw *>(cws.data_ptr<int32_t>());

  InType x = BuildInput<InType>(x_lo, x_hi);

  DcfInst dpf{PrgType(g_nonce0, g_nonce1)};
  typename DcfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws_ptr[i];
  int4 result = dpf.Eval(b, key, x);

  auto out = torch::zeros({4}, torch::kInt32);
  auto *out_ptr = reinterpret_cast<int4 *>(out.data_ptr<int32_t>());
  out_ptr[0] = result;
  return out;
}

// ---------------------------------------------------------------------------
// dcf_eval_all
// ---------------------------------------------------------------------------

// The GPU full-domain wrapper is not templated on the predicate (only the
// default lt Dcf instantiates against it); the discarded branch of a template
// is not type-checked, unlike a plain function's.
template <int kInBits, fss::DcfPred kPred, typename Inst>
void DcfEvalAllGpuIf(bool b, int4 s0_val, const typename Inst::Cw *cws_ptr, int4 *ys_ptr,
    const Inst &dpf, cudaStream_t stream) {
  if constexpr (kInBits <= 24 && kPred == fss::DcfPred::kLt) {
    constexpr int kZ = EvalAllZ(kInBits), kB1 = EvalAllB1(kInBits), kBS = EvalAllBS(kInBits);
    fss::gpu::DcfEvalAllGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, ys_ptr, dpf, stream);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dcf eval_all launch failed: ", cudaGetErrorString(err));
  } else {
    TORCH_CHECK(false, "GPU EvalAll supports in_bits <= 24 and the lt predicate");
  }
}


static torch::Tensor dcf_eval_all(int64_t party, torch::Tensor s0, torch::Tensor cws) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits + 1, 8}), "cws must have shape (kInBits+1, 8)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  int64_t n = 1LL << kInBits;
  auto options = torch::dtype(torch::kInt32).device(s0.device());
  auto ys = torch::zeros({n, 4}, options);

  DcfInst dpf{PrgType(g_nonce0, g_nonce1)};
  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  auto *cws_ptr = reinterpret_cast<const typename DcfInst::Cw *>(cws.data_ptr<int32_t>());
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());
  int4 s0_val = FetchInt4(s0_ptr, s0.is_cuda());
  if (s0.is_cuda()) {
    DcfEvalAllGpuIf<kInBits, kPred>(b, s0_val, cws_ptr, ys_ptr, dpf, CurrentStream());
    return ys;
  }

  typename DcfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws_ptr[i];
  dpf.EvalAll(b, key, cuda::std::span<int4>(ys_ptr, n));

  return ys;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &dcf_gen, "DCF key generation");
  m.def("eval", &dcf_eval, "DCF single-point evaluation");
  m.def("eval_all", &dcf_eval_all, "DCF full-domain evaluation");
}
