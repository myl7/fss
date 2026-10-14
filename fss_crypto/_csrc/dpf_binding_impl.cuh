// SPDX-License-Identifier: Apache-2.0
// DPF binding implementation. Included by JIT-generated .cu files after type
// aliases (kInBits, InType, GroupType, PrgType, DpfInst) are defined.

#include "binding_common.cuh"
#include <fss/eval_all_gpu.cuh>
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// GPU eval kernel
// ---------------------------------------------------------------------------

__global__ void dpf_eval_kernel(
    bool party, int4 s0, const typename DpfInst::Cw *cws, int4 *out, int nonce0, int nonce1, int64_t x_lo,
    int64_t x_hi) {
  DpfInst dpf{PrgType(nonce0, nonce1)};

  typename DpfInst::Key key;
  key.s0 = s0;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws[i];

  InType x = BuildInput<InType>(x_lo, x_hi);

  int4 result = dpf.Eval(party, key, x);
  out[0] = result;
}

// ---------------------------------------------------------------------------
// dpf_gen
// ---------------------------------------------------------------------------

static torch::Tensor dpf_gen(torch::Tensor s0s, int64_t alpha_lo, int64_t alpha_hi, torch::Tensor beta) {
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
  auto *cws = reinterpret_cast<typename DpfInst::Cw *>(cws_tensor.data_ptr<int32_t>());

  DpfInst dpf{PrgType(g_nonce0, g_nonce1)};
  typename DpfInst::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s_ptr, 2), alpha, b_buf);
  for (int i = 0; i <= kInBits; ++i) cws[i] = k0.cws[i];

  return cws_tensor;
}

// ---------------------------------------------------------------------------
// dpf_eval
// ---------------------------------------------------------------------------

static torch::Tensor dpf_eval(int64_t party, torch::Tensor s0, torch::Tensor cws, int64_t x_lo, int64_t x_hi) {
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
    int4 s0_val = *s0_ptr;

    dpf_eval_kernel<<<1, 1, 0, CurrentStream()>>>(b, s0_val,
        reinterpret_cast<const typename DpfInst::Cw *>(cws.data_ptr<int32_t>()),
        reinterpret_cast<int4 *>(out.data_ptr<int32_t>()), g_nonce0, g_nonce1, x_lo, x_hi);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dpf_eval_kernel launch failed: ", cudaGetErrorString(err));

    return out;
  }

  // CPU path.
  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = s0_ptr[0];
  auto *cws_ptr = reinterpret_cast<const typename DpfInst::Cw *>(cws.data_ptr<int32_t>());

  InType x = BuildInput<InType>(x_lo, x_hi);

  DpfInst dpf{PrgType(g_nonce0, g_nonce1)};
  typename DpfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws_ptr[i];
  int4 result = dpf.Eval(b, key, x);

  auto out = torch::zeros({4}, torch::kInt32);
  auto *out_ptr = reinterpret_cast<int4 *>(out.data_ptr<int32_t>());
  out_ptr[0] = result;
  return out;
}

// ---------------------------------------------------------------------------
// dpf_eval_all
// ---------------------------------------------------------------------------

static torch::Tensor dpf_eval_all(int64_t party, torch::Tensor s0, torch::Tensor cws) {
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

  DpfInst dpf{PrgType(g_nonce0, g_nonce1)};
  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = s0_ptr[0];
  auto *cws_ptr = reinterpret_cast<const typename DpfInst::Cw *>(cws.data_ptr<int32_t>());
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

  if (s0.is_cuda()) {
    if constexpr (kInBits <= 24) {
      constexpr int kZ = EvalAllZ(kInBits), kB1 = EvalAllB1(kInBits), kBS = EvalAllBS(kInBits);
      fss::gpu::DpfEvalAllGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, ys_ptr, dpf, CurrentStream());
      auto err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "dpf eval_all launch failed: ", cudaGetErrorString(err));
    } else {
      TORCH_CHECK(false, "GPU EvalAll supports in_bits <= 24, got ", kInBits);
    }
    return ys;
  }

  typename DpfInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws_ptr[i];
  dpf.EvalAll(b, key, cuda::std::span<int4>(ys_ptr, n));

  return ys;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &dpf_gen, "DPF key generation");
  m.def("eval", &dpf_eval, "DPF single-point evaluation");
  m.def("eval_all", &dpf_eval_all, "DPF full-domain evaluation");
}
