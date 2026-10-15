// SPDX-License-Identifier: Apache-2.0
// Packed Half-Tree DPF binding implementation. Included by JIT-generated .cu
// files after type aliases (kInBits, InType, kOutBits, PrgType, PackedInst)
// are defined.

#include "binding_common.cuh"
#include <fss/eval_all_gpu.cuh>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// GPU eval kernel
// ---------------------------------------------------------------------------

__global__ void packed_half_tree_dpf_eval_kernel(
    bool party, const int4 *s0, const typename PackedInst::Cw *cws, const int4 *fcw, int4 *out, int nonce0,
    int nonce1, int4 hash_key, int64_t x_lo, int64_t x_hi) {
  PackedInst dpf{PrgType(nonce0, nonce1), hash_key};

  typename PackedInst::Key key;
  key.s0 = s0[0];
  key.fcw = fcw[0];
  for (int i = 0; i < PackedInst::kDepth; ++i) key.cws[i] = cws[i];

  InType x = BuildInput<InType>(x_lo, x_hi);

  int4 result = dpf.Eval(party, key, x);
  out[0] = result;
}

// ---------------------------------------------------------------------------
// packed_half_tree_dpf_gen
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> packed_half_tree_dpf_gen(
    torch::Tensor s0s, int64_t alpha_lo, int64_t alpha_hi, int64_t beta) {
  TORCH_CHECK(s0s.is_cpu(), "s0s must be on CPU for Gen");
  TORCH_CHECK(s0s.dtype() == torch::kInt32, "s0s must be int32");
  TORCH_CHECK(s0s.sizes() == torch::IntArrayRef({2, 4}), "s0s must have shape (2, 4)");
  TORCH_CHECK(beta >= 0 && static_cast<uint64_t>(beta) <= PackedInst::kBetaMax, "beta must be in [0, ",
      PackedInst::kBetaMax, "], got ", beta);

  EnsureNonce();

  InType alpha = BuildInput<InType>(alpha_lo, alpha_hi);

  auto *s0s_ptr = reinterpret_cast<int4 *>(s0s.data_ptr<int32_t>());

  // Allocate outputs: kDepth correction words, each 16B = 4 int32, and the
  // final correction word.
  auto cws_tensor = torch::zeros({PackedInst::kDepth, 4}, torch::kInt32);
  auto fcw_tensor = torch::zeros({4}, torch::kInt32);
  auto *cws = reinterpret_cast<typename PackedInst::Cw *>(cws_tensor.data_ptr<int32_t>());
  auto *fcw = reinterpret_cast<int4 *>(fcw_tensor.data_ptr<int32_t>());

  PackedInst dpf{PrgType(g_nonce0, g_nonce1), kHashIv[0]};
  typename PackedInst::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s_ptr, 2), alpha, static_cast<uint64_t>(beta));
  for (int i = 0; i < PackedInst::kDepth; ++i) cws[i] = k0.cws[i];
  fcw[0] = k0.fcw;

  return {cws_tensor, fcw_tensor};
}

// ---------------------------------------------------------------------------
// packed_half_tree_dpf_eval
// ---------------------------------------------------------------------------

static torch::Tensor packed_half_tree_dpf_eval(
    int64_t party, torch::Tensor s0, torch::Tensor cws, torch::Tensor fcw, int64_t x_lo, int64_t x_hi) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(fcw.dtype() == torch::kInt32, "fcw must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({PackedInst::kDepth, 4}), "cws must have shape (kDepth, 4)");
  TORCH_CHECK(fcw.sizes() == torch::IntArrayRef({4}), "fcw must have shape (4,)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");
  TORCH_CHECK(s0.device() == fcw.device(), "s0 and fcw must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  auto *s0_ptr = reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>());
  auto *cws_ptr = reinterpret_cast<const typename PackedInst::Cw *>(cws.data_ptr<int32_t>());
  auto *fcw_ptr = reinterpret_cast<const int4 *>(fcw.data_ptr<int32_t>());

  if (s0.is_cuda()) {
    auto out = torch::zeros({4}, torch::dtype(torch::kInt32).device(s0.device()));

    packed_half_tree_dpf_eval_kernel<<<1, 1, 0, CurrentStream()>>>(
        b, s0_ptr, cws_ptr, fcw_ptr, reinterpret_cast<int4 *>(out.data_ptr<int32_t>()), g_nonce0, g_nonce1,
        kHashIv[0], x_lo, x_hi);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "packed_half_tree_dpf_eval_kernel launch failed: ", cudaGetErrorString(err));

    return out;
  }

  // CPU path.
  InType x = BuildInput<InType>(x_lo, x_hi);

  PackedInst dpf{PrgType(g_nonce0, g_nonce1), kHashIv[0]};
  typename PackedInst::Key key;
  key.s0 = s0_ptr[0];
  key.fcw = fcw_ptr[0];
  for (int i = 0; i < PackedInst::kDepth; ++i) key.cws[i] = cws_ptr[i];
  int4 result = dpf.Eval(b, key, x);

  auto out = torch::zeros({4}, torch::kInt32);
  auto *out_ptr = reinterpret_cast<int4 *>(out.data_ptr<int32_t>());
  out_ptr[0] = result;
  return out;
}

// ---------------------------------------------------------------------------
// packed_half_tree_dpf_eval_all
// ---------------------------------------------------------------------------

static torch::Tensor packed_half_tree_dpf_eval_all(int64_t party, torch::Tensor s0, torch::Tensor cws,
    torch::Tensor fcw) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(fcw.dtype() == torch::kInt32, "fcw must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({PackedInst::kDepth, 4}), "cws must have shape (kDepth, 4)");
  TORCH_CHECK(fcw.sizes() == torch::IntArrayRef({4}), "fcw must have shape (4,)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");
  TORCH_CHECK(s0.device() == fcw.device(), "s0 and fcw must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  constexpr int kDepth = PackedInst::kDepth;
  int64_t n = 0;
  if constexpr (kDepth <= kMaxEvalAllBits) n = 1LL << kDepth;
  else TORCH_CHECK(false, "packed_half_tree_dpf.eval_all cannot represent the 2^", kDepth, "-block output size");
  if (s0.is_cuda())
    TORCH_CHECK(kDepth <= kMaxCudaEvalAllBits, "packed_half_tree_dpf.eval_all on CUDA supports tree depth <= ",
        kMaxCudaEvalAllBits, " (grid index width), got ", kDepth);

  auto options = torch::dtype(torch::kInt32).device(s0.device());
  auto ys = torch::zeros({n, 4}, options);

  auto *s0_ptr = reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>());
  auto *cws_ptr = reinterpret_cast<const typename PackedInst::Cw *>(cws.data_ptr<int32_t>());
  auto *fcw_ptr = reinterpret_cast<const int4 *>(fcw.data_ptr<int32_t>());
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

  if (s0.is_cuda()) {
    // s0 and fcw travel as kernel parameters, so fetch the two 16B values
    // from the device before the launch.
    int4 s0_val, fcw_val;
    auto cpy_err = cudaMemcpy(&s0_val, s0_ptr, sizeof(int4), cudaMemcpyDeviceToHost);
    TORCH_CHECK(cpy_err == cudaSuccess, "packed_half_tree_dpf eval_all s0 copy failed: ", cudaGetErrorString(cpy_err));
    cpy_err = cudaMemcpy(&fcw_val, fcw_ptr, sizeof(int4), cudaMemcpyDeviceToHost);
    TORCH_CHECK(
        cpy_err == cudaSuccess, "packed_half_tree_dpf eval_all fcw copy failed: ", cudaGetErrorString(cpy_err));

    constexpr int kZ = WideDomainEvalAllZ(kDepth), kB1 = WideDomainEvalAllB1(kDepth),
        kBS = WideDomainEvalAllBS(kDepth);
    PackedInst dpf{PrgType(g_nonce0, g_nonce1), kHashIv[0]};
    fss::gpu::PackedHalfTreeDpfEvalAllGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, fcw_val, ys_ptr, dpf, CurrentStream());
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "packed_half_tree_dpf eval_all launch failed: ", cudaGetErrorString(err));
    return ys;
  }

  PackedInst dpf{PrgType(g_nonce0, g_nonce1), kHashIv[0]};
  typename PackedInst::Key key;
  key.s0 = s0_ptr[0];
  key.fcw = fcw_ptr[0];
  for (int i = 0; i < kDepth; ++i) key.cws[i] = cws_ptr[i];
  dpf.EvalAll(b, key, cuda::std::span<int4>(ys_ptr, n));

  return ys;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &packed_half_tree_dpf_gen, "Packed Half-Tree DPF key generation");
  m.def("eval", &packed_half_tree_dpf_eval, "Packed Half-Tree DPF single-point evaluation");
  m.def("eval_all", &packed_half_tree_dpf_eval_all, "Packed Half-Tree DPF full-domain evaluation");
}
