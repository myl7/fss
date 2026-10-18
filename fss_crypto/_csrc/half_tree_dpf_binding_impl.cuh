// SPDX-License-Identifier: Apache-2.0
// Half-Tree DPF binding implementation. Included by JIT-generated .cu files
// after type aliases (kInBits, InType, GroupType, PrgType, HtDpfInst) are
// defined.

#include "binding_common.cuh"
#include <fss/eval_all_gpu.cuh>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <tuple>

// ---------------------------------------------------------------------------
// GPU eval kernel
// ---------------------------------------------------------------------------

__global__ void half_tree_dpf_eval_kernel(bool party, const int4 *s0, const typename HtDpfInst::Cw *cws,
    const int4 *ocw, int4 *out, PrgType prg, int4 hash_key, int64_t x_lo, int64_t x_hi) {
  HtDpfInst dpf{prg, hash_key};

  typename HtDpfInst::Key key;
  key.s0 = s0[0];
  key.ocw = ocw[0];
  for (int i = 0; i < kInBits; ++i) key.cws[i] = cws[i];

  InType x = BuildInput<InType>(x_lo, x_hi);

  int4 result = dpf.Eval(party, key, x);
  out[0] = result;
}

// ---------------------------------------------------------------------------
// half_tree_dpf_gen
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> half_tree_dpf_gen(
    torch::Tensor s0s, int64_t alpha_lo, int64_t alpha_hi, torch::Tensor beta) {
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

  // Allocate outputs: kInBits correction words, each 32B = 8 int32, and the
  // output correction word.
  auto cws_tensor = torch::zeros({kInBits, 8}, torch::kInt32);
  auto ocw_tensor = torch::zeros({4}, torch::kInt32);
  auto *cws = reinterpret_cast<typename HtDpfInst::Cw *>(cws_tensor.data_ptr<int32_t>());
  auto *ocw = reinterpret_cast<int4 *>(ocw_tensor.data_ptr<int32_t>());

  HtDpfInst dpf{MakePrg(), kHashIv[0]};
  typename HtDpfInst::Key k0, k1;
  dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s_ptr, 2), alpha, b_buf);
  for (int i = 0; i < kInBits; ++i) cws[i] = k0.cws[i];
  ocw[0] = k0.ocw;

  return {cws_tensor, ocw_tensor};
}

// ---------------------------------------------------------------------------
// half_tree_dpf_eval
// ---------------------------------------------------------------------------

static torch::Tensor half_tree_dpf_eval(
    int64_t party, torch::Tensor s0, torch::Tensor cws, torch::Tensor ocw, int64_t x_lo, int64_t x_hi) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(ocw.dtype() == torch::kInt32, "ocw must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits, 8}), "cws must have shape (kInBits, 8)");
  TORCH_CHECK(ocw.sizes() == torch::IntArrayRef({4}), "ocw must have shape (4,)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");
  TORCH_CHECK(s0.device() == ocw.device(), "s0 and ocw must be on the same device");

  EnsureNonce();
  c10::cuda::OptionalCUDAGuard device_guard;
  if (s0.is_cuda()) device_guard.set_device(s0.device());
  CheckCudaPrg(s0);
  bool b = (party == 1);

  auto *s0_ptr = reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>());
  auto *cws_ptr = reinterpret_cast<const typename HtDpfInst::Cw *>(cws.data_ptr<int32_t>());
  auto *ocw_ptr = reinterpret_cast<const int4 *>(ocw.data_ptr<int32_t>());

  if (s0.is_cuda()) {
    auto out = torch::zeros({4}, torch::dtype(torch::kInt32).device(s0.device()));

    half_tree_dpf_eval_kernel<<<1, 1, 0, CurrentStream()>>>(b, s0_ptr, cws_ptr, ocw_ptr,
        reinterpret_cast<int4 *>(out.data_ptr<int32_t>()), MakePrg(), kHashIv[0], x_lo, x_hi);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "half_tree_dpf_eval_kernel launch failed: ", cudaGetErrorString(err));

    return out;
  }

  // CPU path.
  InType x = BuildInput<InType>(x_lo, x_hi);

  HtDpfInst dpf{MakePrg(), kHashIv[0]};
  typename HtDpfInst::Key key;
  key.s0 = s0_ptr[0];
  key.ocw = ocw_ptr[0];
  for (int i = 0; i < kInBits; ++i) key.cws[i] = cws_ptr[i];
  int4 result = dpf.Eval(b, key, x);

  auto out = torch::zeros({4}, torch::kInt32);
  auto *out_ptr = reinterpret_cast<int4 *>(out.data_ptr<int32_t>());
  out_ptr[0] = result;
  return out;
}

// ---------------------------------------------------------------------------
// half_tree_dpf_eval_all
// ---------------------------------------------------------------------------

static torch::Tensor half_tree_dpf_eval_all(int64_t party, torch::Tensor s0, torch::Tensor cws, torch::Tensor ocw) {
  if constexpr (kInBits > kMaxEvalAllBits) {
    TORCH_CHECK(false, "half_tree_dpf.eval_all cannot represent its full-domain size for expanded bits=", kInBits);
    return {};
  } else {
    TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
    TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
    TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
    TORCH_CHECK(ocw.dtype() == torch::kInt32, "ocw must be int32");
    TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
    TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits, 8}), "cws must have shape (kInBits, 8)");
    TORCH_CHECK(ocw.sizes() == torch::IntArrayRef({4}), "ocw must have shape (4,)");
    TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");
    TORCH_CHECK(s0.device() == ocw.device(), "s0 and ocw must be on the same device");

    EnsureNonce();
    c10::cuda::OptionalCUDAGuard device_guard;
    if (s0.is_cuda()) device_guard.set_device(s0.device());
    CheckCudaPrg(s0);
    bool b = (party == 1);

    int64_t n = 0;
    if constexpr (kInBits <= kMaxEvalAllBits) n = 1LL << kInBits;
    else TORCH_CHECK(false, "half_tree_dpf.eval_all cannot represent the 2^", kInBits, "-entry output size");
    if (s0.is_cuda())
      TORCH_CHECK(kInBits - 1 <= kMaxCudaEvalAllBits,
          "half_tree_dpf.eval_all on CUDA supports in_bits <= ", kMaxCudaEvalAllBits + 1, " (grid index width), got ",
          kInBits);

    auto options = torch::dtype(torch::kInt32).device(s0.device());
    auto ys = torch::zeros({n, 4}, options);

    auto *s0_ptr = reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>());
    auto *cws_ptr = reinterpret_cast<const typename HtDpfInst::Cw *>(cws.data_ptr<int32_t>());
    auto *ocw_ptr = reinterpret_cast<const int4 *>(ocw.data_ptr<int32_t>());
    auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

    if (s0.is_cuda()) {
      // s0 and ocw travel as kernel parameters, so fetch the two 16B values
      // from the device before the launch.
      int4 s0_val = FetchInt4(s0_ptr, true);
      int4 ocw_val = FetchInt4(ocw_ptr, true);

      // Tiling is computed on the scheme depth, one level below the leaves.
      constexpr int kDepth = kInBits - 1;
      constexpr int kZ = WideDomainEvalAllZ(kDepth), kB1 = WideDomainEvalAllB1(kDepth),
                    kBS = WideDomainEvalAllBS(kDepth);
      HtDpfInst dpf{MakePrg(), kHashIv[0]};
      fss::gpu::HalfTreeDpfEvalAllGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, ocw_val, ys_ptr, dpf, CurrentStream());
      auto err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "half_tree_dpf eval_all launch failed: ", cudaGetErrorString(err));
      return ys;
    }

    HtDpfInst dpf{MakePrg(), kHashIv[0]};
    typename HtDpfInst::Key key;
    key.s0 = s0_ptr[0];
    key.ocw = ocw_ptr[0];
    for (int i = 0; i < kInBits; ++i) key.cws[i] = cws_ptr[i];
    dpf.EvalAll(b, key, cuda::std::span<int4>(ys_ptr, n));

    return ys;
  }
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &half_tree_dpf_gen, "Half-Tree DPF key generation");
  m.def("eval", &half_tree_dpf_eval, "Half-Tree DPF single-point evaluation");
  m.def("eval_all", &half_tree_dpf_eval_all, "Half-Tree DPF full-domain evaluation");
}
