// SPDX-License-Identifier: Apache-2.0
// Grotto DCF binding implementation. Included by JIT-generated .cu files after
// type aliases (kInBits, InType, PrgType, GrottoInst) are defined.

#include "binding_common.cuh"
#include <fss/eval_all_gpu.cuh>
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// grotto_gen
// ---------------------------------------------------------------------------

static torch::Tensor grotto_gen(torch::Tensor s0s, int64_t alpha_lo, int64_t alpha_hi) {
  TORCH_CHECK(s0s.is_cpu(), "s0s must be on CPU for Gen");
  TORCH_CHECK(s0s.dtype() == torch::kInt32, "s0s must be int32");
  TORCH_CHECK(s0s.sizes() == torch::IntArrayRef({2, 4}), "s0s must have shape (2, 4)");

  EnsureNonce();

  InType alpha = BuildInput<InType>(alpha_lo, alpha_hi);

  auto *s0s_ptr = reinterpret_cast<int4 *>(s0s.data_ptr<int32_t>());

  // Allocate output: (kInBits+1) correction words, each 32B = 8 int32.
  auto cws_tensor = torch::zeros({kInBits + 1, 8}, torch::kInt32);
  auto *cws = reinterpret_cast<typename GrottoInst::Cw *>(cws_tensor.data_ptr<int32_t>());

  GrottoInst grotto{PrgType(g_nonce0, g_nonce1)};
  typename GrottoInst::Key k0, k1;
  grotto.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s_ptr, 2), alpha);
  for (int i = 0; i <= kInBits; ++i) cws[i] = k0.cws[i];

  return cws_tensor;
}

// ---------------------------------------------------------------------------
// grotto_preprocess
// ---------------------------------------------------------------------------

static torch::Tensor grotto_preprocess(int64_t party, torch::Tensor s0, torch::Tensor cws) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits + 1, 8}), "cws must have shape (kInBits+1, 8)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = FetchInt4(s0_ptr, s0.is_cuda());
  auto *cws_ptr = reinterpret_cast<const typename GrottoInst::Cw *>(cws.data_ptr<int32_t>());

  int64_t n = 1LL << kInBits;
  GrottoInst grotto{PrgType(g_nonce0, g_nonce1)};

  if (s0.is_cuda()) {
    if constexpr (kInBits <= 24) {
      constexpr int kZ = EvalAllZ(kInBits), kB1 = EvalAllB1(kInBits), kBS = EvalAllBS(kInBits);
      auto tree = torch::zeros({2 * n - 1}, torch::dtype(torch::kBool).device(s0.device()));
      auto *p_ptr = tree.data_ptr<bool>();
      fss::gpu::GrottoDcfPreprocessGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, p_ptr, grotto, CurrentStream());
      auto err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "grotto preprocess launch failed: ", cudaGetErrorString(err));
      return tree;
    } else {
      TORCH_CHECK(false, "GPU preprocess supports in_bits <= 24, got ", kInBits);
    }
  }

  // CPU path.
  typename GrottoInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws_ptr[i];

  auto pre = grotto.Preprocess(b, key);
  auto tree = torch::zeros({2 * n - 1}, torch::kBool);
  auto *tree_ptr = tree.data_ptr<bool>();
  const bool *src = pre.p.get();
  for (int64_t i = 0; i < 2 * n - 1; ++i) tree_ptr[i] = src[i];
  return tree;
}

// ---------------------------------------------------------------------------
// grotto_eval
// ---------------------------------------------------------------------------

// CPU only: the parity tree lives in host memory (the device buffer of a GPU
// preprocess is consumed by on-GPU point evaluation, not through this path).
// The party bit matters at preprocess time (it picks the tree); here it only
// tags the view, since Eval() itself reads no party state.
static bool grotto_eval(int64_t party, torch::Tensor tree, int64_t x_lo, int64_t x_hi) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(tree.is_cpu(), "tree must be on CPU for eval");
  TORCH_CHECK(tree.dtype() == torch::kBool, "tree must be bool");

  int64_t n = 1LL << kInBits;
  TORCH_CHECK(tree.dim() == 1 && tree.numel() == 2 * n - 1, "tree must have shape (2*2^in_bits-1,)");

  InType x = BuildInput<InType>(x_lo, x_hi);

  typename GrottoInst::ParityTree pt{tree.data_ptr<bool>(), party == 1};
  return GrottoInst::Eval(pt, x);
}

// ---------------------------------------------------------------------------
// grotto_eval_all
// ---------------------------------------------------------------------------

static torch::Tensor grotto_eval_all(int64_t party, torch::Tensor s0, torch::Tensor cws) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}), "s0 must have shape (4,)");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits + 1, 8}), "cws must have shape (kInBits+1, 8)");
  TORCH_CHECK(s0.device() == cws.device(), "s0 and cws must be on the same device");

  EnsureNonce();
  bool b = (party == 1);

  int64_t n = 1LL << kInBits;
  auto ys = torch::zeros({n}, torch::dtype(torch::kBool).device(s0.device()));

  GrottoInst grotto{PrgType(g_nonce0, g_nonce1)};
  auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
  int4 s0_val = FetchInt4(s0_ptr, s0.is_cuda());
  auto *cws_ptr = reinterpret_cast<const typename GrottoInst::Cw *>(cws.data_ptr<int32_t>());
  auto *ys_ptr = ys.data_ptr<bool>();

  if (s0.is_cuda()) {
    if constexpr (kInBits <= 24) {
      constexpr int kZ = EvalAllZ(kInBits), kB1 = EvalAllB1(kInBits), kBS = EvalAllBS(kInBits);
      fss::gpu::GrottoDcfEvalAllGpu<kZ, kB1, kBS>(b, s0_val, cws_ptr, ys_ptr, grotto, CurrentStream());
      auto err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "grotto eval_all launch failed: ", cudaGetErrorString(err));
    } else {
      TORCH_CHECK(false, "GPU EvalAll supports in_bits <= 24, got ", kInBits);
    }
    return ys;
  }

  // CPU path.
  typename GrottoInst::Key key;
  key.s0 = s0_val;
  for (int i = 0; i <= kInBits; ++i) key.cws[i] = cws_ptr[i];
  grotto.EvalAll(b, key, cuda::std::span<bool>(ys_ptr, n));

  return ys;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &grotto_gen, "Grotto DCF key generation");
  m.def("preprocess", &grotto_preprocess, "Grotto DCF key preprocessing");
  m.def("eval", &grotto_eval, "Grotto DCF single-point evaluation on a preprocessed tree");
  m.def("eval_all", &grotto_eval_all, "Grotto DCF full-domain evaluation");
}
