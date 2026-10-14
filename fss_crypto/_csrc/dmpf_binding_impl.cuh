// SPDX-License-Identifier: Apache-2.0
// DMPF binding implementation. Included by JIT-generated .cu files after type
// aliases (kInBits, InType, GroupType, PrgType, PrpType, DmpfInst) are defined.
//
// DMPF is a host-only scheme (the Cuckoo-hashing PRP has no GPU path), so
// every entry point requires CPU tensors. Alphas and xs arrive as int64
// tensors and are widened to InType, so this binding supports in_bits <= 63.

#include "binding_common.cuh"
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace {

constexpr int kM = DmpfInst::m;

void CheckCpu(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cpu(), name, " must be on CPU (DMPF is host-only)");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void CheckInt32(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32");
}

// Rebuild one party's key from the flattened tensors written by dmpf_gen.
typename DmpfInst::Key RebuildKey(const torch::Tensor &cws, const torch::Tensor &s0, const torch::Tensor &sigma,
    const torch::Tensor &meta) {
  typename DmpfInst::Key key;
  key.sigma = reinterpret_cast<const int4 *>(sigma.data_ptr<int32_t>())[0];
  const auto *meta_ptr = meta.data_ptr<int32_t>();
  key.m_rt = meta_ptr[0];
  key.b_size_rt = meta_ptr[1];
  const auto *cws_ptr = reinterpret_cast<const typename DmpfInst::InnerDpf::Cw *>(cws.data_ptr<int32_t>());
  const auto *s0_ptr = reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>());
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j <= kBucketBits; ++j) key.bks[i].cws[j] = cws_ptr[i * (kBucketBits + 1) + j];
    key.bks[i].s0 = s0_ptr[i];
  }
  return key;
}

}  // namespace

// ---------------------------------------------------------------------------
// dmpf_gen
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dmpf_gen(
    torch::Tensor s0s, torch::Tensor sigma, torch::Tensor alphas, torch::Tensor betas, int64_t t, bool retry) {
  TORCH_CHECK(kInBits <= 63, "this binding supports in_bits <= 63, got ", kInBits);
  CheckCpu(s0s, "s0s");
  CheckCpu(sigma, "sigma");
  CheckCpu(alphas, "alphas");
  CheckCpu(betas, "betas");
  CheckInt32(s0s, "s0s");
  CheckInt32(sigma, "sigma");
  CheckInt32(betas, "betas");
  TORCH_CHECK(alphas.dtype() == torch::kInt64, "alphas must be int64");
  TORCH_CHECK(s0s.sizes() == torch::IntArrayRef({kM, 2, 4}), "s0s must have shape (", kM, ", 2, 4)");
  TORCH_CHECK(sigma.sizes() == torch::IntArrayRef({4}), "sigma must have shape (4,)");
  TORCH_CHECK(alphas.dim() == 1 && alphas.size(0) == t, "alphas must have shape (", t, ")");
  TORCH_CHECK(betas.sizes() == torch::IntArrayRef({t, 4}), "betas must have shape (", t, ", 4)");
  TORCH_CHECK(t >= 30 && t <= kMaxPoints, "t must be in [30, ", kMaxPoints, "], got ", t);

  EnsureNonce();

  // Local copies: Gen perturbs sigma and the seeds on each Cuckoo-hashing
  // retry, and the keys carry the effective values, written back below.
  int4 sigma_local = reinterpret_cast<int4 *>(sigma.data_ptr<int32_t>())[0];
  cuda::std::array<cuda::std::array<int4, 2>, kM> seeds;
  auto *s0s_ptr = reinterpret_cast<int4 *>(s0s.data_ptr<int32_t>());
  for (int i = 0; i < kM; ++i) seeds[i] = {s0s_ptr[2 * i], s0s_ptr[2 * i + 1]};

  std::vector<InType> as(t);
  const auto *alphas_ptr = alphas.data_ptr<int64_t>();
  for (int64_t j = 0; j < t; ++j) as[j] = BuildInput<InType>(alphas_ptr[j], 0);

  cuda::std::span<const int4> b_bufs(reinterpret_cast<int4 *>(betas.data_ptr<int32_t>()), t);

  // Outputs: shared correction words (m buckets, one inner DPF key each),
  // both parties' seeds, the effective sigma, and the runtime parameters.
  auto cws_tensor = torch::zeros({kM, kBucketBits + 1, 8}, torch::kInt32);
  auto s0s_tensor = torch::zeros({2, kM, 4}, torch::kInt32);
  auto sigma_tensor = torch::zeros({4}, torch::kInt32);
  auto meta = torch::zeros({2}, torch::kInt32);

  DmpfInst dmpf{PrgType(g_nonce0, g_nonce1), PrpType{}};
  typename DmpfInst::Key k0, k1;
  TORCH_CHECK(dmpf.Gen(k0, k1, sigma_local, cuda::std::span<cuda::std::array<int4, 2>, kM>(seeds),
                  cuda::std::span<const InType>(as), b_bufs, static_cast<int>(t), 1000, retry) == 0,
      "cuckoo hashing failed");

  auto *cws_ptr = reinterpret_cast<typename DmpfInst::InnerDpf::Cw *>(cws_tensor.data_ptr<int32_t>());
  auto *s0s_out = reinterpret_cast<int4 *>(s0s_tensor.data_ptr<int32_t>());
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j <= kBucketBits; ++j) cws_ptr[i * (kBucketBits + 1) + j] = k0.bks[i].cws[j];
    s0s_out[i] = k0.bks[i].s0;
    s0s_out[kM + i] = k1.bks[i].s0;
  }
  reinterpret_cast<int4 *>(sigma_tensor.data_ptr<int32_t>())[0] = sigma_local;
  auto *meta_ptr = meta.data_ptr<int32_t>();
  meta_ptr[0] = k0.m_rt;
  meta_ptr[1] = k0.b_size_rt;

  return {cws_tensor, s0s_tensor, sigma_tensor, meta};
}

// ---------------------------------------------------------------------------
// dmpf_batch_eval
// ---------------------------------------------------------------------------

static torch::Tensor dmpf_batch_eval(
    int64_t party, torch::Tensor cws, torch::Tensor s0, torch::Tensor sigma, torch::Tensor meta, torch::Tensor xs) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(kInBits <= 63, "this binding supports in_bits <= 63, got ", kInBits);
  CheckCpu(cws, "cws");
  CheckCpu(s0, "s0");
  CheckCpu(sigma, "sigma");
  CheckCpu(meta, "meta");
  CheckCpu(xs, "xs");
  CheckInt32(cws, "cws");
  CheckInt32(s0, "s0");
  CheckInt32(sigma, "sigma");
  CheckInt32(meta, "meta");
  TORCH_CHECK(xs.dtype() == torch::kInt64, "xs must be int64");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kM, kBucketBits + 1, 8}), "cws must have shape (", kM, ", ",
      kBucketBits + 1, ", 8)");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({kM, 4}), "s0 must have shape (", kM, ", 4)");
  TORCH_CHECK(sigma.sizes() == torch::IntArrayRef({4}), "sigma must have shape (4,)");
  TORCH_CHECK(meta.sizes() == torch::IntArrayRef({2}), "meta must have shape (2,)");
  TORCH_CHECK(xs.dim() == 1, "xs must be 1-dimensional");

  EnsureNonce();
  bool b = (party == 1);

  typename DmpfInst::Key key = RebuildKey(cws, s0, sigma, meta);

  const int64_t n = xs.size(0);
  std::vector<InType> xs_local(n);
  const auto *xs_ptr = xs.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) xs_local[i] = BuildInput<InType>(xs_ptr[i], 0);

  auto ys = torch::zeros({n, 4}, torch::kInt32);
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

  DmpfInst dmpf{PrgType(g_nonce0, g_nonce1), PrpType{}};
  dmpf.BatchEval(b, key, cuda::std::span<const InType>(xs_local), cuda::std::span<int4>(ys_ptr, n));

  return ys;
}

// ---------------------------------------------------------------------------
// dmpf_eval_all
// ---------------------------------------------------------------------------

static torch::Tensor dmpf_eval_all(
    int64_t party, torch::Tensor cws, torch::Tensor s0, torch::Tensor sigma, torch::Tensor meta) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(kInBits <= 63, "this binding supports in_bits <= 63, got ", kInBits);
  CheckCpu(cws, "cws");
  CheckCpu(s0, "s0");
  CheckCpu(sigma, "sigma");
  CheckCpu(meta, "meta");
  CheckInt32(cws, "cws");
  CheckInt32(s0, "s0");
  CheckInt32(sigma, "sigma");
  CheckInt32(meta, "meta");
  TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kM, kBucketBits + 1, 8}), "cws must have shape (", kM, ", ",
      kBucketBits + 1, ", 8)");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({kM, 4}), "s0 must have shape (", kM, ", 4)");
  TORCH_CHECK(sigma.sizes() == torch::IntArrayRef({4}), "sigma must have shape (4,)");
  TORCH_CHECK(meta.sizes() == torch::IntArrayRef({2}), "meta must have shape (2,)");

  EnsureNonce();
  bool b = (party == 1);

  typename DmpfInst::Key key = RebuildKey(cws, s0, sigma, meta);

  int64_t n = 1LL << kInBits;
  auto ys = torch::zeros({n, 4}, torch::kInt32);
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

  DmpfInst dmpf{PrgType(g_nonce0, g_nonce1), PrpType{}};
  dmpf.EvalAll(b, key, cuda::std::span<int4>(ys_ptr, n));

  return ys;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &dmpf_gen, "DMPF key generation");
  m.def("batch_eval", &dmpf_batch_eval, "DMPF batch evaluation");
  m.def("eval_all", &dmpf_eval_all, "DMPF full-domain evaluation");
}
