// SPDX-License-Identifier: Apache-2.0
// VDMPF binding implementation. Included by JIT-generated .cu files after type
// aliases (kInBits, InType, GroupType, PrgType, XorHashType, HashType, PrpType,
// VdmpfInst) are defined.
//
// Gen runs on the host. With a device-callable PRP (prp='chacha_feistel')
// batch_eval also accepts CUDA tensors and dispatches to the GPU Merkle-fold
// pipeline; the OpenSSL-backed Aes128Feistel keeps it CPU-only. The CPU path
// keeps the paper-faithful sequential proof chains, so the two paths produce
// bit-identical output shares but different proofs: both parties must
// evaluate on the same device. Alphas and xs arrive as int64 tensors and are
// widened to InType, so this binding supports in_bits <= 63. The inner VDPF
// key carries cws[bucket_bits] (no extra level, unlike DPF), plus cs and ocw
// per bucket.

#include "binding_common.cuh"
#include <fss/point_eval_gpu.cuh>
#include <torch/extension.h>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

constexpr int kM = VdmpfInst::m;

// The GPU pipeline passes the PRP to kernels by value, which only the
// stateless ChaChaFeistel backend survives.
constexpr bool kGpuPrp = std::is_trivially_copyable_v<PrpType>;

// Build a verifiable-scheme hash instance: Blake3 copies a 2x int4 IV by
// value, Sha256 takes a single int4 key.
template <typename Hash>
Hash MakeHash() {
  if constexpr (std::is_constructible_v<Hash, cuda::std::span<const int4, 2>>) {
    return Hash(cuda::std::span<const int4, 2>(kHashIv, 2));
  } else {
    return Hash(kHashIv[0]);
  }
}

void CheckCpu(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cpu(), name, " must be on CPU for Gen");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void CheckContiguous(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void CheckInt32(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32");
}

// Rebuild one party's key from the flattened tensors written by vdmpf_gen.
typename VdmpfInst::Key RebuildKey(const torch::Tensor &cws, const torch::Tensor &cs, const torch::Tensor &ocw,
    const torch::Tensor &s0, const torch::Tensor &sigma, const torch::Tensor &meta) {
  typename VdmpfInst::Key key;
  key.sigma = reinterpret_cast<const int4 *>(sigma.data_ptr<int32_t>())[0];
  const auto *meta_ptr = meta.data_ptr<int32_t>();
  key.m_rt = meta_ptr[0];
  key.b_size_rt = meta_ptr[1];
  const auto *cws_ptr = reinterpret_cast<const typename VdmpfInst::InnerVdpf::Cw *>(cws.data_ptr<int32_t>());
  const auto *cs_ptr = reinterpret_cast<const int4 *>(cs.data_ptr<int32_t>());
  const auto *ocw_ptr = reinterpret_cast<const int4 *>(ocw.data_ptr<int32_t>());
  const auto *s0_ptr = reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>());
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j < kBucketBits; ++j) key.bks[i].cws[j] = cws_ptr[i * kBucketBits + j];
    key.bks[i].cs = {cs_ptr[4 * i], cs_ptr[4 * i + 1], cs_ptr[4 * i + 2], cs_ptr[4 * i + 3]};
    key.bks[i].ocw = ocw_ptr[i];
    key.bks[i].s0 = s0_ptr[i];
  }
  return key;
}

// Runtime key parameters {m_rt, b_size_rt}, fetched to the host from either
// memory space.
std::pair<int, int> FetchMeta(const torch::Tensor &meta) {
  const auto *meta_ptr = meta.data_ptr<int32_t>();
  int v[2];
  if (meta.is_cuda()) {
    CopyDeviceToHost(v, meta_ptr, sizeof(v), "meta");
  } else {
    v[0] = meta_ptr[0];
    v[1] = meta_ptr[1];
  }
  return {v[0], v[1]};
}

// Widen int64 evaluation points to the scheme's InType on the device, so the
// points never round-trip through the host.
__global__ void WidenXsKernel(const int64_t *xs, InType *out, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = BuildInput<InType>(xs[i], 0);
}

torch::Tensor WidenXs(const torch::Tensor &xs, int64_t n) {
  constexpr auto kWideDtype = sizeof(InType) == 4 ? torch::kInt32 : torch::kInt64;
  auto wide = torch::empty({n}, torch::dtype(kWideDtype).device(xs.device()));
  int64_t blocks = (n + 255) / 256;
  WidenXsKernel<<<static_cast<int>(blocks), 256, 0, CurrentStream()>>>(
      xs.data_ptr<int64_t>(), reinterpret_cast<InType *>(wide.data_ptr()), n);
  return wide;
}

// Shared checks of the eval entry point: shapes, dtypes, one device for every
// tensor, and the PRP's device-callability when that device is CUDA.
void CheckEvalTensors(const torch::Tensor &cws, const torch::Tensor &cs, const torch::Tensor &ocw,
    const torch::Tensor &s0, const torch::Tensor &sigma, const torch::Tensor &meta, const torch::Tensor &xs) {
  TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
  TORCH_CHECK(cs.dtype() == torch::kInt32, "cs must be int32");
  TORCH_CHECK(ocw.dtype() == torch::kInt32, "ocw must be int32");
  TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
  TORCH_CHECK(sigma.dtype() == torch::kInt32, "sigma must be int32");
  TORCH_CHECK(meta.dtype() == torch::kInt32, "meta must be int32");
  TORCH_CHECK(xs.dtype() == torch::kInt64, "xs must be int64");
  TORCH_CHECK(
      cws.sizes() == torch::IntArrayRef({kM, kBucketBits, 8}), "cws must have shape (", kM, ", ", kBucketBits, ", 8)");
  TORCH_CHECK(cs.sizes() == torch::IntArrayRef({kM, 4, 4}), "cs must have shape (", kM, ", 4, 4)");
  TORCH_CHECK(ocw.sizes() == torch::IntArrayRef({kM, 4}), "ocw must have shape (", kM, ", 4)");
  TORCH_CHECK(s0.sizes() == torch::IntArrayRef({kM, 4}), "s0 must have shape (", kM, ", 4)");
  TORCH_CHECK(sigma.sizes() == torch::IntArrayRef({4}), "sigma must have shape (4,)");
  TORCH_CHECK(meta.sizes() == torch::IntArrayRef({2}), "meta must have shape (2,)");
  TORCH_CHECK(xs.dim() == 1, "xs must be 1-dimensional");
  for (const torch::Tensor *t : {&cws, &cs, &ocw, &s0, &sigma, &meta, &xs}) {
    CheckContiguous(*t,
        t == &cws         ? "cws"
            : t == &cs    ? "cs"
            : t == &ocw   ? "ocw"
            : t == &s0    ? "s0"
            : t == &sigma ? "sigma"
            : t == &meta  ? "meta"
                          : "xs");
    TORCH_CHECK(t->device() == cws.device(), "all tensors must be on the same device");
  }
  CheckCudaPrg(cws);
  CheckCudaHashes(cws);
  if (cws.is_cuda() && !kGpuPrp) {
    TORCH_CHECK(false, "CUDA tensors require a device-callable prp (prp='chacha_feistel')");
  }
  CheckQueryDomain(xs);
}

}  // namespace

// ---------------------------------------------------------------------------
// vdmpf_gen
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> vdmpf_gen(
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

  CheckQueryDomain(alphas);
  EnsureNonce();

  // Local copies: Gen perturbs sigma and the seeds on each retry, and the keys
  // carry the effective values, written back below.
  int4 sigma_local = reinterpret_cast<int4 *>(sigma.data_ptr<int32_t>())[0];
  cuda::std::array<cuda::std::array<int4, 2>, kM> seeds;
  auto *s0s_ptr = reinterpret_cast<int4 *>(s0s.data_ptr<int32_t>());
  for (int i = 0; i < kM; ++i) seeds[i] = {s0s_ptr[2 * i], s0s_ptr[2 * i + 1]};

  std::vector<InType> as(t);
  const auto *alphas_ptr = alphas.data_ptr<int64_t>();
  for (int64_t j = 0; j < t; ++j) as[j] = BuildInput<InType>(alphas_ptr[j], 0);

  cuda::std::span<const int4> b_bufs(reinterpret_cast<int4 *>(betas.data_ptr<int32_t>()), t);

  // Outputs: shared correction words, correction seeds, and output correction
  // words (one inner VDPF key per bucket), both parties' seeds, the effective
  // sigma, and the runtime parameters.
  auto cws_tensor = torch::zeros({kM, kBucketBits, 8}, torch::kInt32);
  auto cs_tensor = torch::zeros({kM, 4, 4}, torch::kInt32);
  auto ocw_tensor = torch::zeros({kM, 4}, torch::kInt32);
  auto s0s_tensor = torch::zeros({2, kM, 4}, torch::kInt32);
  auto sigma_tensor = torch::zeros({4}, torch::kInt32);
  auto meta = torch::zeros({2}, torch::kInt32);

  VdmpfInst vdmpf{MakePrg(), MakeHash<XorHashType>(), MakeHash<HashType>(), PrpType{}};
  typename VdmpfInst::Key k0, k1;
  TORCH_CHECK(vdmpf.Gen(k0, k1, sigma_local, cuda::std::span<cuda::std::array<int4, 2>, kM>(seeds),
                  cuda::std::span<const InType>(as), b_bufs, static_cast<int>(t), 1000, retry) == 0,
      "cuckoo hashing failed");

  auto *cws_ptr = reinterpret_cast<typename VdmpfInst::InnerVdpf::Cw *>(cws_tensor.data_ptr<int32_t>());
  auto *cs_out = reinterpret_cast<int4 *>(cs_tensor.data_ptr<int32_t>());
  auto *ocw_out = reinterpret_cast<int4 *>(ocw_tensor.data_ptr<int32_t>());
  auto *s0s_out = reinterpret_cast<int4 *>(s0s_tensor.data_ptr<int32_t>());
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j < kBucketBits; ++j) cws_ptr[i * kBucketBits + j] = k0.bks[i].cws[j];
    cs_out[4 * i] = k0.bks[i].cs[0];
    cs_out[4 * i + 1] = k0.bks[i].cs[1];
    cs_out[4 * i + 2] = k0.bks[i].cs[2];
    cs_out[4 * i + 3] = k0.bks[i].cs[3];
    ocw_out[i] = k0.bks[i].ocw;
    s0s_out[i] = k0.bks[i].s0;
    s0s_out[kM + i] = k1.bks[i].s0;
  }
  reinterpret_cast<int4 *>(sigma_tensor.data_ptr<int32_t>())[0] = sigma_local;
  auto *meta_ptr = meta.data_ptr<int32_t>();
  meta_ptr[0] = k0.m_rt;
  meta_ptr[1] = k0.b_size_rt;

  return {cws_tensor, cs_tensor, ocw_tensor, s0s_tensor, sigma_tensor, meta};
}

// ---------------------------------------------------------------------------
// vdmpf_batch_eval
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> vdmpf_batch_eval_impl(int64_t party, torch::Tensor cws,
    torch::Tensor cs, torch::Tensor ocw, torch::Tensor s0, torch::Tensor sigma, torch::Tensor meta, torch::Tensor xs,
    bool merkle) {
  TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
  TORCH_CHECK(kInBits <= 63, "this binding supports in_bits <= 63, got ", kInBits);
  c10::cuda::OptionalCUDAGuard device_guard;
  if (s0.is_cuda()) device_guard.set_device(s0.device());
  xs = xs.contiguous();
  CheckEvalTensors(cws, cs, ocw, s0, sigma, meta, xs);

  EnsureNonce();
  CheckCudaPrg(s0);
  bool b = (party == 1);
  const int64_t n = xs.size(0);

  if (s0.is_cuda()) {
    if constexpr (kGpuPrp) {
      auto [m_rt, b_rt] = FetchMeta(meta);
      int4 sigma_val = FetchInt4(reinterpret_cast<const int4 *>(sigma.data_ptr<int32_t>()), true);
      auto xs_wide = n > 0 ? WidenXs(xs, n) : torch::Tensor();
      auto ys = torch::zeros({n, 4}, torch::dtype(torch::kInt32).device(s0.device()));
      auto pi = torch::zeros({4, 4}, torch::dtype(torch::kInt32).device(s0.device()));

      // One-time repack of the m_rt active bucket keys into the level-major
      // layout the batch kernel consumes; cs and ocws need no repack.
      auto dev_i32 = torch::dtype(torch::kInt32).device(s0.device());
      auto cw_s = torch::empty({static_cast<int64_t>(kBucketBits) * m_rt, 4}, dev_i32);
      auto extra = torch::empty(
          {static_cast<int64_t>(m_rt) * static_cast<int64_t>(sizeof(fss::gpu::ExtBits<kBucketBits>) / sizeof(int32_t))},
          dev_i32);
      if (n > 0)
        fss::gpu::VdpfRepackGpu<kBucketBits, GroupType, PrgType, XorHashType, HashType, uint>(
            reinterpret_cast<const typename VdmpfInst::InnerVdpf::Cw *>(cws.data_ptr<int32_t>()), m_rt,
            reinterpret_cast<int4 *>(cw_s.data_ptr<int32_t>()),
            reinterpret_cast<fss::gpu::ExtBits<kBucketBits> *>(extra.data_ptr<int32_t>()), CurrentStream());

      VdmpfInst vdmpf{MakePrg(), MakeHash<XorHashType>(), MakeHash<HashType>(), PrpType{}};
      {
        fss::gpu::VdmpfBatchEvalMerkleGpu(b, n > 0 ? reinterpret_cast<const InType *>(xs_wide.data_ptr()) : nullptr,
            reinterpret_cast<int4 *>(ys.data_ptr<int32_t>()),
            reinterpret_cast<cuda::std::array<int4, 4> *>(pi.data_ptr<int32_t>()), static_cast<int>(n), sigma_val, m_rt,
            b_rt, reinterpret_cast<const int4 *>(s0.data_ptr<int32_t>()),
            reinterpret_cast<const int4 *>(cw_s.data_ptr<int32_t>()),
            reinterpret_cast<const fss::gpu::ExtBits<kBucketBits> *>(extra.data_ptr<int32_t>()),
            reinterpret_cast<const cuda::std::array<int4, 4> *>(cs.data_ptr<int32_t>()),
            reinterpret_cast<const int4 *>(ocw.data_ptr<int32_t>()), vdmpf, CurrentStream());
      }
      auto err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "vdmpf batch_eval launch failed: ", cudaGetErrorString(err));
      return {ys, pi};
    }
  }

  typename VdmpfInst::Key key = RebuildKey(cws, cs, ocw, s0, sigma, meta);

  std::vector<InType> xs_local(n);
  const auto *xs_ptr = xs.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) xs_local[i] = BuildInput<InType>(xs_ptr[i], 0);

  auto ys = torch::zeros({n, 4}, torch::kInt32);
  auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

  cuda::std::array<int4, 4> pi;
  VdmpfInst vdmpf{MakePrg(), MakeHash<XorHashType>(), MakeHash<HashType>(), PrpType{}};
  if (merkle) {
    vdmpf.BatchEvalMerkle(b, key, cuda::std::span<const InType>(xs_local), cuda::std::span<int4>(ys_ptr, n), pi);
  } else {
    vdmpf.BatchEval(b, key, cuda::std::span<const InType>(xs_local), cuda::std::span<int4>(ys_ptr, n), pi);
  }

  auto pi_tensor = torch::zeros({4, 4}, torch::kInt32);
  auto *pi_ptr = reinterpret_cast<int4 *>(pi_tensor.data_ptr<int32_t>());
  for (int i = 0; i < 4; ++i) pi_ptr[i] = pi[i];

  return {ys, pi_tensor};
}

static std::tuple<torch::Tensor, torch::Tensor> vdmpf_batch_eval(int64_t party, torch::Tensor cws, torch::Tensor cs,
    torch::Tensor ocw, torch::Tensor s0, torch::Tensor sigma, torch::Tensor meta, torch::Tensor xs) {
  return vdmpf_batch_eval_impl(party, cws, cs, ocw, s0, sigma, meta, xs, false);
}

static std::tuple<torch::Tensor, torch::Tensor> vdmpf_batch_eval_merkle(int64_t party, torch::Tensor cws,
    torch::Tensor cs, torch::Tensor ocw, torch::Tensor s0, torch::Tensor sigma, torch::Tensor meta, torch::Tensor xs) {
  return vdmpf_batch_eval_impl(party, cws, cs, ocw, s0, sigma, meta, xs, true);
}

// ---------------------------------------------------------------------------
// vdmpf_verify
// ---------------------------------------------------------------------------

static bool vdmpf_verify(torch::Tensor pi0, torch::Tensor pi1) {
  CheckInt32(pi0, "pi0");
  CheckInt32(pi1, "pi1");
  CheckContiguous(pi0, "pi0");
  CheckContiguous(pi1, "pi1");
  TORCH_CHECK(pi0.sizes() == torch::IntArrayRef({4, 4}), "pi0 must have shape (4, 4)");
  TORCH_CHECK(pi1.sizes() == torch::IntArrayRef({4, 4}), "pi1 must have shape (4, 4)");
  TORCH_CHECK(pi0.device() == pi1.device(), "pi0 and pi1 must be on the same device");

  c10::cuda::OptionalCUDAGuard device_guard;
  if (pi0.is_cuda()) device_guard.set_device(pi0.device());

  // Verify() compares 64B proofs on the host, so CUDA proofs are fetched.
  cuda::std::array<int4, 4> h0 = FetchCs(reinterpret_cast<const int4 *>(pi0.data_ptr<int32_t>()), pi0.is_cuda());
  cuda::std::array<int4, 4> h1 = FetchCs(reinterpret_cast<const int4 *>(pi1.data_ptr<int32_t>()), pi1.is_cuda());
  return VdmpfInst::Verify(cuda::std::span<const int4, 4>(h0.data(), 4), cuda::std::span<const int4, 4>(h1.data(), 4));
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen", &vdmpf_gen, "VDMPF key generation");
  m.def("batch_eval", &vdmpf_batch_eval, "VDMPF batch evaluation");
  m.def("batch_eval_merkle", &vdmpf_batch_eval_merkle, "VDMPF Merkle batch evaluation");
  m.def("verify", &vdmpf_verify, "VDMPF proof verification");
}
