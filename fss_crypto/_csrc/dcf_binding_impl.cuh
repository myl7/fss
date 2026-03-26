// SPDX-License-Identifier: Apache-2.0
// DCF binding implementation. Included by JIT-generated .cu files after type
// aliases (kInBits, InType, GroupType, PrgType, kPred, DcfInst) are defined.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <random>

// ---------------------------------------------------------------------------
// Nonce management
// ---------------------------------------------------------------------------

namespace {

// Host nonce: 2 ints, randomly generated once.
int g_host_nonce[2] = {0, 0};
int *g_device_nonce = nullptr;
bool g_host_nonce_initialized = false;
bool g_device_nonce_initialized = false;

void EnsureHostNonce() {
    if (g_host_nonce_initialized) return;
    std::random_device rd;
    g_host_nonce[0] = static_cast<int>(rd());
    g_host_nonce[1] = static_cast<int>(rd());
    g_host_nonce_initialized = true;
}

void EnsureNonce() {
    EnsureHostNonce();
    if (g_device_nonce_initialized) return;

    auto err = cudaMalloc(&g_device_nonce, 2 * sizeof(int));
    TORCH_CHECK(err == cudaSuccess,
                "cudaMalloc for nonce failed: ", cudaGetErrorString(err));

    err = cudaMemcpy(g_device_nonce, g_host_nonce, 2 * sizeof(int),
                     cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess,
                "cudaMemcpy for nonce failed: ", cudaGetErrorString(err));

    g_device_nonce_initialized = true;
}

}  // namespace

// ---------------------------------------------------------------------------
// GPU eval kernel
// ---------------------------------------------------------------------------

__global__ void dcf_eval_kernel(bool party, int4 s0,
                                const typename DcfInst::Cw *cws,
                                int4 *out, int *dev_nonce,
                                int64_t x_lo, int64_t x_hi) {
    DcfInst dcf{PrgType(dev_nonce)};

    // Reconstruct x from lo/hi.
    InType x;
    if constexpr (sizeof(InType) <= 8) {
        x = static_cast<InType>(static_cast<uint64_t>(x_lo));
    } else {
        x = static_cast<InType>(static_cast<uint64_t>(x_lo)) |
            (static_cast<InType>(static_cast<uint64_t>(x_hi)) << 64);
    }

    int4 result = dcf.Eval(party, s0, cws, x);
    out[0] = result;
}

// ---------------------------------------------------------------------------
// dcf_gen
// ---------------------------------------------------------------------------

static torch::Tensor dcf_gen(torch::Tensor s0s, int64_t alpha_lo,
                             int64_t alpha_hi, torch::Tensor beta) {
    TORCH_CHECK(s0s.is_cpu(), "s0s must be on CPU for Gen");
    TORCH_CHECK(beta.is_cpu(), "beta must be on CPU for Gen");
    TORCH_CHECK(s0s.dtype() == torch::kInt32, "s0s must be int32");
    TORCH_CHECK(beta.dtype() == torch::kInt32, "beta must be int32");
    TORCH_CHECK(s0s.sizes() == torch::IntArrayRef({2, 4}),
                "s0s must have shape (2, 4)");
    TORCH_CHECK(beta.sizes() == torch::IntArrayRef({4}),
                "beta must have shape (4,)");

    EnsureHostNonce();

    // Build alpha from lo/hi.
    InType alpha;
    if constexpr (sizeof(InType) <= 8) {
        alpha = static_cast<InType>(static_cast<uint64_t>(alpha_lo));
    } else {
        alpha = static_cast<InType>(static_cast<uint64_t>(alpha_lo)) |
                (static_cast<InType>(static_cast<uint64_t>(alpha_hi)) << 64);
    }

    // Interpret s0s as int4[2] and beta as int4.
    auto *s0s_ptr = reinterpret_cast<int4 *>(s0s.data_ptr<int32_t>());
    auto *beta_ptr = reinterpret_cast<int4 *>(beta.data_ptr<int32_t>());
    int4 b_buf = beta_ptr[0];

    // Allocate output: (kInBits+1) correction words, each 32B = 8 int32.
    auto cws_tensor = torch::zeros({kInBits + 1, 8}, torch::kInt32);
    auto *cws = reinterpret_cast<typename DcfInst::Cw *>(
        cws_tensor.data_ptr<int32_t>());

    DcfInst dcf{PrgType(g_host_nonce)};
    dcf.Gen(cws, s0s_ptr, alpha, b_buf);

    return cws_tensor;
}

// ---------------------------------------------------------------------------
// dcf_eval
// ---------------------------------------------------------------------------

static torch::Tensor dcf_eval(int64_t party, torch::Tensor s0,
                               torch::Tensor cws, int64_t x_lo,
                               int64_t x_hi) {
    TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
    TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
    TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
    TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}),
                "s0 must have shape (4,)");
    TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits + 1, 8}),
                "cws must have shape (kInBits+1, 8)");
    TORCH_CHECK(s0.device() == cws.device(),
                "s0 and cws must be on the same device");

    EnsureHostNonce();
    bool b = (party == 1);

    if (s0.is_cuda()) {
        EnsureNonce();
        auto out = torch::zeros({4}, torch::dtype(torch::kInt32)
                                         .device(s0.device()));

        auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
        int4 s0_val = *s0_ptr;

        dcf_eval_kernel<<<1, 1>>>(
            b, s0_val,
            reinterpret_cast<const typename DcfInst::Cw *>(
                cws.data_ptr<int32_t>()),
            reinterpret_cast<int4 *>(out.data_ptr<int32_t>()),
            g_device_nonce, x_lo, x_hi);

        auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "dcf_eval_kernel launch failed: ",
                    cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        TORCH_CHECK(err == cudaSuccess,
                    "dcf_eval_kernel sync failed: ",
                    cudaGetErrorString(err));

        return out;
    }

    // CPU path.
    auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
    int4 s0_val = s0_ptr[0];
    auto *cws_ptr = reinterpret_cast<const typename DcfInst::Cw *>(
        cws.data_ptr<int32_t>());

    InType x;
    if constexpr (sizeof(InType) <= 8) {
        x = static_cast<InType>(static_cast<uint64_t>(x_lo));
    } else {
        x = static_cast<InType>(static_cast<uint64_t>(x_lo)) |
            (static_cast<InType>(static_cast<uint64_t>(x_hi)) << 64);
    }

    DcfInst dcf{PrgType(g_host_nonce)};
    int4 result = dcf.Eval(b, s0_val, cws_ptr, x);

    auto out = torch::zeros({4}, torch::kInt32);
    auto *out_ptr = reinterpret_cast<int4 *>(out.data_ptr<int32_t>());
    out_ptr[0] = result;
    return out;
}

// ---------------------------------------------------------------------------
// dcf_eval_all
// ---------------------------------------------------------------------------

static torch::Tensor dcf_eval_all(int64_t party, torch::Tensor s0,
                                   torch::Tensor cws) {
    TORCH_CHECK(party == 0 || party == 1, "party must be 0 or 1");
    TORCH_CHECK(s0.is_cpu(), "s0 must be on CPU for EvalAll");
    TORCH_CHECK(cws.is_cpu(), "cws must be on CPU for EvalAll");
    TORCH_CHECK(s0.dtype() == torch::kInt32, "s0 must be int32");
    TORCH_CHECK(cws.dtype() == torch::kInt32, "cws must be int32");
    TORCH_CHECK(s0.sizes() == torch::IntArrayRef({4}),
                "s0 must have shape (4,)");
    TORCH_CHECK(cws.sizes() == torch::IntArrayRef({kInBits + 1, 8}),
                "cws must have shape (kInBits+1, 8)");

    EnsureHostNonce();
    bool b = (party == 1);

    int64_t n = 1LL << kInBits;
    auto ys = torch::zeros({n, 4}, torch::kInt32);

    auto *s0_ptr = reinterpret_cast<int4 *>(s0.data_ptr<int32_t>());
    int4 s0_val = s0_ptr[0];
    auto *cws_ptr = reinterpret_cast<const typename DcfInst::Cw *>(
        cws.data_ptr<int32_t>());
    auto *ys_ptr = reinterpret_cast<int4 *>(ys.data_ptr<int32_t>());

    DcfInst dcf{PrgType(g_host_nonce)};
    dcf.EvalAll(b, s0_val, cws_ptr, ys_ptr);

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
