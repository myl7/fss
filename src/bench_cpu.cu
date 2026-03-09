#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/vdpf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/grotto_dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prg/aes128_soft.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/sha256.cuh>
#include <fss/hash/blake3.cuh>
#include <memory>
#include <vector>

using BytesGroup = fss::group::Bytes;
using UintGroup = fss::group::Uint<uint64_t>;

// --- AES PRG helpers ---

template <int mul>
struct AesCtx {
    using Prg = fss::prg::Aes128Mmo<mul>;
    cuda::std::array<EVP_CIPHER_CTX *, mul> ctxs;
    Prg prg;

    AesCtx() : ctxs(MakeCtxs()), prg(ctxs) {}
    ~AesCtx() {
        Prg::FreeCtxs(ctxs);
    }

private:
    static cuda::std::array<EVP_CIPHER_CTX *, mul> MakeCtxs() {
        unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        unsigned char key2[16] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
        unsigned char key3[16] = {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
        if constexpr (mul == 1) {
            const unsigned char *keys[1] = {key0};
            return Prg::CreateCtxs(keys);
        } else if constexpr (mul == 2) {
            const unsigned char *keys[2] = {key0, key1};
            return Prg::CreateCtxs(keys);
        } else {
            const unsigned char *keys[4] = {key0, key1, key2, key3};
            return Prg::CreateCtxs(keys);
        }
    }
};

// --- ChaCha PRG helper ---

static int gNonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

// --- AesSoft PRG helper ---

static uint8_t gAesSoftKeys[4][16] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
    {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8},
    {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1},
};

// --- Hash helpers ---

static int4 gHashKey = {static_cast<int>(0xAAAAAAAAu), static_cast<int>(0xBBBBBBBBu),
    static_cast<int>(0xCCCCCCCCu), static_cast<int>(0xDDDDDDDDu)};

static int4 gBlake3Iv[2] = {
    {0x11111111, 0x22222222, 0x33333333, 0x44444444},
    {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)},
};

template <typename H>
H MakeHash() {
    if constexpr (std::is_same_v<H, fss::hash::Sha256>) {
        return H(gHashKey);
    } else {
        return H(cuda::std::span<const int4, 2>(gBlake3Iv, 2));
    }
}

// --- DPF benchmarks ---

template <int in_bits, typename Group, typename Prg>
static void BM_DpfGen(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, Prg, uint>;

    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename DpfType::Cw cws[in_bits + 1];

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<2>>) {
        AesCtx<2> ctx;
        DpfType dpf{ctx.prg};
        for (auto _ : state) {
            dpf.Gen(cws, seeds, alpha, beta);
            benchmark::DoNotOptimize(cws);
        }
    } else {
        Prg prg(gNonce);
        DpfType dpf{prg};
        for (auto _ : state) {
            dpf.Gen(cws, seeds, alpha, beta);
            benchmark::DoNotOptimize(cws);
        }
    }
}

template <int in_bits, typename Group, typename Prg>
static void BM_DpfEval(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, Prg, uint>;

    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename DpfType::Cw cws[in_bits + 1];
    uint x = 100;

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<2>>) {
        AesCtx<2> ctx;
        DpfType dpf{ctx.prg};
        dpf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            int4 y = dpf.Eval(false, seeds[0], cws, x);
            benchmark::DoNotOptimize(y);
        }
    } else if constexpr (std::is_same_v<Prg, fss::prg::Aes128Soft<2>>) {
        Prg prg(gAesSoftKeys);
        DpfType dpf{prg};
        dpf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            int4 y = dpf.Eval(false, seeds[0], cws, x);
            benchmark::DoNotOptimize(y);
        }
    } else {
        Prg prg(gNonce);
        DpfType dpf{prg};
        dpf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            int4 y = dpf.Eval(false, seeds[0], cws, x);
            benchmark::DoNotOptimize(y);
        }
    }
}

template <int in_bits, typename Group, typename Prg>
static void BM_DpfEvalAll(benchmark::State &state) {
    using DpfType = fss::Dpf<in_bits, Group, Prg, uint>;

    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename DpfType::Cw cws[in_bits + 1];

    constexpr size_t n = size_t{1} << in_bits;
    std::vector<int4> ys(n);

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<2>>) {
        AesCtx<2> ctx;
        DpfType dpf{ctx.prg};
        dpf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            dpf.EvalAll(false, seeds[0], cws, ys.data());
            benchmark::DoNotOptimize(ys.data());
        }
    } else {
        Prg prg(gNonce);
        DpfType dpf{prg};
        dpf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            dpf.EvalAll(false, seeds[0], cws, ys.data());
            benchmark::DoNotOptimize(ys.data());
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- DCF benchmarks ---

template <int in_bits, typename Group, typename Prg>
static void BM_DcfGen(benchmark::State &state) {
    using DcfType = fss::Dcf<in_bits, Group, Prg, uint>;

    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename DcfType::Cw cws[in_bits + 1];

    AesCtx<4> ctx;
    DcfType dcf{ctx.prg};
    for (auto _ : state) {
        dcf.Gen(cws, seeds, alpha, beta);
        benchmark::DoNotOptimize(cws);
    }
}

template <int in_bits, typename Group, typename Prg>
static void BM_DcfEval(benchmark::State &state) {
    using DcfType = fss::Dcf<in_bits, Group, Prg, uint>;

    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename DcfType::Cw cws[in_bits + 1];
    uint x = 100;

    AesCtx<4> ctx;
    DcfType dcf{ctx.prg};
    dcf.Gen(cws, seeds, alpha, beta);
    for (auto _ : state) {
        int4 y = dcf.Eval(false, seeds[0], cws, x);
        benchmark::DoNotOptimize(y);
    }
}

template <int in_bits, typename Group, typename Prg>
static void BM_DcfEvalAll(benchmark::State &state) {
    using DcfType = fss::Dcf<in_bits, Group, Prg, uint>;

    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename DcfType::Cw cws[in_bits + 1];

    constexpr size_t n = size_t{1} << in_bits;
    std::vector<int4> ys(n);

    AesCtx<4> ctx;
    DcfType dcf{ctx.prg};
    dcf.Gen(cws, seeds, alpha, beta);
    for (auto _ : state) {
        dcf.EvalAll(false, seeds[0], cws, ys.data());
        benchmark::DoNotOptimize(ys.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- VDPF benchmarks ---

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash>
static void BM_VdpfGen(benchmark::State &state) {
    using VdpfType = fss::Vdpf<in_bits, Group, Prg, XorHash, Hash, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename VdpfType::Cw cws[in_bits];
    cuda::std::array<int4, 4> cs;
    int4 ocw;

    auto xor_hash = MakeHash<XorHash>();
    auto hash_ = MakeHash<Hash>();

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<2>>) {
        AesCtx<2> ctx;
        VdpfType vdpf{ctx.prg, xor_hash, hash_};
        for (auto _ : state) {
            vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
            benchmark::DoNotOptimize(cws);
        }
    } else {
        Prg prg(gNonce);
        VdpfType vdpf{prg, xor_hash, hash_};
        for (auto _ : state) {
            vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
            benchmark::DoNotOptimize(cws);
        }
    }
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash>
static void BM_VdpfEval(benchmark::State &state) {
    using VdpfType = fss::Vdpf<in_bits, Group, Prg, XorHash, Hash, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename VdpfType::Cw cws[in_bits];
    cuda::std::array<int4, 4> cs;
    int4 ocw;
    uint x = 100;

    auto xor_hash = MakeHash<XorHash>();
    auto hash_ = MakeHash<Hash>();

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<2>>) {
        AesCtx<2> ctx;
        VdpfType vdpf{ctx.prg, xor_hash, hash_};
        vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
        for (auto _ : state) {
            int4 y;
            auto pi_tilde =
                vdpf.Eval(false, s0s[0], cuda::std::span<const typename VdpfType::Cw>(cws, in_bits),
                    cuda::std::span<const int4, 4>(cs), ocw, x, y);
            cuda::std::array<int4, 4> pi;
            vdpf.Prove({&pi_tilde, 1}, cuda::std::span<const int4, 4>(cs), pi);
            benchmark::DoNotOptimize(y);
            benchmark::DoNotOptimize(pi);
        }
    } else {
        Prg prg(gNonce);
        VdpfType vdpf{prg, xor_hash, hash_};
        vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
        for (auto _ : state) {
            int4 y;
            auto pi_tilde =
                vdpf.Eval(false, s0s[0], cuda::std::span<const typename VdpfType::Cw>(cws, in_bits),
                    cuda::std::span<const int4, 4>(cs), ocw, x, y);
            cuda::std::array<int4, 4> pi;
            vdpf.Prove({&pi_tilde, 1}, cuda::std::span<const int4, 4>(cs), pi);
            benchmark::DoNotOptimize(y);
            benchmark::DoNotOptimize(pi);
        }
    }
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash>
static void BM_VdpfProve(benchmark::State &state) {
    using VdpfType = fss::Vdpf<in_bits, Group, Prg, XorHash, Hash, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename VdpfType::Cw cws[in_bits];
    cuda::std::array<int4, 4> cs;
    int4 ocw;
    uint x = 100;

    auto xor_hash = MakeHash<XorHash>();
    auto hash_ = MakeHash<Hash>();
    Prg prg(gNonce);
    VdpfType vdpf{prg, xor_hash, hash_};
    vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);

    int4 y;
    auto pi_tilde =
        vdpf.Eval(false, s0s[0], cuda::std::span<const typename VdpfType::Cw>(cws, in_bits),
            cuda::std::span<const int4, 4>(cs), ocw, x, y);

    for (auto _ : state) {
        cuda::std::array<int4, 4> pi;
        vdpf.Prove({&pi_tilde, 1}, cuda::std::span<const int4, 4>(cs), pi);
        benchmark::DoNotOptimize(pi);
    }
}

template <int in_bits, typename Group, typename Prg, typename XorHash, typename Hash>
static void BM_VdpfEvalAll(benchmark::State &state) {
    using VdpfType = fss::Vdpf<in_bits, Group, Prg, XorHash, Hash, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename VdpfType::Cw cws[in_bits];
    cuda::std::array<int4, 4> cs;
    int4 ocw;

    constexpr size_t n = size_t{1} << in_bits;
    std::vector<int4> ys(n);

    auto xor_hash = MakeHash<XorHash>();
    auto hash_ = MakeHash<Hash>();
    AesCtx<2> ctx;
    VdpfType vdpf{ctx.prg, xor_hash, hash_};
    vdpf.Gen(cws, cs, ocw, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
    for (auto _ : state) {
        cuda::std::array<int4, 4> pi;
        vdpf.EvalAll(false, s0s[0], cuda::std::span<const typename VdpfType::Cw>(cws, in_bits),
            cuda::std::span<const int4, 4>(cs), ocw, cuda::std::span<int4>(ys.data(), n), pi);
        benchmark::DoNotOptimize(ys.data());
        benchmark::DoNotOptimize(pi);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- HalfTreeDpf benchmarks ---

template <int in_bits, typename Group, typename Prg>
static void BM_HalfTreeDpfGen(benchmark::State &state) {
    using HtDpfType = fss::HalfTreeDpf<in_bits, Group, Prg, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename HtDpfType::Cw cws[in_bits];
    int4 ocw;

    AesCtx<1> ctx;
    HtDpfType dpf{ctx.prg, {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, 0x2468ace0}};
    for (auto _ : state) {
        dpf.Gen(cws, ocw, s0s, alpha, beta);
        benchmark::DoNotOptimize(cws);
    }
}

template <int in_bits, typename Group, typename Prg>
static void BM_HalfTreeDpfEval(benchmark::State &state) {
    using HtDpfType = fss::HalfTreeDpf<in_bits, Group, Prg, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename HtDpfType::Cw cws[in_bits];
    int4 ocw;
    uint x = 100;

    AesCtx<1> ctx;
    HtDpfType dpf{ctx.prg, {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, 0x2468ace0}};
    dpf.Gen(cws, ocw, s0s, alpha, beta);
    for (auto _ : state) {
        int4 y = dpf.Eval(false, s0s[0], cws, ocw, x);
        benchmark::DoNotOptimize(y);
    }
}

template <int in_bits, typename Group, typename Prg>
static void BM_HalfTreeDpfEvalAll(benchmark::State &state) {
    using HtDpfType = fss::HalfTreeDpf<in_bits, Group, Prg, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    int4 beta = {7, 0, 0, 0};
    typename HtDpfType::Cw cws[in_bits];
    int4 ocw;

    constexpr size_t n = size_t{1} << in_bits;
    std::vector<int4> ys(n);

    AesCtx<1> ctx;
    HtDpfType dpf{ctx.prg, {0x12345678, static_cast<int>(0x9abcdef0u), 0x13572468, 0x2468ace0}};
    dpf.Gen(cws, ocw, s0s, alpha, beta);
    for (auto _ : state) {
        dpf.EvalAll(false, s0s[0], cws, ocw, ys.data());
        benchmark::DoNotOptimize(ys.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- GrottoDcf benchmarks ---

template <int in_bits, typename Prg>
static void BM_GrottoDcfEval(benchmark::State &state) {
    using GdcfType = fss::GrottoDcf<in_bits, Prg, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    typename GdcfType::Cw cws[in_bits + 1];

    constexpr size_t N = size_t{1} << in_bits;
    uint x = 100;

    AesCtx<2> ctx;
    GdcfType dcf{ctx.prg};
    dcf.Gen(cws, s0s, alpha);

    auto p = std::make_unique<bool[]>(2 * N - 1);
    typename GdcfType::ParityTree pt{p.get(), false};
    dcf.Preprocess(pt, s0s[0], cws);

    for (auto _ : state) {
        bool y = GdcfType::Eval(pt, x);
        benchmark::DoNotOptimize(y);
    }
}

template <int in_bits, typename Prg>
static void BM_GrottoDcfPreprocess(benchmark::State &state) {
    using GdcfType = fss::GrottoDcf<in_bits, Prg, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    typename GdcfType::Cw cws[in_bits + 1];

    constexpr size_t N = size_t{1} << in_bits;

    AesCtx<2> ctx;
    GdcfType dcf{ctx.prg};
    dcf.Gen(cws, s0s, alpha);

    auto p = std::make_unique<bool[]>(2 * N - 1);
    typename GdcfType::ParityTree pt{p.get(), false};

    for (auto _ : state) {
        dcf.Preprocess(pt, s0s[0], cws);
        benchmark::DoNotOptimize(p.get());
    }
}

template <int in_bits, typename Prg>
static void BM_GrottoDcfPreprocessEvalAll(benchmark::State &state) {
    using GdcfType = fss::GrottoDcf<in_bits, Prg, uint>;

    int4 s0s[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };
    uint alpha = 42;
    typename GdcfType::Cw cws[in_bits + 1];

    constexpr size_t N = size_t{1} << in_bits;

    AesCtx<2> ctx;
    GdcfType dcf{ctx.prg};
    dcf.Gen(cws, s0s, alpha);

    auto p = std::make_unique<bool[]>(2 * N - 1);
    typename GdcfType::ParityTree pt{p.get(), false};
    auto ys = std::make_unique<bool[]>(N);

    for (auto _ : state) {
        dcf.Preprocess(pt, s0s[0], cws);
        dcf.EvalAll(false, s0s[0], cws, ys.get());
        benchmark::DoNotOptimize(p.get());
        benchmark::DoNotOptimize(ys.get());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

// --- Registration ---

using DpfAes = fss::prg::Aes128Mmo<2>;
using DpfChaCha = fss::prg::ChaCha<2>;
using DpfAesSoft = fss::prg::Aes128Soft<2>;
using DcfAes = fss::prg::Aes128Mmo<4>;
using HtDpfAes = fss::prg::Aes128Mmo<1>;
using GdcfAes = fss::prg::Aes128Mmo<2>;
using Sha256 = fss::hash::Sha256;
using Blake3 = fss::hash::Blake3;

// 1. BM_DpfEval_Uint_Aes/20
BENCHMARK(BM_DpfEval<20, UintGroup, DpfAes>)->Name("BM_DpfEval_Uint_Aes/20");
// 2. BM_DpfEval_Uint_Aes/14
BENCHMARK(BM_DpfEval<14, UintGroup, DpfAes>)->Name("BM_DpfEval_Uint_Aes/14");
// 3. BM_DpfEval_Uint_Aes/17
BENCHMARK(BM_DpfEval<17, UintGroup, DpfAes>)->Name("BM_DpfEval_Uint_Aes/17");
// 4. BM_DpfGen_Uint_Aes/20
BENCHMARK(BM_DpfGen<20, UintGroup, DpfAes>)->Name("BM_DpfGen_Uint_Aes/20");
// 5. BM_DpfEval_Bytes_Aes/20
BENCHMARK(BM_DpfEval<20, BytesGroup, DpfAes>)->Name("BM_DpfEval_Bytes_Aes/20");
// 6. BM_DpfEvalAll_Uint_Aes/20
BENCHMARK(BM_DpfEvalAll<20, UintGroup, DpfAes>)->Name("BM_DpfEvalAll_Uint_Aes/20");

// 7. BM_DpfEval_Uint_ChaCha/20
BENCHMARK(BM_DpfEval<20, UintGroup, DpfChaCha>)->Name("BM_DpfEval_Uint_ChaCha/20");
// 8. BM_DpfEval_Uint_AesSoft/20
BENCHMARK(BM_DpfEval<20, UintGroup, DpfAesSoft>)->Name("BM_DpfEval_Uint_AesSoft/20");

// 9. BM_DcfEval_Uint_Aes/20
BENCHMARK(BM_DcfEval<20, UintGroup, DcfAes>)->Name("BM_DcfEval_Uint_Aes/20");
// 10. BM_DcfGen_Uint_Aes/20
BENCHMARK(BM_DcfGen<20, UintGroup, DcfAes>)->Name("BM_DcfGen_Uint_Aes/20");
// 11. BM_DcfEval_Bytes_Aes/20
BENCHMARK(BM_DcfEval<20, BytesGroup, DcfAes>)->Name("BM_DcfEval_Bytes_Aes/20");
// 12. BM_DcfEvalAll_Uint_Aes/20
BENCHMARK(BM_DcfEvalAll<20, UintGroup, DcfAes>)->Name("BM_DcfEvalAll_Uint_Aes/20");
// 13. BM_DcfEvalAll_Bytes_Aes/20
BENCHMARK(BM_DcfEvalAll<20, BytesGroup, DcfAes>)->Name("BM_DcfEvalAll_Bytes_Aes/20");

// 14. BM_VdpfEval_Uint_Aes_Sha256/20
BENCHMARK((BM_VdpfEval<20, UintGroup, DpfAes, Sha256, Sha256>))
    ->Name("BM_VdpfEval_Uint_Aes_Sha256/20");
// 15. BM_VdpfGen_Uint_Aes_Sha256/20
BENCHMARK((BM_VdpfGen<20, UintGroup, DpfAes, Sha256, Sha256>))
    ->Name("BM_VdpfGen_Uint_Aes_Sha256/20");
// 16. BM_VdpfEval_Uint_Aes_Blake3/20
BENCHMARK((BM_VdpfEval<20, UintGroup, DpfAes, Blake3, Blake3>))
    ->Name("BM_VdpfEval_Uint_Aes_Blake3/20");
// 17. BM_VdpfProve_Uint_ChaCha_Blake3/20
BENCHMARK((BM_VdpfProve<20, UintGroup, DpfChaCha, Blake3, Blake3>))
    ->Name("BM_VdpfProve_Uint_ChaCha_Blake3/20");
// 18. BM_VdpfEvalAll_Uint_Aes_Sha256/20
BENCHMARK((BM_VdpfEvalAll<20, UintGroup, DpfAes, Sha256, Sha256>))
    ->Name("BM_VdpfEvalAll_Uint_Aes_Sha256/20");

// 19. BM_HalfTreeDpfEval_Uint_Aes/20
BENCHMARK(BM_HalfTreeDpfEval<20, UintGroup, HtDpfAes>)->Name("BM_HalfTreeDpfEval_Uint_Aes/20");
// 20. BM_HalfTreeDpfGen_Uint_Aes/20
BENCHMARK(BM_HalfTreeDpfGen<20, UintGroup, HtDpfAes>)->Name("BM_HalfTreeDpfGen_Uint_Aes/20");
// 21. BM_HalfTreeDpfEvalAll_Uint_Aes/20
BENCHMARK(BM_HalfTreeDpfEvalAll<20, UintGroup, HtDpfAes>)
    ->Name("BM_HalfTreeDpfEvalAll_Uint_Aes/20");

// 22. BM_GrottoDcfEval_Aes/20
BENCHMARK(BM_GrottoDcfEval<20, GdcfAes>)->Name("BM_GrottoDcfEval_Aes/20");
// 23. BM_GrottoDcfPreprocess_Aes/20
BENCHMARK(BM_GrottoDcfPreprocess<20, GdcfAes>)->Name("BM_GrottoDcfPreprocess_Aes/20");
// 24. BM_GrottoDcfPreprocessEvalAll_Aes/20
BENCHMARK(BM_GrottoDcfPreprocessEvalAll<20, GdcfAes>)->Name("BM_GrottoDcfPreprocessEvalAll_Aes/20");
