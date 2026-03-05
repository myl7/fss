#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prg/chacha.cuh>

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
        if constexpr (mul == 2) {
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

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<4>>) {
        AesCtx<4> ctx;
        DcfType dcf{ctx.prg};
        for (auto _ : state) {
            dcf.Gen(cws, seeds, alpha, beta);
            benchmark::DoNotOptimize(cws);
        }
    } else {
        Prg prg(gNonce);
        DcfType dcf{prg};
        for (auto _ : state) {
            dcf.Gen(cws, seeds, alpha, beta);
            benchmark::DoNotOptimize(cws);
        }
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

    if constexpr (std::is_same_v<Prg, fss::prg::Aes128Mmo<4>>) {
        AesCtx<4> ctx;
        DcfType dcf{ctx.prg};
        dcf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            int4 y = dcf.Eval(false, seeds[0], cws, x);
            benchmark::DoNotOptimize(y);
        }
    } else {
        Prg prg(gNonce);
        DcfType dcf{prg};
        dcf.Gen(cws, seeds, alpha, beta);
        for (auto _ : state) {
            int4 y = dcf.Eval(false, seeds[0], cws, x);
            benchmark::DoNotOptimize(y);
        }
    }
}

// --- Registration ---
// Combinations: 14/Uint/Aes, 17/Uint/Aes, 20/Uint/Aes, 20/Bytes/Aes, 20/Uint/ChaCha

using DpfAes = fss::prg::Aes128Mmo<2>;
using DpfChaCha = fss::prg::ChaCha<2>;
using DcfAes = fss::prg::Aes128Mmo<4>;
using DcfChaCha = fss::prg::ChaCha<4>;

// DPF
BENCHMARK(BM_DpfGen<14, UintGroup, DpfAes>)->Name("BM_DpfGen_Uint_Aes/14");
BENCHMARK(BM_DpfGen<17, UintGroup, DpfAes>)->Name("BM_DpfGen_Uint_Aes/17");
BENCHMARK(BM_DpfGen<20, UintGroup, DpfAes>)->Name("BM_DpfGen_Uint_Aes/20");
BENCHMARK(BM_DpfGen<20, BytesGroup, DpfAes>)->Name("BM_DpfGen_Bytes_Aes/20");
BENCHMARK(BM_DpfGen<20, UintGroup, DpfChaCha>)->Name("BM_DpfGen_Uint_ChaCha/20");

BENCHMARK(BM_DpfEval<14, UintGroup, DpfAes>)->Name("BM_DpfEval_Uint_Aes/14");
BENCHMARK(BM_DpfEval<17, UintGroup, DpfAes>)->Name("BM_DpfEval_Uint_Aes/17");
BENCHMARK(BM_DpfEval<20, UintGroup, DpfAes>)->Name("BM_DpfEval_Uint_Aes/20");
BENCHMARK(BM_DpfEval<20, BytesGroup, DpfAes>)->Name("BM_DpfEval_Bytes_Aes/20");
BENCHMARK(BM_DpfEval<20, UintGroup, DpfChaCha>)->Name("BM_DpfEval_Uint_ChaCha/20");

// DCF
BENCHMARK(BM_DcfGen<14, UintGroup, DcfAes>)->Name("BM_DcfGen_Uint_Aes/14");
BENCHMARK(BM_DcfGen<17, UintGroup, DcfAes>)->Name("BM_DcfGen_Uint_Aes/17");
BENCHMARK(BM_DcfGen<20, UintGroup, DcfAes>)->Name("BM_DcfGen_Uint_Aes/20");
BENCHMARK(BM_DcfGen<20, BytesGroup, DcfAes>)->Name("BM_DcfGen_Bytes_Aes/20");
BENCHMARK(BM_DcfGen<20, UintGroup, DcfChaCha>)->Name("BM_DcfGen_Uint_ChaCha/20");

BENCHMARK(BM_DcfEval<14, UintGroup, DcfAes>)->Name("BM_DcfEval_Uint_Aes/14");
BENCHMARK(BM_DcfEval<17, UintGroup, DcfAes>)->Name("BM_DcfEval_Uint_Aes/17");
BENCHMARK(BM_DcfEval<20, UintGroup, DcfAes>)->Name("BM_DcfEval_Uint_Aes/20");
BENCHMARK(BM_DcfEval<20, BytesGroup, DcfAes>)->Name("BM_DcfEval_Bytes_Aes/20");
BENCHMARK(BM_DcfEval<20, UintGroup, DcfChaCha>)->Name("BM_DcfEval_Uint_ChaCha/20");
