// Sample: DPF and DCF on CPU
//
// Shows how to use Gen/Eval for both DPF (point function) and DCF (comparison
// function) entirely on the host side, using AES-128 MMO PRG.
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/aes128_mmo.cuh>

// 8-bit input domain for a small example
constexpr int kInBits = 8;
using In = uint8_t;
using Group = fss::group::Bytes;

// DPF uses mul=2, DCF uses mul=4
using DpfPrg = fss::prg::Aes128Mmo<2>;
using DcfPrg = fss::prg::Aes128Mmo<4>;
using Dpf = fss::Dpf<kInBits, Group, DpfPrg, In>;
using Dcf = fss::Dcf<kInBits, Group, DcfPrg, In>;

// Compare two int4 values
static bool Equal(int4 a, int4 b) {
    return memcmp(&a, &b, sizeof(int4)) == 0;
}

// Reconstruct: convert Eval outputs to group elements, add, convert back
static int4 Reconstruct(int4 y0, int4 y1) {
    return (Group::From(y0) + Group::From(y1)).Into();
}

static void DpfSample() {
    printf("=== DPF Sample ===\n");

    // Create AES cipher contexts with 2 keys (mul=2 for DPF)
    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const unsigned char *keys[2] = {key0, key1};
    auto ctxs = DpfPrg::CreateCtxs(keys);

    DpfPrg prg(ctxs);
    Dpf dpf{prg};

    // Secret inputs: alpha (point), beta (payload)
    In alpha = 42;
    int4 beta = {7, 0, 0, 0};  // LSB of .w must be 0

    // Random seeds for the two parties (LSB of .w must be 0)
    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };

    // Key generation (done by a trusted dealer)
    Dpf::Cw cws[kInBits + 1];
    dpf.Gen(cws, seeds, alpha, beta);

    // Evaluation (each party runs independently)
    // At x == alpha: y0 + y1 == beta
    int4 y0 = dpf.Eval(false, seeds[0], cws, alpha);
    int4 y1 = dpf.Eval(true, seeds[1], cws, alpha);
    int4 sum = Reconstruct(y0, y1);
    printf("  Eval(x=%d == alpha): y0+y1 == beta? %s\n", alpha, Equal(sum, beta) ? "yes" : "NO");

    // At x != alpha: y0 + y1 == 0
    int4 zero = {0, 0, 0, 0};
    In x = 100;
    y0 = dpf.Eval(false, seeds[0], cws, x);
    y1 = dpf.Eval(true, seeds[1], cws, x);
    sum = Reconstruct(y0, y1);
    printf("  Eval(x=%d != alpha): y0+y1 == 0?    %s\n", x, Equal(sum, zero) ? "yes" : "NO");

    DpfPrg::FreeCtxs(ctxs);
}

static void DcfSample() {
    printf("=== DCF Sample ===\n");

    // Create AES cipher contexts with 4 keys (mul=4 for DCF)
    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    unsigned char key2[16] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
    unsigned char key3[16] = {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
    const unsigned char *keys[4] = {key0, key1, key2, key3};
    auto ctxs = DcfPrg::CreateCtxs(keys);

    DcfPrg prg(ctxs);
    Dcf dcf{prg};

    // Secret inputs: alpha (threshold), beta (payload)
    In alpha = 42;
    int4 beta = {7, 0, 0, 0};

    // Random seeds (LSB of .w must be 0)
    int4 seeds[2] = {
        {0x11111111, 0x22222222, 0x33333333, 0x44444440},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
    };

    // Key generation
    Dcf::Cw cws[kInBits + 1];
    dcf.Gen(cws, seeds, alpha, beta);

    // Evaluation
    int4 zero = {0, 0, 0, 0};

    // x < alpha: y0 + y1 == beta
    In x_lt = 10;
    int4 y0 = dcf.Eval(false, seeds[0], cws, x_lt);
    int4 y1 = dcf.Eval(true, seeds[1], cws, x_lt);
    int4 sum = Reconstruct(y0, y1);
    printf("  Eval(x=%d  < alpha): y0+y1 == beta? %s\n", x_lt, Equal(sum, beta) ? "yes" : "NO");

    // x == alpha: y0 + y1 == 0
    y0 = dcf.Eval(false, seeds[0], cws, alpha);
    y1 = dcf.Eval(true, seeds[1], cws, alpha);
    sum = Reconstruct(y0, y1);
    printf("  Eval(x=%d == alpha): y0+y1 == 0?    %s\n", alpha, Equal(sum, zero) ? "yes" : "NO");

    // x > alpha: y0 + y1 == 0
    In x_gt = 200;
    y0 = dcf.Eval(false, seeds[0], cws, x_gt);
    y1 = dcf.Eval(true, seeds[1], cws, x_gt);
    sum = Reconstruct(y0, y1);
    printf("  Eval(x=%d > alpha): y0+y1 == 0?    %s\n", x_gt, Equal(sum, zero) ? "yes" : "NO");

    DcfPrg::FreeCtxs(ctxs);
}

int main() {
    DpfSample();
    printf("\n");
    DcfSample();
    return 0;
}
