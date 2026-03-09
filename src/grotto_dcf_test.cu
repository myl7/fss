#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <memory>
#include <fss/grotto_dcf.cuh>
#include <fss/prg/chacha.cuh>

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;

static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

class GrottoDcfChaChaTest : public ::testing::Test {
protected:
    using GrottoDcfType = fss::GrottoDcf<kAlphaBits, fss::prg::ChaCha<2>, uint16_t>;

    int4 s0s[2];
    GrottoDcfType::Cw cws[kAlphaBits + 1];
    fss::prg::ChaCha<2> prg;

    GrottoDcfChaChaTest() : prg(gChaChaDeviceNonces) {}

    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};

        GrottoDcfType dcf{prg};
        dcf.Gen(cws, s0s, kAlpha);
    }
};

// Test Preprocess + Eval for all x in domain
TEST_F(GrottoDcfChaChaTest, PreprocessEvalAll) {
    GrottoDcfType dcf{prg};
    constexpr size_t N = 1ULL << kAlphaBits;

    auto p0 = std::make_unique<bool[]>(2 * N - 1);
    auto p1 = std::make_unique<bool[]>(2 * N - 1);
    GrottoDcfType::ParityTree pt0{p0.get(), false};
    GrottoDcfType::ParityTree pt1{p1.get(), true};

    dcf.Preprocess(pt0, s0s[0], cws);
    dcf.Preprocess(pt1, s0s[1], cws);

    for (size_t x = 0; x < N; ++x) {
        bool share0 = GrottoDcfType::Eval(pt0, static_cast<uint16_t>(x));
        bool share1 = GrottoDcfType::Eval(pt1, static_cast<uint16_t>(x));
        bool result = share0 ^ share1;
        bool expected = (kAlpha <= x);
        EXPECT_EQ(result, expected) << "Failed at x=" << x;
    }
}

// Test EvalAll
TEST_F(GrottoDcfChaChaTest, EvalAll) {
    GrottoDcfType dcf{prg};
    constexpr size_t N = 1ULL << kAlphaBits;

    auto ys0 = std::make_unique<bool[]>(N);
    auto ys1 = std::make_unique<bool[]>(N);

    dcf.EvalAll(false, s0s[0], cws, ys0.get());
    dcf.EvalAll(true, s0s[1], cws, ys1.get());

    for (size_t x = 0; x < N; ++x) {
        bool result = ys0[x] ^ ys1[x];
        bool expected = (kAlpha <= x);
        EXPECT_EQ(result, expected) << "Failed at x=" << x;
    }
}

// Test Preprocess + Eval matches EvalAll
TEST_F(GrottoDcfChaChaTest, PreprocessEvalMatchesEvalAll) {
    GrottoDcfType dcf{prg};
    constexpr size_t N = 1ULL << kAlphaBits;

    // Preprocess path
    auto p0 = std::make_unique<bool[]>(2 * N - 1);
    GrottoDcfType::ParityTree pt0{p0.get(), false};
    dcf.Preprocess(pt0, s0s[0], cws);

    // EvalAll path
    auto ys0 = std::make_unique<bool[]>(N);
    dcf.EvalAll(false, s0s[0], cws, ys0.get());

    for (size_t x = 0; x < N; ++x) {
        bool from_preprocess = GrottoDcfType::Eval(pt0, static_cast<uint16_t>(x));
        EXPECT_EQ(from_preprocess, ys0[x]) << "Mismatch at x=" << x;
    }
}

// Test edge cases: alpha=0 and alpha=N-1
TEST_F(GrottoDcfChaChaTest, EdgeAlphaZero) {
    GrottoDcfType dcf{prg};
    constexpr size_t N = 1ULL << kAlphaBits;

    GrottoDcfType::Cw edge_cws[kAlphaBits + 1];
    dcf.Gen(edge_cws, s0s, static_cast<uint16_t>(0));

    auto ys0 = std::make_unique<bool[]>(N);
    auto ys1 = std::make_unique<bool[]>(N);
    dcf.EvalAll(false, s0s[0], edge_cws, ys0.get());
    dcf.EvalAll(true, s0s[1], edge_cws, ys1.get());

    for (size_t x = 0; x < N; ++x) {
        bool result = ys0[x] ^ ys1[x];
        // alpha=0, so 1[0 <= x] = true for all x
        EXPECT_TRUE(result) << "Failed at x=" << x;
    }
}

TEST_F(GrottoDcfChaChaTest, EdgeAlphaMax) {
    GrottoDcfType dcf{prg};
    constexpr size_t N = 1ULL << kAlphaBits;

    uint16_t alpha_max = static_cast<uint16_t>(N - 1);
    GrottoDcfType::Cw edge_cws[kAlphaBits + 1];
    dcf.Gen(edge_cws, s0s, alpha_max);

    auto ys0 = std::make_unique<bool[]>(N);
    auto ys1 = std::make_unique<bool[]>(N);
    dcf.EvalAll(false, s0s[0], edge_cws, ys0.get());
    dcf.EvalAll(true, s0s[1], edge_cws, ys1.get());

    for (size_t x = 0; x < N; ++x) {
        bool result = ys0[x] ^ ys1[x];
        // alpha=N-1, so 1[N-1 <= x] = true only at x=N-1
        bool expected = (x == N - 1);
        EXPECT_EQ(result, expected) << "Failed at x=" << x;
    }
}
