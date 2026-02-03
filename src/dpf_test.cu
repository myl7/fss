// SPDX-License-Identifier: MIT
/**
 * @file dpf_test.cu
 * @copyright MIT License. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
using Uint64Group = fss::group::Uint<uint64_t>;

// Helper to determine comparison size for each group type
template <typename Group>
constexpr size_t GroupCompareSize() {
    if constexpr (std::is_same_v<Group, Uint64Group>) {
        return 8;  // Compare only first 8 bytes for uint64
    } else if constexpr (std::is_same_v<Group, Uint127Group>) {
        return 16;  // Compare 16 bytes (127 bits + control bit which is always 0)
    } else {
        return sizeof(int4);  // Full 16 bytes for Bytes group
    }
}

// Test parameters
constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;
constexpr __uint128_t kBeta = 604;

// Random nonce for ChaCha
static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

// Test fixture template for DPF
template <typename Group, typename Prg>
class DpfTestBase : public ::testing::Test {
protected:
    using DpfType = fss::Dpf<kAlphaBits, Group, Prg, uint16_t>;

    int4 s0s[2];
    typename DpfType::Cw cws[kAlphaBits + 1];
    Prg prg;

    DpfTestBase(Prg p) : prg(p) {}

    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    }

    void TestEvalAtAlpha() {
        DpfType dpf{prg};

        int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
        dpf.Gen(cws, s0s, kAlpha, b_buf);

        // Eval at alpha
        int4 y0 = dpf.Eval(false, s0s[0], cws, kAlpha);
        int4 y1 = dpf.Eval(true, s0s[1], cws, kAlpha);

        auto g0 = Group::From(y0);
        auto g1 = Group::From(y1);
        auto result = g0 + g1;
        auto expected = Group::From(b_buf);

        int4 result_buf = result.Into();
        int4 expected_buf = expected.Into();

        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
    }

    void TestEvalAtNonAlpha() {
        DpfType dpf{prg};

        int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
        dpf.Gen(cws, s0s, kAlpha, b_buf);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

        for (int i = 0; i < 100; i++) {
            uint16_t x = dis(gen);
            if (x == kAlpha) continue;

            int4 y0 = dpf.Eval(false, s0s[0], cws, x);
            int4 y1 = dpf.Eval(true, s0s[1], cws, x);

            auto g0 = Group::From(y0);
            auto g1 = Group::From(y1);
            auto result = g0 + g1;

            // Check that result equals identity (0)
            int4 result_buf = result.Into();
            int4 zero = {0, 0, 0, 0};

            EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                << "Failed at x=" << x;
        }
    }
};

// ChaCha PRG tests
class DpfBytesChaChaTest : public DpfTestBase<BytesGroup, fss::prg::ChaCha<2>> {
public:
    DpfBytesChaChaTest() : DpfTestBase(fss::prg::ChaCha<2>(gChaChaDeviceNonces)) {}
};

class DpfUint128ChaChaTest : public DpfTestBase<Uint127Group, fss::prg::ChaCha<2>> {
public:
    DpfUint128ChaChaTest() : DpfTestBase(fss::prg::ChaCha<2>(gChaChaDeviceNonces)) {}
};

class DpfUint64ChaChaTest : public DpfTestBase<Uint64Group, fss::prg::ChaCha<2>> {
public:
    DpfUint64ChaChaTest() : DpfTestBase(fss::prg::ChaCha<2>(gChaChaDeviceNonces)) {}
};

// AES PRG tests
class DpfBytesAesTest : public ::testing::Test {
protected:
    using Group = BytesGroup;
    using Prg = fss::prg::Aes128Mmo<2>;
    using DpfType = fss::Dpf<kAlphaBits, Group, Prg, uint16_t>;

    int4 s0s[2];
    DpfType::Cw cws[kAlphaBits + 1];
    EVP_CIPHER_CTX *ctxs[2];

    void SetUp() override {
        unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        const unsigned char *keys[2] = {key0, key1};
        auto ctx_arr = Prg::InitCtxs(keys);
        ctxs[0] = ctx_arr[0];
        ctxs[1] = ctx_arr[1];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    }

    void TearDown() override {
        Prg::FreeCtxs(ctxs);
    }
};

// Tests for ChaCha
TEST_F(DpfBytesChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(DpfBytesChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(DpfUint128ChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(DpfUint128ChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(DpfUint64ChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(DpfUint64ChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}

// Tests for AES
TEST_F(DpfBytesAesTest, EvalAtAlpha) {
    Prg prg(ctxs);
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 32) & 0xFFFFFFFF), static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    dpf.Gen(cws, s0s, kAlpha, b_buf);

    int4 y0 = dpf.Eval(false, s0s[0], cws, kAlpha);
    int4 y1 = dpf.Eval(true, s0s[1], cws, kAlpha);

    auto g0 = Group::From(y0);
    auto g1 = Group::From(y1);
    auto result = g0 + g1;
    auto expected = Group::From(b_buf);

    int4 result_buf = result.Into();
    int4 expected_buf = expected.Into();
    EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
}

TEST_F(DpfBytesAesTest, EvalAtNonAlpha) {
    Prg prg(ctxs);
    DpfType dpf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 32) & 0xFFFFFFFF), static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    dpf.Gen(cws, s0s, kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

    for (int i = 0; i < 100; i++) {
        uint16_t x = dis(gen);
        if (x == kAlpha) continue;

        int4 y0 = dpf.Eval(false, s0s[0], cws, x);
        int4 y1 = dpf.Eval(true, s0s[1], cws, x);

        auto g0 = Group::From(y0);
        auto g1 = Group::From(y1);
        auto result = g0 + g1;

        int4 result_buf = result.Into();
        int4 zero = {0, 0, 0, 0};
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
    }
}
