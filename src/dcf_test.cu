// SPDX-License-Identifier: MIT
/**
 * @file dcf_test.cu
 * @copyright MIT License. Copyright (C) 2026 Yulong Ming <i@myl.moe>.
 * @author Yulong Ming <i@myl.moe>
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <fss/dcf.cuh>
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

// Random nonce for ChaCha (DCF needs mul=4)
static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

// Test fixture template for DCF
template <typename Group, typename Prg, fss::DcfPred pred = fss::DcfPred::kLt>
class DcfTestBase : public ::testing::Test {
protected:
    using DcfType = fss::Dcf<kAlphaBits, Group, Prg, uint16_t, pred>;

    int4 s0s[2];
    typename DcfType::Cw cws[kAlphaBits + 1];
    Prg prg;

    DcfTestBase(Prg p) : prg(p) {}

    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    }

    void TestEvalLt() {
        DcfType dcf{prg};

        int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
        dcf.Gen(cws, s0s, kAlpha, b_buf);

        std::random_device rd;
        std::mt19937 gen(rd());

        // Test x < alpha should return beta
        std::uniform_int_distribution<uint16_t> dis_lt(0, kAlpha - 1);
        for (int i = 0; i < 50; i++) {
            uint16_t x = dis_lt(gen);

            int4 y0 = dcf.Eval(false, s0s[0], cws, x);
            int4 y1 = dcf.Eval(true, s0s[1], cws, x);

            auto g0 = Group::From(y0);
            auto g1 = Group::From(y1);
            auto result = g0 + g1;
            auto expected = Group::From(b_buf);

            int4 result_buf = result.Into();
            int4 expected_buf = expected.Into();
            EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0)
                << "Failed at x=" << x;
        }

        // Test x >= alpha should return 0
        std::uniform_int_distribution<uint16_t> dis_ge(kAlpha, 0xFFFF);
        for (int i = 0; i < 50; i++) {
            uint16_t x = dis_ge(gen);

            int4 y0 = dcf.Eval(false, s0s[0], cws, x);
            int4 y1 = dcf.Eval(true, s0s[1], cws, x);

            auto g0 = Group::From(y0);
            auto g1 = Group::From(y1);
            auto result = g0 + g1;

            int4 result_buf = result.Into();
            int4 zero = {0, 0, 0, 0};
            EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                << "Failed at x=" << x;
        }
    }
};

// ChaCha PRG tests (mul=4 for DCF)
class DcfBytesChaChaTest : public DcfTestBase<BytesGroup, fss::prg::ChaCha<4>> {
public:
    DcfBytesChaChaTest() : DcfTestBase(fss::prg::ChaCha<4>(gChaChaDeviceNonces)) {}
};

class DcfUint128ChaChaTest : public DcfTestBase<Uint127Group, fss::prg::ChaCha<4>> {
public:
    DcfUint128ChaChaTest() : DcfTestBase(fss::prg::ChaCha<4>(gChaChaDeviceNonces)) {}
};

class DcfUint64ChaChaTest : public DcfTestBase<Uint64Group, fss::prg::ChaCha<4>> {
public:
    DcfUint64ChaChaTest() : DcfTestBase(fss::prg::ChaCha<4>(gChaChaDeviceNonces)) {}
};

// AES PRG tests (mul=4 for DCF)
class DcfBytesAesTest : public ::testing::Test {
protected:
    using Group = BytesGroup;
    using Prg = fss::prg::Aes128Mmo<4>;
    using DcfType = fss::Dcf<kAlphaBits, Group, Prg, uint16_t>;

    int4 s0s[2];
    DcfType::Cw cws[kAlphaBits + 1];
    EVP_CIPHER_CTX *ctxs[4];

    void SetUp() override {
        unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        unsigned char key2[16] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
        unsigned char key3[16] = {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
        const unsigned char *keys[4] = {key0, key1, key2, key3};
        auto ctx_arr = Prg::InitCtxs(keys);
        for (int i = 0; i < 4; i++) ctxs[i] = ctx_arr[i];

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
TEST_F(DcfBytesChaChaTest, EvalLt) {
    TestEvalLt();
}
TEST_F(DcfUint128ChaChaTest, EvalLt) {
    TestEvalLt();
}
TEST_F(DcfUint64ChaChaTest, EvalLt) {
    TestEvalLt();
}

// Tests for AES
TEST_F(DcfBytesAesTest, EvalLt) {
    Prg prg(ctxs);
    DcfType dcf{prg};

    int4 b_buf = {static_cast<int>(kBeta & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 32) & 0xFFFFFFFF), static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
        static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    dcf.Gen(cws, s0s, kAlpha, b_buf);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Test x < alpha
    std::uniform_int_distribution<uint16_t> dis_lt(0, kAlpha - 1);
    for (int i = 0; i < 50; i++) {
        uint16_t x = dis_lt(gen);

        int4 y0 = dcf.Eval(false, s0s[0], cws, x);
        int4 y1 = dcf.Eval(true, s0s[1], cws, x);

        auto g0 = Group::From(y0);
        auto g1 = Group::From(y1);
        auto result = g0 + g1;
        auto expected = Group::From(b_buf);

        int4 result_buf = result.Into();
        int4 expected_buf = expected.Into();
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
    }

    // Test x >= alpha
    std::uniform_int_distribution<uint16_t> dis_ge(kAlpha, 0xFFFF);
    for (int i = 0; i < 50; i++) {
        uint16_t x = dis_ge(gen);

        int4 y0 = dcf.Eval(false, s0s[0], cws, x);
        int4 y1 = dcf.Eval(true, s0s[1], cws, x);

        auto g0 = Group::From(y0);
        auto g1 = Group::From(y1);
        auto result = g0 + g1;

        int4 result_buf = result.Into();
        int4 zero = {0, 0, 0, 0};
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
    }
}
