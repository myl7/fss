#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/half_tree_dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/prg/aes128_mmo_soft.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
using Uint64Group = fss::group::Uint<uint64_t>;

template <typename Group>
constexpr size_t GroupCompareSize() {
    if constexpr (std::is_same_v<Group, Uint64Group>) {
        return 8;
    } else if constexpr (std::is_same_v<Group, Uint127Group>) {
        return 16;
    } else {
        return sizeof(int4);
    }
}

constexpr uint16_t kAlpha = 107;
constexpr int kAlphaBits = 16;
constexpr __uint128_t kBeta = 604;

static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

// ChaCha-based test fixture
template <typename Group>
class HalfTreeDpfChaChaTest : public ::testing::Test {
protected:
    using Prg = fss::prg::ChaCha<1>;
    using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, Prg, uint16_t>;

    int4 s0s[2];
    typename DpfType::Cw cws[kAlphaBits];
    int4 ocw;
    Prg prg;
    int4 hash_key;

    HalfTreeDpfChaChaTest() : prg(gChaChaDeviceNonces) {}

    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
    }

    int4 MakeBBuf() {
        return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    }

    void TestEvalAtAlpha() {
        DpfType dpf{prg, hash_key};

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, kAlpha, b_buf);

        int4 y0 = dpf.Eval(false, s0s[0], cws, ocw, kAlpha);
        int4 y1 = dpf.Eval(true, s0s[1], cws, ocw, kAlpha);

        auto g0 = Group::From(y0);
        auto g1 = Group::From(y1);
        auto result = g0 + g1;
        auto expected = Group::From(b_buf);

        int4 result_buf = result.Into();
        int4 expected_buf = expected.Into();

        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
    }

    void TestEvalAtNonAlpha() {
        DpfType dpf{prg, hash_key};

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, kAlpha, b_buf);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

        for (int i = 0; i < 100; i++) {
            uint16_t x = dis(gen);
            if (x == kAlpha) continue;

            int4 y0 = dpf.Eval(false, s0s[0], cws, ocw, x);
            int4 y1 = dpf.Eval(true, s0s[1], cws, ocw, x);

            auto g0 = Group::From(y0);
            auto g1 = Group::From(y1);
            auto result = g0 + g1;

            int4 result_buf = result.Into();
            int4 zero = {0, 0, 0, 0};

            EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                << "Failed at x=" << x;
        }
    }

    void TestEvalAll() {
        DpfType dpf{prg, hash_key};

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, kAlpha, b_buf);

        constexpr size_t n = 1ULL << kAlphaBits;
        std::vector<int4> ys0(n), ys1(n);

        dpf.EvalAll(false, s0s[0], cws, ocw, ys0.data());
        dpf.EvalAll(true, s0s[1], cws, ocw, ys1.data());

        auto expected = Group::From(b_buf);
        int4 expected_buf = expected.Into();
        int4 zero = {0, 0, 0, 0};

        for (size_t x = 0; x < n; ++x) {
            auto g0 = Group::From(ys0[x]);
            auto g1 = Group::From(ys1[x]);
            auto result = g0 + g1;
            int4 result_buf = result.Into();

            if (x == kAlpha) {
                EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0)
                    << "Failed at alpha=" << x;
            } else {
                EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                    << "Failed at x=" << x;
            }
        }
    }
};

using HalfTreeDpfBytesChaChaTest = HalfTreeDpfChaChaTest<BytesGroup>;
using HalfTreeDpfUint128ChaChaTest = HalfTreeDpfChaChaTest<Uint127Group>;
using HalfTreeDpfUint64ChaChaTest = HalfTreeDpfChaChaTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesChaChaTest, EvalAll) {
    TestEvalAll();
}

TEST_F(HalfTreeDpfUint128ChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128ChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128ChaChaTest, EvalAll) {
    TestEvalAll();
}

TEST_F(HalfTreeDpfUint64ChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64ChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64ChaChaTest, EvalAll) {
    TestEvalAll();
}

// Aes128Soft-based test fixture
template <typename Group>
class HalfTreeDpfAesSoftTest : public ::testing::Test {
protected:
    using Prg = fss::prg::Aes128Soft<1>;
    using DpfType = fss::HalfTreeDpf<kAlphaBits, Group, Prg, uint16_t>;

    int4 s0s[2];
    typename DpfType::Cw cws[kAlphaBits];
    int4 ocw;
    uint8_t aes_key[1][16] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};
    int4 hash_key;

    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
    }

    int4 MakeBBuf() {
        return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    }

    void TestEvalAtAlpha() {
        uint32_t te0[256];
        uint8_t sbox[256];
        fss::prg::aes_detail::InitTe0(te0);
        fss::prg::aes_detail::InitSbox(sbox);
        Prg prg(aes_key, te0, sbox);
        DpfType dpf{prg, hash_key};

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, kAlpha, b_buf);

        int4 y0 = dpf.Eval(false, s0s[0], cws, ocw, kAlpha);
        int4 y1 = dpf.Eval(true, s0s[1], cws, ocw, kAlpha);

        auto result = Group::From(y0) + Group::From(y1);
        auto expected = Group::From(b_buf);

        int4 result_buf = result.Into();
        int4 expected_buf = expected.Into();
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
    }

    void TestEvalAtNonAlpha() {
        uint32_t te0[256];
        uint8_t sbox[256];
        fss::prg::aes_detail::InitTe0(te0);
        fss::prg::aes_detail::InitSbox(sbox);
        Prg prg(aes_key, te0, sbox);
        DpfType dpf{prg, hash_key};

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, kAlpha, b_buf);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

        for (int i = 0; i < 100; i++) {
            uint16_t x = dis(gen);
            if (x == kAlpha) continue;

            int4 y0 = dpf.Eval(false, s0s[0], cws, ocw, x);
            int4 y1 = dpf.Eval(true, s0s[1], cws, ocw, x);

            auto result = Group::From(y0) + Group::From(y1);
            int4 result_buf = result.Into();
            int4 zero = {0, 0, 0, 0};
            EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                << "Failed at x=" << x;
        }
    }

    void TestEvalAll() {
        uint32_t te0[256];
        uint8_t sbox[256];
        fss::prg::aes_detail::InitTe0(te0);
        fss::prg::aes_detail::InitSbox(sbox);
        Prg prg(aes_key, te0, sbox);
        DpfType dpf{prg, hash_key};

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, kAlpha, b_buf);

        constexpr size_t n = 1ULL << kAlphaBits;
        std::vector<int4> ys0(n), ys1(n);

        dpf.EvalAll(false, s0s[0], cws, ocw, ys0.data());
        dpf.EvalAll(true, s0s[1], cws, ocw, ys1.data());

        auto expected = Group::From(b_buf);
        int4 expected_buf = expected.Into();
        int4 zero = {0, 0, 0, 0};

        for (size_t x = 0; x < n; ++x) {
            auto g0 = Group::From(ys0[x]);
            auto g1 = Group::From(ys1[x]);
            auto result = g0 + g1;
            int4 result_buf = result.Into();

            if (x == kAlpha) {
                EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0)
                    << "Failed at alpha=" << x;
            } else {
                EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                    << "Failed at x=" << x;
            }
        }
    }
};

using HalfTreeDpfBytesAesSoftTest = HalfTreeDpfAesSoftTest<BytesGroup>;
using HalfTreeDpfUint128AesSoftTest = HalfTreeDpfAesSoftTest<Uint127Group>;
using HalfTreeDpfUint64AesSoftTest = HalfTreeDpfAesSoftTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesAesSoftTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesAesSoftTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesAesSoftTest, EvalAll) {
    TestEvalAll();
}

TEST_F(HalfTreeDpfUint128AesSoftTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128AesSoftTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128AesSoftTest, EvalAll) {
    TestEvalAll();
}

TEST_F(HalfTreeDpfUint64AesSoftTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64AesSoftTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64AesSoftTest, EvalAll) {
    TestEvalAll();
}

// Edge case: in_bits = 1
template <typename Group>
class HalfTreeDpfOneBitTest : public ::testing::Test {
protected:
    using Prg = fss::prg::ChaCha<1>;
    using DpfType = fss::HalfTreeDpf<1, Group, Prg, uint8_t>;

    int4 s0s[2];
    typename DpfType::Cw cws[1];
    int4 ocw;
    Prg prg;
    int4 hash_key;

    HalfTreeDpfOneBitTest() : prg(gChaChaDeviceNonces) {}

    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis;

        s0s[0] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        s0s[1] = {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
        hash_key = {dis(gen), dis(gen), dis(gen), dis(gen)};
    }

    int4 MakeBBuf() {
        return {static_cast<int>(kBeta & 0xFFFFFFFF), static_cast<int>((kBeta >> 32) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 64) & 0xFFFFFFFF),
            static_cast<int>((kBeta >> 96) & 0xFFFFFFFE)};
    }

    void TestEvalAtAlpha() {
        DpfType dpf{prg, hash_key};
        uint8_t alpha = 1;

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, alpha, b_buf);

        int4 y0 = dpf.Eval(false, s0s[0], cws, ocw, alpha);
        int4 y1 = dpf.Eval(true, s0s[1], cws, ocw, alpha);

        auto result = Group::From(y0) + Group::From(y1);
        auto expected = Group::From(b_buf);

        int4 result_buf = result.Into();
        int4 expected_buf = expected.Into();
        EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0);
    }

    void TestEvalAtNonAlpha() {
        DpfType dpf{prg, hash_key};
        uint8_t alpha = 1;

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, alpha, b_buf);

        uint8_t x = 0;
        int4 y0 = dpf.Eval(false, s0s[0], cws, ocw, x);
        int4 y1 = dpf.Eval(true, s0s[1], cws, ocw, x);

        auto result = Group::From(y0) + Group::From(y1);
        int4 result_buf = result.Into();
        int4 zero = {0, 0, 0, 0};
        EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0);
    }

    void TestEvalAll() {
        DpfType dpf{prg, hash_key};
        uint8_t alpha = 1;

        int4 b_buf = MakeBBuf();
        dpf.Gen(cws, ocw, s0s, alpha, b_buf);

        int4 ys0[2], ys1[2];
        dpf.EvalAll(false, s0s[0], cws, ocw, ys0);
        dpf.EvalAll(true, s0s[1], cws, ocw, ys1);

        auto expected = Group::From(b_buf);
        int4 expected_buf = expected.Into();
        int4 zero = {0, 0, 0, 0};

        for (int x = 0; x < 2; ++x) {
            auto result = Group::From(ys0[x]) + Group::From(ys1[x]);
            int4 result_buf = result.Into();

            if (x == alpha) {
                EXPECT_EQ(memcmp(&result_buf, &expected_buf, GroupCompareSize<Group>()), 0)
                    << "Failed at alpha=" << x;
            } else {
                EXPECT_EQ(memcmp(&result_buf, &zero, GroupCompareSize<Group>()), 0)
                    << "Failed at x=" << x;
            }
        }
    }
};

using HalfTreeDpfBytesOneBitTest = HalfTreeDpfOneBitTest<BytesGroup>;
using HalfTreeDpfUint128OneBitTest = HalfTreeDpfOneBitTest<Uint127Group>;
using HalfTreeDpfUint64OneBitTest = HalfTreeDpfOneBitTest<Uint64Group>;

TEST_F(HalfTreeDpfBytesOneBitTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfBytesOneBitTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfBytesOneBitTest, EvalAll) {
    TestEvalAll();
}

TEST_F(HalfTreeDpfUint128OneBitTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint128OneBitTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint128OneBitTest, EvalAll) {
    TestEvalAll();
}

TEST_F(HalfTreeDpfUint64OneBitTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(HalfTreeDpfUint64OneBitTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(HalfTreeDpfUint64OneBitTest, EvalAll) {
    TestEvalAll();
}
