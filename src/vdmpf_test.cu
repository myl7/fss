#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fss/vdmpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/prp/aes128_feistel.cuh>

using BytesGroup = fss::group::Bytes;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;

static int gChaChaDeviceNonces[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

template <typename Group>
class VdmpfChaChaTest : public ::testing::Test {
protected:
    static constexpr int kInBits = 16;
    static constexpr int kMaxPoints = 30;
    static constexpr int kBucketBits = 14;
    static constexpr int kT = 30;

    using Prg = fss::prg::ChaCha<2>;
    using XorHash = fss::hash::Blake3;
    using Hash = fss::hash::Blake3;
    using Prp = fss::prp::Aes128Feistel;
    using VdmpfType =
        fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, Group, Prg, XorHash, Hash, Prp, uint16_t>;

    static constexpr int m = VdmpfType::m;

    uint16_t alphas[kT] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
        1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
        2900, 3000};
    int4 betas[kT];

    Prg prg;
    int4 hash_iv[2] = {{0x11111111, 0x22222222, 0x33333333, 0x44444444},
        {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888888u)}};
    XorHash xor_hash;
    Hash hash_;
    Prp prp_;

    VdmpfChaChaTest()
        : prg(gChaChaDeviceNonces),
          xor_hash(cuda::std::span<const int4, 2>(hash_iv, 2)),
          hash_(cuda::std::span<const int4, 2>(hash_iv, 2)) {}

    void SetUp() override {
        for (int i = 0; i < kT; ++i) {
            betas[i] = {(i + 1) * 11, 0, 0, 0};
        }
    }

    int4 RandomSeed(std::mt19937 &gen) {
        std::uniform_int_distribution<int> dis;
        return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
    }

    void GenKeys(typename VdmpfType::Key &k0, typename VdmpfType::Key &k1) {
        VdmpfType vdmpf{prg, xor_hash, hash_, prp_};
        std::random_device rd;
        std::mt19937 gen(rd());

        int ret;
        do {
            int4 sigma = RandomSeed(gen);
            cuda::std::array<cuda::std::array<int4, 2>, m> s0s;
            for (int i = 0; i < m; ++i) {
                s0s[i] = {RandomSeed(gen), RandomSeed(gen)};
            }
            ret = vdmpf.Gen(k0, k1, sigma, cuda::std::span<const cuda::std::array<int4, 2>, m>(s0s),
                std::span<const uint16_t>(alphas, kT), std::span<const int4>(betas, kT), kT);
        } while (ret != 0);
    }

    void TestEvalAtAlpha() {
        typename VdmpfType::Key k0, k1;
        GenKeys(k0, k1);
        VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

        std::vector<uint16_t> xs(alphas, alphas + kT);
        std::vector<int4> ys0(kT), ys1(kT);
        cuda::std::array<int4, 4> pi0, pi1;

        vdmpf.BatchEval(false, k0, std::span<const uint16_t>(xs), std::span<int4>(ys0), pi0);
        vdmpf.BatchEval(true, k1, std::span<const uint16_t>(xs), std::span<int4>(ys1), pi1);

        for (int i = 0; i < kT; ++i) {
            auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
            int4 r = result.Into();
            int4 e = betas[i];
            e.w &= ~1;
            EXPECT_EQ(memcmp(&r, &e, sizeof(int4)), 0) << "Failed at alpha=" << alphas[i];
        }
    }

    void TestEvalAtNonAlpha() {
        typename VdmpfType::Key k0, k1;
        GenKeys(k0, k1);
        VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

        constexpr int kNumTrials = 100;
        std::vector<uint16_t> xs;
        xs.reserve(kNumTrials);
        for (int i = 0; i < kNumTrials; ++i) {
            uint16_t x = dis(gen);
            bool is_alpha = false;
            for (int j = 0; j < kT; ++j) {
                if (x == alphas[j]) {
                    is_alpha = true;
                    break;
                }
            }
            if (is_alpha) {
                --i;
                continue;
            }
            xs.push_back(x);
        }

        std::vector<int4> ys0(xs.size()), ys1(xs.size());
        cuda::std::array<int4, 4> pi0, pi1;

        vdmpf.BatchEval(false, k0, std::span<const uint16_t>(xs), std::span<int4>(ys0), pi0);
        vdmpf.BatchEval(true, k1, std::span<const uint16_t>(xs), std::span<int4>(ys1), pi1);

        int4 zero = {0, 0, 0, 0};
        for (size_t i = 0; i < xs.size(); ++i) {
            auto result = Group::From(ys0[i]) + Group::From(ys1[i]);
            int4 r = result.Into();
            EXPECT_EQ(memcmp(&r, &zero, sizeof(int4)), 0) << "Failed at x=" << xs[i];
        }
    }

    void TestVerifyBatchEval() {
        typename VdmpfType::Key k0, k1;
        GenKeys(k0, k1);
        VdmpfType vdmpf{prg, xor_hash, hash_, prp_};

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);

        // Mix alpha and non-alpha points.
        std::vector<uint16_t> xs(alphas, alphas + kT);
        for (int i = 0; i < 50; ++i) {
            xs.push_back(dis(gen));
        }

        std::vector<int4> ys0(xs.size()), ys1(xs.size());
        cuda::std::array<int4, 4> pi0, pi1;

        vdmpf.BatchEval(false, k0, std::span<const uint16_t>(xs), std::span<int4>(ys0), pi0);
        vdmpf.BatchEval(true, k1, std::span<const uint16_t>(xs), std::span<int4>(ys1), pi1);

        EXPECT_TRUE(VdmpfType::Verify(
            cuda::std::span<const int4, 4>(pi0), cuda::std::span<const int4, 4>(pi1)));
    }
};

using VdmpfBytesChaChaTest = VdmpfChaChaTest<BytesGroup>;
using VdmpfUint128ChaChaTest = VdmpfChaChaTest<Uint127Group>;

TEST_F(VdmpfBytesChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(VdmpfBytesChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(VdmpfBytesChaChaTest, VerifyBatchEval) {
    TestVerifyBatchEval();
}

TEST_F(VdmpfUint128ChaChaTest, EvalAtAlpha) {
    TestEvalAtAlpha();
}
TEST_F(VdmpfUint128ChaChaTest, EvalAtNonAlpha) {
    TestEvalAtNonAlpha();
}
TEST_F(VdmpfUint128ChaChaTest, VerifyBatchEval) {
    TestVerifyBatchEval();
}

TEST(CuckooHashTest, Compact) {
    fss::prp::Aes128Feistel prp;
    constexpr int t = 30;
    uint16_t as[t];
    for (int i = 0; i < t; ++i) as[i] = static_cast<uint16_t>(i * 100 + 10);
    int m = fss::cuckoo_hash::ChBucket(t, 80);
    std::vector<std::pair<int, int>> table(m, {-1, -1});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;
    int4 sigma;
    int ret;
    constexpr __uint128_t n = __uint128_t{1} << 16;
    int B = static_cast<int>((n * 3 + m - 1) / m);
    do {
        sigma = {dis(gen), dis(gen), dis(gen), dis(gen)};
        std::fill(table.begin(), table.end(), std::pair<int, int>{-1, -1});
        fss::cuckoo_hash::Compact<fss::prp::Aes128Feistel, uint16_t> compact{prp};
        ret = compact.Run(std::span<const uint16_t>(as, t), m, sigma, n, B, 1000,
            std::span<std::pair<int, int>>(table));
    } while (ret != 0);

    // Each alpha should appear exactly once in the table.
    int count = 0;
    for (int i = 0; i < m; ++i) {
        if (table[i].first != -1) count++;
    }
    EXPECT_EQ(count, t);
}
