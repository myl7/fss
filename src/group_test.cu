#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>

using BytesGroup = fss::group::Bytes;
using Uint8Group = fss::group::Uint<uint8_t>;
using Uint8ModGroup = fss::group::Uint<uint8_t, 131>;
using Uint16Group = fss::group::Uint<uint16_t>;
using Uint16ModGroup = fss::group::Uint<uint16_t, 32771>;
using Uint32Group = fss::group::Uint<uint32_t>;
using Uint32ModGroup = fss::group::Uint<uint32_t, 2147483659U>;
using Uint64Group = fss::group::Uint<uint64_t>;
using Uint64ModGroup = fss::group::Uint<uint64_t, 9223372036854775837ULL>;
using Uint127Group = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 127)>;
using Uint127ModGroup = fss::group::Uint<__uint128_t, (static_cast<__uint128_t>(1) << 126)>;

// Helper to generate a random clamped int4 (LSB of .w cleared).
static int4 RandomClampedInt4(std::mt19937 &gen) {
    std::uniform_int_distribution<int> dis;
    return {dis(gen), dis(gen), dis(gen), dis(gen) & ~1};
}

// Helper to determine comparison size for each group type.
// Uint groups only use the low sizeof(T) bytes; Bytes uses the full 16B.
template <typename Group>
constexpr size_t GroupCompareSize() {
    return sizeof(int4);
}
template <typename T, T mod>
constexpr size_t GroupCompareSize<fss::group::Uint<T, mod>>() {
    if constexpr (sizeof(T) == 16) return sizeof(int4);
    else return sizeof(T);
}

template <typename Group>
class GroupAxiomTest : public ::testing::Test {
protected:
    static constexpr int kTrials = 100;
    std::mt19937 gen;

    void SetUp() override {
        std::random_device rd;
        gen.seed(rd());
    }

    Group RandomElement() {
        return Group::From(RandomClampedInt4(gen));
    }

    void ExpectEqual(const Group &lhs, const Group &rhs) {
        int4 l = lhs.Into();
        int4 r = rhs.Into();
        EXPECT_EQ(memcmp(&l, &r, GroupCompareSize<Group>()), 0);
    }
};

using GroupTypes =
    ::testing::Types<BytesGroup, Uint8Group, Uint8ModGroup, Uint16Group, Uint16ModGroup,
        Uint32Group, Uint32ModGroup, Uint64Group, Uint64ModGroup, Uint127Group, Uint127ModGroup>;
TYPED_TEST_SUITE(GroupAxiomTest, GroupTypes);

// (a + b) + c == a + (b + c)
TYPED_TEST(GroupAxiomTest, Associativity) {
    for (int i = 0; i < this->kTrials; i++) {
        auto a = this->RandomElement();
        auto b = this->RandomElement();
        auto c = this->RandomElement();
        this->ExpectEqual((a + b) + c, a + (b + c));
    }
}

// a + 0 == a and 0 + a == a
TYPED_TEST(GroupAxiomTest, Identity) {
    TypeParam zero;
    for (int i = 0; i < this->kTrials; i++) {
        auto a = this->RandomElement();
        this->ExpectEqual(a + zero, a);
        this->ExpectEqual(zero + a, a);
    }
}

// a + (-a) == 0 and (-a) + a == 0
TYPED_TEST(GroupAxiomTest, Inverses) {
    TypeParam zero;
    for (int i = 0; i < this->kTrials; i++) {
        auto a = this->RandomElement();
        this->ExpectEqual(a + (-a), zero);
        this->ExpectEqual((-a) + a, zero);
    }
}
