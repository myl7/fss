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
    EXPECT_EQ(memcmp(&l, &r, sizeof(int4)), 0);
  }
};

using GroupTypes = ::testing::Types<BytesGroup, Uint8Group, Uint8ModGroup, Uint16Group, Uint16ModGroup, Uint32Group,
    Uint32ModGroup, Uint64Group, Uint64ModGroup, Uint127Group, Uint127ModGroup>;
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

// Independently encode unsigned words as a clamped int4 for boundary checks.
static int4 BoundaryBlock(uint32_t x, uint32_t y, uint32_t z, uint32_t top) {
  return {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z),
      static_cast<int>(top << 1)};
}

static void ExpectBoundaryBlock(const int4 &actual, const int4 &expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

template <typename Group>
static void ExpectCanonicalBlock(const int4 &block, const int4 &expected) {
  ExpectBoundaryBlock(Group::From(block).Into(), expected);
}

template <typename Group>
static void ExpectOneWrapsToZero(const int4 &canonical_max_block) {
  const Group value = Group::From(canonical_max_block);
  const Group wrapped = value + Group::From(BoundaryBlock(1, 0, 0, 0));
  ExpectBoundaryBlock(wrapped.Into(), BoundaryBlock(0, 0, 0, 0));
}

TEST(GroupBoundaryBehavior, UnsignedCanonicalMaxEncodesAndWraps) {
  // Uint32 max, then the modular variant reducing the same all-ones block.
  ExpectCanonicalBlock<Uint32Group>(BoundaryBlock(0xffffffffU, 0, 0, 0),
      BoundaryBlock(0xffffffffU, 0, 0, 0));
  ExpectOneWrapsToZero<Uint32Group>(BoundaryBlock(0xffffffffU, 0, 0, 0));
  ExpectCanonicalBlock<Uint32ModGroup>(BoundaryBlock(0xffffffffU, 0, 0, 0),
      BoundaryBlock(0x7ffffff4U, 0, 0, 0));
  ExpectOneWrapsToZero<Uint32ModGroup>(BoundaryBlock(2147483658U, 0, 0, 0));

  // Uint64 max, then the modular variant reducing the same all-ones block.
  ExpectCanonicalBlock<Uint64Group>(BoundaryBlock(0xffffffffU, 0xffffffffU, 0, 0),
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0, 0));
  ExpectOneWrapsToZero<Uint64Group>(BoundaryBlock(0xffffffffU, 0xffffffffU, 0, 0));
  ExpectCanonicalBlock<Uint64ModGroup>(BoundaryBlock(0xffffffffU, 0xffffffffU, 0, 0),
      BoundaryBlock(0xffffffe2U, 0x7fffffffU, 0, 0));
  ExpectOneWrapsToZero<Uint64ModGroup>(BoundaryBlock(0x0000001cU, 0x80000000U, 0, 0));

  // Uint127 max, then the modular variant reducing the same 127 all-ones bits.
  ExpectCanonicalBlock<Uint127Group>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU),
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU));
  ExpectOneWrapsToZero<Uint127Group>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU));
  ExpectCanonicalBlock<Uint127ModGroup>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU),
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x3fffffffU));
  ExpectOneWrapsToZero<Uint127ModGroup>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x3fffffffU));
}
