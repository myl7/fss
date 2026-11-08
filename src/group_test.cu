#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <cstring>
#include <fss/group/bytes.cuh>
#include <fss/group/uint.cuh>
#include <fss/group/uint128.cuh>

using BytesGroup = fss::group::Bytes;
using Uint8Group = fss::group::Uint<uint8_t>;
using Uint8ModGroup = fss::group::Uint<uint8_t, 131>;
using Uint16Group = fss::group::Uint<uint16_t>;
using Uint16ModGroup = fss::group::Uint<uint16_t, 32771>;
using Uint32Group = fss::group::Uint<uint32_t>;
using Uint32ModGroup = fss::group::Uint<uint32_t, 2147483659U>;
using Uint64Group = fss::group::Uint<uint64_t>;
using Uint64ModGroup = fss::group::Uint<uint64_t, 9223372036854775837ULL>;
using Uint128Group = fss::group::Uint128<>;
using Uint128ModGroup =
    fss::group::Uint128<fss::group::Uint128Modulus{.low = 0, .high = uint64_t{1} << 62}>;

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
    Uint32ModGroup, Uint64Group, Uint64ModGroup, Uint128Group, Uint128ModGroup>;
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

  // The 127-bit clamp maximum, then the modular variant reducing the same 127 all-ones bits.
  ExpectCanonicalBlock<Uint128Group>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU),
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU));
  ExpectOneWrapsToZero<Uint128Group>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU));
  ExpectCanonicalBlock<Uint128ModGroup>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU),
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x3fffffffU));
  ExpectOneWrapsToZero<Uint128ModGroup>(
      BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x3fffffffU));
}

TEST(GroupPreconditionTest, BytesFromAcceptsClampedBlock) {
  int4 clamped = {1, 2, 3, 4};
  EXPECT_NO_THROW((void)BytesGroup::From(clamped));
}

// The storage types Uint keeps supporting, with wrap and modulo modes.
static_assert(Groupable<fss::group::Uint<uint8_t>>);
static_assert(Groupable<fss::group::Uint<uint16_t, 251>>);
static_assert(Groupable<fss::group::Uint<uint32_t>>);
static_assert(Groupable<fss::group::Uint<uint64_t, 9223372036854775783ULL>>);
// __uint128_t storage and out-of-range Uint128 moduli are rejected at compile
// time; uint_reject_compile_fail.cu covers those failures per case because a
// requires-expression naming the type is a hard error under the nvcc frontend.
// The default modulus is 2^127, matching the previous common127 encoding.
static_assert(std::is_same_v<Uint128Group, fss::group::Uint128<>>);
static_assert(std::is_same_v<Uint128Group, fss::group::Uint128<fss::group::Uint128Modulus{.low = 0, .high = uint64_t{1} << 63}>>);

// Encode an unsigned 128-bit value below 2^127 as a clamped block.
static int4 ClampedBlockOf(__uint128_t value) {
  return BoundaryBlock(static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32),
      static_cast<uint32_t>(value >> 64), static_cast<uint32_t>(value >> 96));
}

static void ExpectUint128Value(const char *what, __uint128_t actual, __uint128_t expected) {
  EXPECT_TRUE(actual == expected) << what << ": actual " << static_cast<unsigned long long>(actual >> 64)
      << ":" << static_cast<unsigned long long>(actual & 0xffffffffffffffffULL) << ", expected "
      << static_cast<unsigned long long>(expected >> 64) << ":"
      << static_cast<unsigned long long>(expected & 0xffffffffffffffffULL);
}

TEST(Uint128Behavior, DefaultModulusMatchesCommon127) {
  // Elements below 2^127 pass From() unreduced and Into() re-encodes them.
  const __uint128_t max127 = (static_cast<__uint128_t>(1) << 127) - 1;
  ExpectBoundaryBlock(Uint128Group::From(ClampedBlockOf(max127)).Into(), ClampedBlockOf(max127));
  ExpectUint128Value("max127 value", Uint128Group::From(ClampedBlockOf(max127)).Value(), max127);
  // The clamp bit of buf.w is dropped even when the upper words are set.
  ExpectBoundaryBlock(Uint128Group::From(BoundaryBlock(0xffffffffU, 0xffffffffU, 0xffffffffU, 0x7fffffffU)).Into(),
      ClampedBlockOf(max127));
  // max127 + 1 wraps to zero, and the additive inverse of the maximum is one.
  ExpectBoundaryBlock((Uint128Group::From(ClampedBlockOf(max127)) + Uint128Group::From(BoundaryBlock(1, 0, 0, 0))).Into(),
      ClampedBlockOf(0));
  ExpectBoundaryBlock((-Uint128Group::From(ClampedBlockOf(max127))).Into(), ClampedBlockOf(1));
  // A sum crossing the low/high 64-bit halves carries without wrapping.
  const __uint128_t low_max = 0xffffffffffffffffULL;
  ExpectBoundaryBlock(
      (Uint128Group::From(ClampedBlockOf(low_max)) + Uint128Group::From(ClampedBlockOf(2))).Into(),
      ClampedBlockOf(low_max + 2));
}

TEST(Uint128Behavior, ModulusHalvesCombineInOrder) {
  // mod = (1 << 64) + 1 from low = 1 and high = 1.
  using Mod2164plus1 = fss::group::Uint128<fss::group::Uint128Modulus{.low = 1, .high = 1}>;
  // 2^64 stays below the modulus, 2^64 + 1 is the modulus itself, so the sum wraps to zero.
  const auto near = Mod2164plus1::From(ClampedBlockOf(static_cast<__uint128_t>(1) << 64));
  const auto one = Mod2164plus1::From(BoundaryBlock(1, 0, 0, 0));
  ExpectBoundaryBlock((near + one).Into(), ClampedBlockOf(0));
  // The additive inverse of 2^64 is 1, because 2^64 + 1 = mod.
  ExpectBoundaryBlock((-near).Into(), ClampedBlockOf(1));
  // From() reduces values at or above the modulus: mod itself maps to zero.
  const __uint128_t mod = (static_cast<__uint128_t>(1) << 64) + 1;
  ExpectBoundaryBlock(Mod2164plus1::From(ClampedBlockOf(mod)).Into(), ClampedBlockOf(0));
  ExpectBoundaryBlock(Mod2164plus1::From(ClampedBlockOf(mod + 5)).Into(), ClampedBlockOf(5));
  // A modulus with both halves large keeps From() and addition consistent.
  using ModWide =
      fss::group::Uint128<fss::group::Uint128Modulus{.low = 0xfedcba9876543210ULL, .high = 0x1234567890abcdefULL}>;
  const __uint128_t wide_mod = (static_cast<__uint128_t>(0x1234567890abcdefULL) << 64) | 0xfedcba9876543210ULL;
  const __uint128_t wide_max = (static_cast<__uint128_t>(1) << 127) - 1;
  ExpectBoundaryBlock(ModWide::From(ClampedBlockOf(wide_max)).Into(), ClampedBlockOf(wide_max % wide_mod));
  ExpectBoundaryBlock(ModWide::From(ClampedBlockOf(wide_mod - 1)).Into(), ClampedBlockOf(wide_mod - 1));
  ExpectBoundaryBlock(
      (ModWide::From(ClampedBlockOf(wide_mod - 1)) + ModWide::From(BoundaryBlock(1, 0, 0, 0))).Into(),
      ClampedBlockOf(0));
}

#if FSS_ENABLE_V1_COMPAT
// The deprecated v1.3.0 wide-storage spellings, restored only for uint64-range
// positive moduli as forwardings to Uint128. Aliases keep this suite's build
// log readable; the direct spellings and their deprecation diagnostic are
// covered by uint_compat_compile_check.cu, and the rejected moduli by
// uint_reject_compile_fail.cu.
using LegacyUint128Small = fss::group::Uint<__uint128_t, 17>;
using LegacyUint128Cross = fss::group::Uint<__uint128_t, (uint64_t{1} << 32) + 15>;
using LegacyUint128Pow63m1 = fss::group::Uint<__uint128_t, (uint64_t{1} << 63) - 1>;
using LegacyUint128Pow63 = fss::group::Uint<__uint128_t, uint64_t{1} << 63>;
using LegacyUint128MaxM58 = fss::group::Uint<__uint128_t, UINT64_MAX - 58>;
using LegacyUint128U64Max = fss::group::Uint<__uint128_t, UINT64_MAX>;

static_assert(sizeof(LegacyUint128Small) == 16 && sizeof(LegacyUint128Cross) == 16 &&
    sizeof(LegacyUint128Pow63m1) == 16 && sizeof(LegacyUint128Pow63) == 16 &&
    sizeof(LegacyUint128MaxM58) == 16 && sizeof(LegacyUint128U64Max) == 16);
static_assert(Groupable<LegacyUint128Small> && Groupable<LegacyUint128Cross> && Groupable<LegacyUint128Pow63m1>);
static_assert(Groupable<LegacyUint128Pow63> && Groupable<LegacyUint128MaxM58> && Groupable<LegacyUint128U64Max>);
// The operators and conversions keep the legacy group and clamped-block types.
static_assert(std::is_same_v<LegacyUint128Small, decltype(LegacyUint128Small{} + LegacyUint128Small{})>);
static_assert(std::is_same_v<LegacyUint128Small, decltype(-LegacyUint128Small{})>);
static_assert(std::is_same_v<LegacyUint128Small, decltype(LegacyUint128Small::From(int4{}))>);
static_assert(std::is_same_v<int4, decltype(LegacyUint128Small{}.Into())>);
static_assert(std::is_same_v<__uint128_t, decltype(LegacyUint128Small{}.Value())>);

// One legacy spelling and its Uint128 forwarding target must agree on every
// operation for the same inputs.
template <__uint128_t mod>
static void ExpectLegacyMatchesModernOnBlock(const int4 &block) {
  using Legacy = fss::group::Uint<__uint128_t, mod>;
  using Modern = fss::group::Uint128<fss::group::Uint128Modulus{static_cast<uint64_t>(mod), 0}>;
  const Legacy legacy = Legacy::From(block);
  const Modern modern = Modern::From(block);
  ExpectUint128Value("From value", legacy.Value(), modern.Value());
  ExpectBoundaryBlock(legacy.Into(), modern.Into());
  const Legacy legacy_neg = -legacy;
  const Modern modern_neg = -modern;
  ExpectBoundaryBlock(legacy_neg.Into(), modern_neg.Into());
}

TEST(UintCompatBehavior, ForwardsToUint128WithSameModulus) {
  const __uint128_t max127 = (static_cast<__uint128_t>(1) << 127) - 1;
  ExpectLegacyMatchesModernOnBlock<17>(ClampedBlockOf(max127));
  ExpectLegacyMatchesModernOnBlock<17>(ClampedBlockOf(5));
  ExpectLegacyMatchesModernOnBlock<(uint64_t{1} << 32) + 15>(ClampedBlockOf(max127));
  ExpectLegacyMatchesModernOnBlock<(uint64_t{1} << 63) - 1>(ClampedBlockOf(max127));
  ExpectLegacyMatchesModernOnBlock<uint64_t{1} << 63>(ClampedBlockOf(max127));
  ExpectLegacyMatchesModernOnBlock<UINT64_MAX - 58>(ClampedBlockOf(max127));
  ExpectLegacyMatchesModernOnBlock<UINT64_MAX>(ClampedBlockOf(max127));
}

TEST(UintCompatBehavior, ModulusBoundariesReduceWrapAndNegate) {
  // From() reduces the modulus itself and values just above it.
  ExpectBoundaryBlock(LegacyUint128Small::From(ClampedBlockOf(17)).Into(), ClampedBlockOf(0));
  ExpectBoundaryBlock(LegacyUint128Small::From(ClampedBlockOf(22)).Into(), ClampedBlockOf(5));
  // A 127-bit input reduces through the full modulo path of the forwarding target.
  const __uint128_t max127 = (static_cast<__uint128_t>(1) << 127) - 1;
  ExpectUint128Value("max127 mod 17", LegacyUint128Small::From(ClampedBlockOf(max127)).Value(), max127 % 17);
  // mod - 1 plus one wraps to zero at the low and high edges of the restored range.
  ExpectBoundaryBlock(
      (LegacyUint128Pow63::From(ClampedBlockOf((uint64_t{1} << 63) - 1)) +
          LegacyUint128Pow63::From(BoundaryBlock(1, 0, 0, 0))).Into(),
      ClampedBlockOf(0));
  ExpectBoundaryBlock(
      (LegacyUint128U64Max::From(ClampedBlockOf(UINT64_MAX - 1)) +
          LegacyUint128U64Max::From(BoundaryBlock(1, 0, 0, 0))).Into(),
      ClampedBlockOf(0));
  // The additive inverse of 5 is mod - 5, and zero inverts to zero.
  ExpectUint128Value("negate 5 mod 17", (-LegacyUint128Small::From(ClampedBlockOf(5))).Value(), 12);
  ExpectUint128Value("negate 0", (-LegacyUint128Small::From(ClampedBlockOf(0))).Value(), 0);
  // A carry across the low 32-bit words stays inside the low 64 bits.
  ExpectBoundaryBlock(
      (LegacyUint128U64Max::From(BoundaryBlock(0xfffffffeU, 0, 0, 0)) +
          LegacyUint128U64Max::From(BoundaryBlock(3, 0, 0, 0))).Into(),
      ClampedBlockOf(static_cast<__uint128_t>(0xffffffffU) + 2));
}
#endif  // FSS_ENABLE_V1_COMPAT

#ifndef NDEBUG
// The zero LSB is a caller precondition of the shared leaf conversion: Debug
// builds abort with a diagnostic, and Release builds do not check it.
TEST(GroupPreconditionTest, BytesFromAbortsOnNonzeroLsbInDebug) {
  int4 odd = {1, 2, 3, 5};
  EXPECT_DEATH((void)BytesGroup::From(odd), "Assertion.*failed");
}
#endif  // NDEBUG
