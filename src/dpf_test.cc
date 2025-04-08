#include <random>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>
#include <gtest/gtest.h>
#include <dpf.h>

extern "C" void prg_init(const uint8_t *state, int state_len);

using random_bytes_engine = std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t>;

class DpfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::random_device rd;
    random_bytes_engine rbe(rd());
    uint8_t keys[32];
    std::generate(std::begin(keys), std::end(keys), std::ref(rbe));
    prg_init((uint8_t *)keys, 32);

    kS0s = (uint8_t *)malloc(kLambda * 2);
    assert(kS0s != NULL);
    std::generate(kS0s, kS0s + kLambda * 2, std::ref(rbe));
    uint8_t t0 = kS0s[kLambda - 1] & 1;
    uint8_t t1 = t0 ^ 1;
    kS0s[kLambda * 2 - 1] = t1 ? kS0s[kLambda * 2 - 1] | 1 : kS0s[kLambda * 2 - 1] & ~1;
  }

  void TearDown() override {
    free(kS0s);
  }

  static constexpr uint16_t kAlpha = 107;
  static constexpr uint16_t kAlphaBitlen = 16;
  static constexpr __uint128_t kBeta = 604;

  uint8_t *kS0s;
};

TEST_F(DpfTest, EvalAtAlpha) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 6);
  assert(sbuf != NULL);

  DpfKey key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare point function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  __uint128_t beta_int = kBeta;
  uint8_t *beta = (uint8_t *)&beta_int;
  PointFunc pf = {alpha_bits, beta};

  // Generate DPF
  memcpy(sbuf, kS0s, kLambda * 2);
  dpf_gen(key, pf, sbuf);

  // Evaluate at alpha (x = alpha)
  Bits x_bits = {alpha, kAlphaBitlen};

  // Party 0 eval
  memcpy(sbuf, kS0s, kLambda);
  dpf_eval(sbuf, 0, key, x_bits);
  __uint128_t y0;
  memcpy(&y0, sbuf, kLambda);

  // Party 1 eval
  memcpy(sbuf, kS0s + kLambda, kLambda);
  dpf_eval(sbuf, 1, key, x_bits);
  __uint128_t y1;
  memcpy(&y1, sbuf, kLambda);

  // Check result
  EXPECT_EQ(y0 + y1, kBeta);

  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(DpfTest, EvalAtRandPoints) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 6);
  assert(sbuf != NULL);

  DpfKey key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare point function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  __uint128_t beta_int = kBeta;
  uint8_t *beta = (uint8_t *)&beta_int;
  PointFunc pf = {alpha_bits, beta};

  // Generate DPF
  memcpy(sbuf, kS0s, kLambda * 2);
  dpf_gen(key, pf, sbuf);

  // Test at random points
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(0, UINT16_MAX);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x = dis(gen);
    Bits x_bits = {(uint8_t *)&x, 16};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dpf_eval(sbuf, 0, key, x_bits);
    __uint128_t y0;
    memcpy(&y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dpf_eval(sbuf, 1, key, x_bits);
    __uint128_t y1;
    memcpy(&y1, sbuf, kLambda);

    // Check result
    if (x == kAlpha) {
      EXPECT_EQ(y0 + y1, kBeta);
    } else {
      EXPECT_EQ(y0 + y1, 0);
    }
  }

  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}
