#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>
#include <gtest/gtest.h>
#include <fss/dcf.h>

extern "C" void prg_init(const uint8_t *state, int state_len);

using random_bytes_engine = std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t>;

class DcfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::random_device rd;
    random_bytes_engine rbe(rd());
    uint8_t keys[4 * kLambda];
    std::generate(std::begin(keys), std::end(keys), std::ref(rbe));
    prg_init((uint8_t *)keys, 4 * kLambda);

    kS0s = (uint8_t *)malloc(kLambda * 2);
    assert(kS0s != NULL);
    std::generate(kS0s, kS0s + kLambda * 2, std::ref(rbe));
  }

  void TearDown() override {
    prg_free();
    free(kS0s);
  }

  static constexpr uint16_t kAlpha = 107;
  static constexpr int kAlphaBitlen = 16;
  static constexpr __uint128_t kBeta = 604;

  uint8_t *kS0s;
};

TEST_F(DcfTest, EvalAtRandLtPoints) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare comparison function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  CmpFunc cf = {p, kLtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key, cf, sbuf);

  // Test at random points less than alpha
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(0, kAlpha - 1);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x = dis(gen);
    Bits x_bits = {(uint8_t *)&x, kAlphaBitlen};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dcf_eval(sbuf, 0, key, x_bits);
    uint8_t y0[kLambda];
    memcpy(y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dcf_eval(sbuf, 1, key, x_bits);
    uint8_t y1[kLambda];
    memcpy(y1, sbuf, kLambda);

    group_add(y0, y1);

    // Check result
    EXPECT_EQ(memcmp(y0, beta, kLambda), 0) << "Result differ at x = " << x;
  }

  free(beta);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(DcfTest, EvalAtRandGePoints) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare comparison function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  CmpFunc cf = {p, kLtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key, cf, sbuf);

  // Test at random points greater than or equal to alpha
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(kAlpha + 1, UINT16_MAX);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x = i == 0 ? kAlpha : dis(gen);
    Bits x_bits = {(uint8_t *)&x, kAlphaBitlen};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dcf_eval(sbuf, 0, key, x_bits);
    uint8_t y0[kLambda];
    memcpy(y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dcf_eval(sbuf, 1, key, x_bits);
    uint8_t y1[kLambda];
    memcpy(y1, sbuf, kLambda);

    group_add(y0, y1);

    // Check result
    uint8_t zero[kLambda];
    memset(zero, 0, kLambda);
    EXPECT_EQ(memcmp(y0, zero, kLambda), 0) << "Result differ at x = " << x;
  }

  free(beta);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(DcfTest, EvalAtRandGtPointsForGtAlpha) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare comparison function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  CmpFunc cf = {p, kGtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key, cf, sbuf);

  // Test at random points strictly greater than alpha
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(kAlpha + 1, UINT16_MAX);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x = dis(gen);
    Bits x_bits = {(uint8_t *)&x, kAlphaBitlen};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dcf_eval(sbuf, 0, key, x_bits);
    uint8_t y0[kLambda];
    memcpy(y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dcf_eval(sbuf, 1, key, x_bits);
    uint8_t y1[kLambda];
    memcpy(y1, sbuf, kLambda);

    group_add(y0, y1);

    // Check result
    EXPECT_EQ(memcmp(y0, beta, kLambda), 0) << "Result differ at x = " << x;
  }

  free(beta);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(DcfTest, EvalAtRandLePointsForGtAlpha) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare comparison function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  CmpFunc cf = {p, kGtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key, cf, sbuf);

  // Test at random points less than or equal to alpha
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dis(0, kAlpha);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x = i == 0 ? kAlpha : dis(gen);
    Bits x_bits = {(uint8_t *)&x, kAlphaBitlen};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dcf_eval(sbuf, 0, key, x_bits);
    uint8_t y0[kLambda];
    memcpy(y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dcf_eval(sbuf, 1, key, x_bits);
    uint8_t y1[kLambda];
    memcpy(y1, sbuf, kLambda);

    group_add(y0, y1);

    // Check result
    uint8_t zero[kLambda];
    memset(zero, 0, kLambda);
    EXPECT_EQ(memcmp(y0, zero, kLambda), 0) << "Result differ at x = " << x;
  }

  free(beta);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(DcfTest, EvalFullDomainEqEvalPoints) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare comparison function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  CmpFunc cf = {p, kLtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key, cf, sbuf);

  // Allocate buffers for full evaluation
  uint8_t *ys0_full = (uint8_t *)malloc(kLambda * (1 << kAlphaBitlen));
  uint8_t *ys1_full = (uint8_t *)malloc(kLambda * (1 << kAlphaBitlen));
  assert(ys0_full != NULL && ys1_full != NULL);

  // Party 0 full eval
  memcpy(ys0_full, kS0s, kLambda);
  dcf_eval_full_domain(ys0_full, 0, key, kAlphaBitlen);

  // Party 1 full eval
  memcpy(ys1_full, kS0s + kLambda, kLambda);
  dcf_eval_full_domain(ys1_full, 1, key, kAlphaBitlen);

  // Compare with point-by-point evaluation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> lt_dis(0, kAlpha - 1);
  std::uniform_int_distribution<uint16_t> gt_dis(kAlpha + 1, UINT16_MAX);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x;
    if (i == 0) {
      // Test x = alpha
      x = kAlpha;
    } else if (i < kNumTrials / 2) {
      // Test x < alpha
      x = lt_dis(gen);
    } else {
      // Test x > alpha
      x = gt_dis(gen);
    }
    // Point-by-point evaluation
    Bits x_bits = {(uint8_t *)&x, kAlphaBitlen};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dcf_eval(sbuf, 0, key, x_bits);
    uint8_t y0[kLambda];
    memcpy(y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dcf_eval(sbuf, 1, key, x_bits);
    uint8_t y1[kLambda];
    memcpy(y1, sbuf, kLambda);

    // Get full domain evaluation results
    uint8_t *y0_full = ys0_full + (int)x * kLambda;
    uint8_t *y1_full = ys1_full + (int)x * kLambda;

    // Compare party 0 shares
    EXPECT_EQ(memcmp(y0, y0_full, kLambda), 0) << "Party 0 shares differ at x = " << x;

    // Compare party 1 shares
    EXPECT_EQ(memcmp(y1, y1_full, kLambda), 0) << "Party 1 shares differ at x = " << x;
  }

  free(beta);
  free(ys0_full);
  free(ys1_full);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(DcfTest, EvalFullDomainEqEvalPointsForGtAlpha) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare comparison function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  CmpFunc cf = {p, kGtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key, cf, sbuf);

  // Allocate buffers for full evaluation
  uint8_t *ys0_full = (uint8_t *)malloc(kLambda * (1 << kAlphaBitlen));
  uint8_t *ys1_full = (uint8_t *)malloc(kLambda * (1 << kAlphaBitlen));
  assert(ys0_full != NULL && ys1_full != NULL);

  // Party 0 full eval
  memcpy(ys0_full, kS0s, kLambda);
  dcf_eval_full_domain(ys0_full, 0, key, kAlphaBitlen);

  // Party 1 full eval
  memcpy(ys1_full, kS0s + kLambda, kLambda);
  dcf_eval_full_domain(ys1_full, 1, key, kAlphaBitlen);

  // Compare with point-by-point evaluation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> lt_dis(0, kAlpha - 1);
  std::uniform_int_distribution<uint16_t> gt_dis(kAlpha + 1, UINT16_MAX);

  constexpr int kNumTrials = 100;

  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x;
    if (i == 0) {
      // Test x = alpha
      x = kAlpha;
    } else if (i < kNumTrials / 2) {
      // Test x < alpha
      x = lt_dis(gen);
    } else {
      // Test x > alpha
      x = gt_dis(gen);
    }
    // Point-by-point evaluation
    Bits x_bits = {(uint8_t *)&x, kAlphaBitlen};

    // Party 0 eval
    memcpy(sbuf, kS0s, kLambda);
    dcf_eval(sbuf, 0, key, x_bits);
    uint8_t y0[kLambda];
    memcpy(y0, sbuf, kLambda);

    // Party 1 eval
    memcpy(sbuf, kS0s + kLambda, kLambda);
    dcf_eval(sbuf, 1, key, x_bits);
    uint8_t y1[kLambda];
    memcpy(y1, sbuf, kLambda);

    // Get full domain evaluation results
    uint8_t *y0_full = ys0_full + (int)x * kLambda;
    uint8_t *y1_full = ys1_full + (int)x * kLambda;

    // Compare party 0 shares
    EXPECT_EQ(memcmp(y0, y0_full, kLambda), 0) << "Party 0 shares differ at x = " << x;

    // Compare party 1 shares
    EXPECT_EQ(memcmp(y1, y1_full, kLambda), 0) << "Party 1 shares differ at x = " << x;
  }

  free(beta);
  free(ys0_full);
  free(ys1_full);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}
