#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>
#include <gtest/gtest.h>
#include <fss/cw_mac_bytes.h>
#include <sodium.h>
#include <fss/dpf.h>
#include <fss/dcf.h>

extern "C" void prg_init(const uint8_t *state, int state_len);

using random_bytes_engine = std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t>;

class CwMacBytesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (sodium_init() < 0) {
      throw std::runtime_error("Failed to initialize libsodium");
    }

    std::random_device rd;
    random_bytes_engine rbe(rd());
    uint8_t keys[4 * kLambda];
    std::generate(std::begin(keys), std::end(keys), std::ref(rbe));
    prg_init((uint8_t *)keys, 4 * kLambda);

    kS0s = (uint8_t *)malloc(kLambda * 2);
    assert(kS0s != NULL);
    std::generate(kS0s, kS0s + kLambda * 2, std::ref(rbe));

    // Initialize keys with sodium random scalar
    kWkeys = (uint8_t *)malloc(kCwMacWkeyLen * (1ULL << kAlphaBitlen));
    assert(kWkeys != NULL);
    for (size_t i = 0; i < (1ULL << kAlphaBitlen); i++) {
      gen_wkey(kWkeys + i * kCwMacWkeyLen);
    }

    // Generate public keys (g^key)
    kPubWkeys = (uint8_t *)malloc(kCwMacPubWkeyLen * (1ULL << kAlphaBitlen));
    assert(kPubWkeys != NULL);
    for (size_t i = 0; i < (1ULL << kAlphaBitlen); i++) {
      gen_pub_wkey(kPubWkeys + i * kCwMacPubWkeyLen, kWkeys + i * kCwMacWkeyLen);
    }
  }

  void TearDown() override {
    prg_free();
    free(kWkeys);
    free(kPubWkeys);
    free(kS0s);
  }

  static constexpr uint8_t kAlpha = 107;
  static constexpr int kAlphaBitlen = 8;
  static constexpr __uint128_t kBeta = 6;
  static constexpr uint8_t kAlphaL = kAlpha - 10;
  static constexpr uint8_t kAlphaR = kAlpha + 10;

  uint8_t *kWkeys;
  uint8_t *kPubWkeys;
  uint8_t *kS0s;
};

TEST_F(CwMacBytesTest, VerifyDpf) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 6);
  assert(sbuf != NULL);

  Key key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDpfCwLen * kAlphaBitlen);
  assert(key.cws != NULL);

  // Prepare point function
  uint8_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p = {alpha_bits, beta};
  PointFunc pf = {p};

  // Generate DPF
  memcpy(sbuf, kS0s, kLambda * 2);
  dpf_gen(key, pf, sbuf);

  // Allocate buffers for full evaluation
  uint8_t *ys0_full = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  uint8_t *ys1_full = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  assert(ys0_full != NULL && ys1_full != NULL);

  // Party 0 full eval
  memcpy(ys0_full, kS0s, kLambda);
  dpf_eval_full_domain(ys0_full, 0, key, kAlphaBitlen);

  // Party 1 full eval
  memcpy(ys1_full, kS0s + kLambda, kLambda);
  dpf_eval_full_domain(ys1_full, 1, key, kAlphaBitlen);

  // Get evaluation results at alpha
  uint8_t *y0_alpha = ys0_full + (int)kAlpha * kLambda;
  uint8_t *y1_alpha = ys1_full + (int)kAlpha * kLambda;
  // TODO: Rename keys
  uint8_t *wkey = kWkeys + (int)kAlpha * kCwMacWkeyLen;

  // Generate and verify MAC
  uint8_t t0[kCwMacLen];
  uint8_t t1[kCwMacLen];
  gen_cw_mac(t0, t1, y0_alpha, y1_alpha, 1, kLambda, wkey);

  uint8_t beta0[kCwMacCommitLen];
  uint8_t beta1[kCwMacCommitLen];
  commit_cw_mac(beta0, 0, t0, ys0_full, 1ULL << kAlphaBitlen, kLambda, kPubWkeys);
  commit_cw_mac(beta1, 1, t1, ys1_full, 1ULL << kAlphaBitlen, kLambda, kPubWkeys);
  int res = verify_cw_mac(beta0, beta1);
  __uint128_t beta0_int = *(__uint128_t *)beta0;
  EXPECT_EQ(beta0_int, 0);
  EXPECT_EQ(res, 1);

  free(ys0_full);
  free(ys1_full);
  free(key.cw_np1);
  free(key.cws);
  free(sbuf);
}

TEST_F(CwMacBytesTest, VerifyDif) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  // Generate two DCF keys
  Key key_l, key_r;
  key_l.cw_np1 = (uint8_t *)malloc(kLambda);
  key_l.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  key_r.cw_np1 = (uint8_t *)malloc(kLambda);
  key_r.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(key_l.cw_np1 != NULL && key_l.cws != NULL);
  assert(key_r.cw_np1 != NULL && key_r.cws != NULL);

  // Prepare comparison functions
  uint8_t alpha_l_int = kAlphaL;
  uint8_t alpha_r_int = kAlphaR;
  uint8_t *alpha_l = (uint8_t *)&alpha_l_int;
  uint8_t *alpha_r = (uint8_t *)&alpha_r_int;
  Bits alpha_bits_l = {alpha_l, kAlphaBitlen};
  Bits alpha_bits_r = {alpha_r, kAlphaBitlen};
  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  memset(beta, 0, kLambda);
  memcpy(beta, &kBeta, 16);
  Point p_l = {alpha_bits_l, beta};
  Point p_r = {alpha_bits_r, beta};
  CmpFunc cf_l = {p_l, kLtAlpha};
  CmpFunc cf_r = {p_r, kLtAlpha};

  // Generate DCF keys
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key_l, cf_l, sbuf);
  memcpy(sbuf, kS0s, kLambda * 2);
  dcf_gen(key_r, cf_r, sbuf);

  // Allocate buffers for full evaluation
  uint8_t *ys0_full_l = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  uint8_t *ys1_full_l = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  uint8_t *ys0_full_r = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  uint8_t *ys1_full_r = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  uint8_t *ys0_full = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  uint8_t *ys1_full = (uint8_t *)malloc(kLambda * (1ULL << kAlphaBitlen));
  assert(ys0_full_l != NULL && ys1_full_l != NULL);
  assert(ys0_full_r != NULL && ys1_full_r != NULL);
  assert(ys0_full != NULL && ys1_full != NULL);

  // Party 0 full eval for both DCFs
  memcpy(ys0_full_l, kS0s, kLambda);
  dcf_eval_full_domain(ys0_full_l, 0, key_l, kAlphaBitlen);
  memcpy(ys0_full_r, kS0s, kLambda);
  dcf_eval_full_domain(ys0_full_r, 0, key_r, kAlphaBitlen);

  // Party 1 full eval for both DCFs
  memcpy(ys1_full_l, kS0s + kLambda, kLambda);
  dcf_eval_full_domain(ys1_full_l, 1, key_l, kAlphaBitlen);
  memcpy(ys1_full_r, kS0s + kLambda, kLambda);
  dcf_eval_full_domain(ys1_full_r, 1, key_r, kAlphaBitlen);

  // Compute ys0_full = y0_l + -y0_r for each lambda bytes
  for (size_t i = 0; i < (1ULL << kAlphaBitlen); i++) {
    uint8_t *y0_l = ys0_full_l + i * kLambda;
    uint8_t *y0_r = ys0_full_r + i * kLambda;
    uint8_t *y0 = ys0_full + i * kLambda;
    memcpy(y0, y0_l, kLambda);
    group_neg(y0);
    group_add(y0, y0_r);
  }

  // Compute ys1_full = y1_l + -y1_r for each lambda bytes
  for (size_t i = 0; i < (1ULL << kAlphaBitlen); i++) {
    uint8_t *y1_l = ys1_full_l + i * kLambda;
    uint8_t *y1_r = ys1_full_r + i * kLambda;
    uint8_t *y1 = ys1_full + i * kLambda;
    memcpy(y1, y1_l, kLambda);
    group_neg(y1);
    group_add(y1, y1_r);
  }

  // Get evaluation results at alpha_l and alpha_r
  uint8_t *y0_alpha_l = ys0_full + (int)kAlphaL * kLambda;
  uint8_t *y1_alpha_l = ys1_full + (int)kAlphaL * kLambda;
  uint8_t *wkey = kWkeys + (int)kAlphaL * kCwMacWkeyLen;

  // Generate and verify MAC
  uint8_t t0[kCwMacLen];
  uint8_t t1[kCwMacLen];
  gen_cw_mac(t0, t1, y0_alpha_l, y1_alpha_l, kAlphaR - kAlphaL, kLambda, wkey);

  uint8_t beta0[kCwMacCommitLen];
  uint8_t beta1[kCwMacCommitLen];
  commit_cw_mac(beta0, 0, t0, ys0_full, 1ULL << kAlphaBitlen, kLambda, kPubWkeys);
  commit_cw_mac(beta1, 1, t1, ys1_full, 1ULL << kAlphaBitlen, kLambda, kPubWkeys);
  int res = verify_cw_mac(beta0, beta1);
  __uint128_t beta0_int = *(__uint128_t *)beta0;
  EXPECT_EQ(beta0_int, 0);
  EXPECT_EQ(res, 1);

  // Cleanup
  free(ys0_full);
  free(ys1_full);
  free(ys0_full_l);
  free(ys1_full_l);
  free(ys0_full_r);
  free(ys1_full_r);
  free(key_l.cw_np1);
  free(key_l.cws);
  free(key_r.cw_np1);
  free(key_r.cws);
  free(sbuf);
}
