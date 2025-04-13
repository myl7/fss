#include <random>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>
#include <gtest/gtest.h>
#include <cw_mac_bytes.h>
#include <dpf.h>
#include <sodium.h>

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
    uint8_t keys[32];
    std::generate(std::begin(keys), std::end(keys), std::ref(rbe));
    prg_init((uint8_t *)keys, 32);

    kS0s = (uint8_t *)malloc(kLambda * 2);
    assert(kS0s != NULL);
    std::generate(kS0s, kS0s + kLambda * 2, std::ref(rbe));

    // Initialize keys with sodium random scalar
    kKeys = (uint8_t *)malloc(crypto_core_ristretto255_SCALARBYTES * (1ULL << kAlphaBitlen));
    assert(kKeys != NULL);
    for (size_t i = 0; i < (1ULL << kAlphaBitlen); i++) {
      crypto_core_ristretto255_scalar_random(kKeys + i * crypto_core_ristretto255_SCALARBYTES);
    }

    // Generate public keys (g^key)
    kPubkeys = (uint8_t *)malloc(crypto_core_ristretto255_BYTES * (1ULL << kAlphaBitlen));
    assert(kPubkeys != NULL);
    for (size_t i = 0; i < (1ULL << kAlphaBitlen); i++) {
      crypto_scalarmult_ristretto255_base(
        kPubkeys + i * crypto_core_ristretto255_BYTES, kKeys + i * crypto_core_ristretto255_SCALARBYTES);
    }
  }

  void TearDown() override {
    free(kKeys);
    free(kPubkeys);
    free(kS0s);
  }

  static constexpr uint16_t kAlpha = 107;
  static constexpr uint16_t kAlphaBitlen = 8;
  static constexpr __uint128_t kBeta = 604;

  uint8_t *kKeys;
  uint8_t *kPubkeys;
  uint8_t *kS0s;
};

TEST_F(CwMacBytesTest, VerifyDpf) {
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 6);
  assert(sbuf != NULL);

  DpfKey key;
  key.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(key.cw_np1 != NULL);
  key.cws = (uint8_t *)malloc(kDpfCwLen * kAlphaBitlen);
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

  // Allocate buffers for full evaluation
  uint8_t *ys0_full = (uint8_t *)malloc(kLambda * (1 << kAlphaBitlen));
  uint8_t *ys1_full = (uint8_t *)malloc(kLambda * (1 << kAlphaBitlen));
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
  uint8_t *privkey = kKeys + (int)kAlpha * crypto_core_ristretto255_SCALARBYTES;

  // Generate and verify MAC
  uint8_t t0[crypto_core_ristretto255_BYTES];
  uint8_t t1[crypto_core_ristretto255_BYTES];
  gen_cw_mac(t0, t1, y0_alpha, y1_alpha, 1, kLambda, privkey);

  uint8_t beta0[crypto_core_ristretto255_BYTES];
  uint8_t beta1[crypto_core_ristretto255_BYTES];
  commit_cw_mac(beta0, 0, t0, ys0_full, 1ULL << kAlphaBitlen, kLambda, kPubkeys);
  commit_cw_mac(beta1, 1, t1, ys1_full, 1ULL << kAlphaBitlen, kLambda, kPubkeys);
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
