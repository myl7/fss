// For real-time extensions
#define _POSIX_C_SOURCE 199309L

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <dpf.h>

// Constants
#define kSeed 114514
#define kAlpha 107
#define kAlphaBitlen 16
#define kBeta 604
#define kIterNum 10000

extern void prg_init(const uint8_t *state, int state_len);

static inline double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

int main() {
  srand(kSeed);
  assert(kLambda == 16);
  double t;

  // Initialize PRG
  uint8_t keys[2][16] = {{0}, {1}};
  prg_init((uint8_t *)keys, 32);

  // Initialize seeds
  uint8_t *s0s = (uint8_t *)malloc(kLambda * 2);
  assert(s0s != NULL);
  memset(s0s, 0, kLambda * 2);
  s0s[0] = 114;
  s0s[kLambda - 1] = 51;
  s0s[kLambda] = 4;

  // Prepare point function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  Bits alpha_bits = {alpha, kAlphaBitlen};
  __uint128_t beta_int = kBeta;
  uint8_t *beta = (uint8_t *)&beta_int;
  PointFunc pf = {alpha_bits, beta};

  // Allocate buffers
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 6);
  assert(sbuf != NULL);

  DpfKey k;
  k.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(k.cw_np1 != NULL);
  k.cws = (uint8_t *)malloc(kDpfCwLen * kAlphaBitlen);
  assert(k.cws != NULL);

  // Generate DPF and measure time
  t = get_time();
  for (int i = 0; i < kIterNum; i++) {
    memcpy(sbuf, s0s, kLambda * 2);
    dpf_gen(k, pf, sbuf);
  }
  printf("dpf_gen (s/%d): %lf\n", kIterNum, get_time() - t);

  // Generate random evaluation points
  uint16_t *x_int = (uint16_t *)malloc(sizeof(uint16_t) * kIterNum);
  assert(x_int != NULL);
  for (int i = 0; i < kIterNum; i++) {
    if (i == 0) {
      x_int[i] = kAlpha;  // First point is alpha
    } else {
      x_int[i] = rand() & UINT16_MAX;
    }
  }

  // Allocate result buffers
  uint8_t *y0 = (uint8_t *)malloc(kLambda * kIterNum);
  assert(y0 != NULL);
  uint8_t *y1 = (uint8_t *)malloc(kLambda * kIterNum);
  assert(y1 != NULL);

  // Evaluate at points and measure time
  t = get_time();
  for (int i = 0; i < kIterNum; i++) {
    Bits x_bits = {(uint8_t *)(x_int + i), kAlphaBitlen};
    memcpy(sbuf, s0s, kLambda);
    dpf_eval(sbuf, 0, k, x_bits);
    memcpy(y0 + i * kLambda, sbuf, kLambda);
  }
  printf("dpf_eval (s/%d): %lf\n", kIterNum, get_time() - t);

  // Evaluate second party
  for (int i = 0; i < kIterNum; i++) {
    Bits x_bits = {(uint8_t *)(x_int + i), kAlphaBitlen};
    memcpy(sbuf, s0s + kLambda, kLambda);
    dpf_eval(sbuf, 1, k, x_bits);
    memcpy(y1 + i * kLambda, sbuf, kLambda);
  }

  // Verify results
  const int kNumTrials = 100;
  for (int i = 0; i < kNumTrials; i++) {
    group_add(y0 + i * kLambda, y1 + i * kLambda);
    __uint128_t y_int;
    memcpy(&y_int, y0 + i * kLambda, kLambda);
    if (x_int[i] == kAlpha) {
      assert(y_int == kBeta);
    } else {
      assert(y_int == 0);
    }
  }

  // Cleanup
  prg_free();
  free(s0s);
  free(sbuf);
  free(k.cw_np1);
  free(k.cws);
  free(x_int);
  free(y0);
  free(y1);
  return 0;
}
