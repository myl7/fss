// For real-time extensions
#define _POSIX_C_SOURCE 199309L

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <dpf.h>

#define kSeed 114514
#define kAlpha 107
#define kAlphaBitlen 16
#define kBeta 604

extern void prg_init(const uint8_t keys[2][16]);

static inline double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

int main() {
  srand(kSeed);
  assert(kLambda == 16);
  double t;

  uint8_t keys[2][16] = {{0}, {1}};
  prg_init(keys);

  uint16_t alpha_int = kAlpha;
  uint8_t *alpha = (uint8_t *)&alpha_int;
  int alpha_bitlen = kAlphaBitlen;
  Bits alpha_bits = {alpha, alpha_bitlen};
  __uint128_t beta_int = kBeta;
  uint8_t *beta = (uint8_t *)&beta_int;
  PointFunc pf = {alpha_bits, beta};

  uint8_t *s0s = (uint8_t *)malloc(kLambda * 2);
  if (s0s == NULL) return -1;
  memset(s0s, 0, kLambda * 2);
  s0s[0] = 114;
  s0s[kLambda - 1] = 51;
  s0s[kLambda] = 4;
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 6);
  if (sbuf == NULL) return -1;
  memcpy(sbuf, s0s, kLambda * 2);

  DpfKey k;
  k.cw_np1 = (uint8_t *)malloc(kLambda);
  if (k.cw_np1 == NULL) return -1;
  k.cws = (uint8_t *)malloc(kCwLen * kAlphaBitlen);
  if (k.cws == NULL) return -1;

  t = get_time();
  for (int i = 0; i < 1000; i++) {
    memcpy(sbuf, s0s, kLambda * 2);
    dpf_gen(k, pf, sbuf);
  }
  printf("dpf_gen (s/1k): %lf\n", get_time() - t);

  uint16_t *x_int = (uint16_t *)malloc(2 * 1000);
  __uint128_t *y0_int = (__uint128_t *)malloc(kLambda * 1000);
  __uint128_t *y1_int = (__uint128_t *)malloc(kLambda * 1000);
  for (int i = 0; i < 1000; i++) {
    if (i == 0) {
      x_int[i] = kAlpha;
    } else {
      x_int[i] = rand() & UINT16_MAX;
    }
  }

  t = get_time();
  for (int i = 0; i < 1000; i++) {
    Bits x_bits = {(uint8_t *)(x_int + i), 16};
    memcpy(sbuf, s0s, kLambda);
    dpf_eval(sbuf, 0, k, x_bits);
    memcpy(y0_int + i, sbuf, kLambda);
  }
  printf("dpf_eval (s/1k): %lf\n", get_time() - t);

  for (int i = 0; i < 1000; i++) {
    Bits x_bits = {(uint8_t *)(x_int + i), 16};
    memcpy(sbuf, s0s + kLambda, kLambda);
    dpf_eval(sbuf, 1, k, x_bits);
    memcpy(y1_int + i, sbuf, kLambda);
  }

  for (int i = 0; i < 1000; i++) {
    __uint128_t y_int = *(y0_int + i) + *(y1_int + i);
    if (x_int[i] == kAlpha) {
      assert(y_int == kBeta);
    } else {
      assert(y_int == 0);
    }
  }

  free(s0s);
  free(sbuf);
  free(k.cw_np1);
  free(k.cws);
  free(x_int);
  free(y0_int);
  free(y1_int);
  return 0;
}
