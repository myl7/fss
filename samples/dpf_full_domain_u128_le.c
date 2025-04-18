// For real-time extensions
#define _POSIX_C_SOURCE 199309L

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <dpf.h>
#include <omp.h>

// Constants
#define kSeed 114514
#define kAlpha 107
#define kAlphaBitlen 16
#define kBeta 604
#define kIterNum 1000

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

  // Configure OpenMP for nested parallelism
  // omp_set_max_active_levels(kParallelDepth + 1);
  printf("OpenMP thread num: %d\n", omp_get_max_threads());

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

  // Generate DPF
  memcpy(sbuf, s0s, kLambda * 2);
  dpf_gen(k, pf, sbuf);

  // Allocate result buffers for full domain evaluation
  size_t domain_size = 1ULL << kAlphaBitlen;
  uint8_t *y0_full = (uint8_t *)malloc(kLambda * domain_size);
  assert(y0_full != NULL);
  uint8_t *y1_full = (uint8_t *)malloc(kLambda * domain_size);
  assert(y1_full != NULL);

  // Evaluate full domain and measure time
  t = get_time();
  for (int i = 0; i < kIterNum; i++) {
    memcpy(y0_full, s0s, kLambda);
    dpf_eval_full_domain(y0_full, 0, k, kAlphaBitlen);
  }
  printf("dpf_eval_full_domain (s/%d): %lf\n", kIterNum, get_time() - t);

  // Evaluate second party
  memcpy(y1_full, s0s + kLambda, kLambda);
  dpf_eval_full_domain(y1_full, 1, k, kAlphaBitlen);

  // Verify results at random points
  const int kNumTrials = 100;
  for (int i = 0; i < kNumTrials; i++) {
    uint16_t x = i == 0 ? kAlpha : rand() & UINT16_MAX;
    uint8_t *y0 = y0_full + (int)x * kLambda;
    uint8_t *y1 = y1_full + (int)x * kLambda;

    group_add(y0, y1);
    __uint128_t y_int;
    memcpy(&y_int, y0, kLambda);
    if (x == kAlpha) {
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
  free(y0_full);
  free(y1_full);
  return 0;
}
