// For real-time extensions
#define _POSIX_C_SOURCE 199309L

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <dcf.h>
#include <omp.h>

#define kSeed 114514
#define kAlphaBitlen 17
#define kAlphaBytelen 3

static inline double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

static void gen_rand_bytes(uint8_t *buf, size_t len) {
  for (size_t i = 0; i < len; i++) {
    buf[i] = rand() & 0xFF;
  }
}

// Get int from little-endian kAlphaBitlen bits
static uint32_t get_alpha_int_le(uint8_t *alpha) {
  uint32_t val = 0;
  for (int i = 0; i < kAlphaBytelen; i++) {
    val |= ((uint32_t)alpha[i]) << (i * 8);
  }
  val &= ((1ULL << kAlphaBitlen) - 1);
  return val;
}

int main() {
  assert((kAlphaBitlen + 7) / 8 == kAlphaBytelen);
  assert(kAlphaBytelen <= 4);
  srand(kSeed);
  double t;
  int iter_num;
  printf("OpenMP thread num: %d\n", omp_get_max_threads());
  printf("Alpha bitlen: %d\n", kAlphaBitlen);
  printf("Lambda (B): %d\n", kLambda);

  // Init PRG
  uint8_t *keys = (uint8_t *)malloc(4 * kLambda);
  assert(keys != NULL);
  gen_rand_bytes(keys, 4 * kLambda);
  prg_init(keys, 4 * kLambda);
  free(keys);

  // Sample s0s
  uint8_t *s0s = (uint8_t *)malloc(kLambda * 2);
  assert(s0s != NULL);
  gen_rand_bytes(s0s, kLambda * 2);

  // Prepare comparison function
  uint8_t alpha[kAlphaBytelen];
  gen_rand_bytes(alpha, kAlphaBytelen);
  uint32_t alpha_int = get_alpha_int_le(alpha);
  Bits alpha_bits = {alpha, kAlphaBitlen};

  uint8_t *beta = (uint8_t *)malloc(kLambda);
  assert(beta != NULL);
  gen_rand_bytes(beta, kLambda);

  CmpFunc cf = {alpha_bits, beta, kLtAlpha};

  // Alloc sbuf and k
  uint8_t *sbuf = (uint8_t *)malloc(kLambda * 10);
  assert(sbuf != NULL);

  DcfKey k;
  k.cw_np1 = (uint8_t *)malloc(kLambda);
  assert(k.cw_np1 != NULL);
  k.cws = (uint8_t *)malloc(kDcfCwLen * kAlphaBitlen);
  assert(k.cws != NULL);

  // DCF gen
  memcpy(sbuf, s0s, kLambda * 2);
  dcf_gen(k, cf, sbuf);

  free(sbuf);

  // Alloc result buf
  size_t domain_size = 1ULL << kAlphaBitlen;
  uint8_t *y0s = (uint8_t *)malloc(kLambda * domain_size);
  assert(y0s != NULL);
  uint8_t *y1s = (uint8_t *)malloc(kLambda * domain_size);
  assert(y1s != NULL);

  // DCF full domain eval
  iter_num = 100;
  t = get_time();
  for (int i = 0; i < iter_num; i++) {
    memcpy(y0s, s0s, kLambda);
    dcf_eval_full_domain(y0s, 0, k, kAlphaBitlen);
  }
  printf("dcf_eval_full_domain (ms): %lf\n", (get_time() - t) / iter_num * 1e3);

  memcpy(y1s, s0s + kLambda, kLambda);
  dcf_eval_full_domain(y1s, 1, k, kAlphaBitlen);

  uint8_t *zero = (uint8_t *)malloc(kLambda);
  assert(zero != NULL);
  memset(zero, 0, kLambda);

  // Verify results
  const int kNumTrials = 100;
  uint8_t *y = (uint8_t *)malloc(kLambda);
  assert(y != NULL);
  for (int i = 0; i < kNumTrials; i++) {
    uint8_t x[kAlphaBytelen];
    if (i == 0) {
      memcpy(x, alpha, kAlphaBytelen);
    } else {
      gen_rand_bytes(x, kAlphaBytelen);
    }
    uint32_t x_int = get_alpha_int_le(x);

    uint8_t *y0 = y0s + x_int * kLambda;
    uint8_t *y1 = y1s + x_int * kLambda;
    memcpy(y, y0, kLambda);
    group_add(y, y1);

    uint8_t *y_expected = x_int < alpha_int ? beta : zero;
    if (memcmp(y, y_expected, kLambda) != 0) {
      printf("Mismatch found at x = %lu\n", x_int);
      return 1;
    }
  }

  free(zero);

  // Cleanup
  prg_free();
  free(s0s);
  free(beta);
  free(k.cw_np1);
  free(k.cws);
  free(y0s);
  free(y1s);
  return 0;
}
