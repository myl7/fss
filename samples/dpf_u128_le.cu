#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <fss/dpf.h>

#define kThreadsPerBlock 256
#define kIterNum 1000000

// Constants
#define kSeed 114514
#define kAlpha 107
#define kAlphaBitlen 16
#define kBeta 604

extern void prg_init(const uint8_t *state, int state_len);

static inline double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

__global__ void gen_kernel(
  uint8_t *cw_np1_dev, uint8_t *cws_dev, uint8_t *alpha_dev, uint8_t *beta_dev, uint8_t *sbuf_dev) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= kIterNum) return;

  uint8_t *sbuf = sbuf_dev + tid * kLambda * 6;
  uint8_t *cw_np1 = cw_np1_dev + tid * kLambda;
  uint8_t *cws = cws_dev + tid * kDpfCwLen * kAlphaBitlen;

  Key k = {cws, cw_np1};
  PointFunc pf = {{alpha_dev, kAlphaBitlen}, beta_dev};
  dpf_gen(k, pf, sbuf);
}

__global__ void eval_kernel(
  uint8_t *sbuf_dev, uint8_t b, uint8_t *cw_np1_dev, uint8_t *cws_dev, uint8_t *x_dev) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= kIterNum) return;

  uint8_t *sbuf = sbuf_dev + tid * kLambda * 3;
  uint8_t *x = x_dev + tid * sizeof(uint16_t);
  uint8_t *cw_np1 = cw_np1_dev + tid * kLambda;
  uint8_t *cws = cws_dev + tid * kDpfCwLen * kAlphaBitlen;

  Key k = {cws, cw_np1};
  Bits x_bits = {x, kAlphaBitlen};
  dpf_eval(sbuf, b, k, x_bits);
}

void cudaAssert(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    assert(false);
  }
}

int main() {
  srand(kSeed);
  assert(kLambda == 16);
  double t;
  cudaError_t err;

  // Initialize PRG
  uint8_t keys[1][8] = {{1}};
  prg_init((uint8_t *)keys, 8);

  // Initialize seeds
  uint8_t *s0s = (uint8_t *)malloc(kLambda * 2);
  assert(s0s != NULL);
  memset(s0s, 0, kLambda * 2);
  s0s[0] = 114;
  s0s[kLambda - 1] = 51;
  s0s[kLambda] = 4;

  // Prepare point function
  uint16_t alpha_int = kAlpha;
  uint8_t *alpha_dev;
  err = cudaMalloc(&alpha_dev, sizeof(uint16_t));
  cudaAssert(err);
  err = cudaMemcpy(alpha_dev, &alpha_int, sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaAssert(err);
  __uint128_t beta_int = kBeta;
  uint8_t *beta_dev;
  err = cudaMalloc(&beta_dev, sizeof(__uint128_t));
  cudaAssert(err);
  err = cudaMemcpy(beta_dev, &beta_int, sizeof(__uint128_t), cudaMemcpyHostToDevice);
  cudaAssert(err);

  // Allocate buffers for generation
  uint8_t *sbuf_gen_dev;
  err = cudaMalloc(&sbuf_gen_dev, kLambda * 6 * kIterNum);
  cudaAssert(err);
  uint8_t *cw_np1_dev;
  err = cudaMalloc(&cw_np1_dev, kLambda * kIterNum);
  cudaAssert(err);
  uint8_t *cws_dev;
  err = cudaMalloc(&cws_dev, kDpfCwLen * kAlphaBitlen * kIterNum);
  cudaAssert(err);

  // Prepare temporary buffer for generation
  uint8_t *sbuf_tmp = (uint8_t *)malloc(kLambda * 6 * kIterNum);
  assert(sbuf_tmp != NULL);
  for (int i = 0; i < kIterNum; i++) {
    memcpy(sbuf_tmp + i * kLambda * 6, s0s, kLambda * 2);
  }

  t = get_time();
  // Copy initial seeds to device
  err = cudaMemcpy(sbuf_gen_dev, sbuf_tmp, kLambda * 6 * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);

  // Generate DPF and measure time
  int block_num = (kIterNum + kThreadsPerBlock - 1) / kThreadsPerBlock;
  gen_kernel<<<block_num, kThreadsPerBlock>>>(cw_np1_dev, cws_dev, alpha_dev, beta_dev, sbuf_gen_dev);
  cudaDeviceSynchronize();
  printf("dpf_gen (s/%d): %lf\n", kIterNum, get_time() - t);

  // Free memory no longer needed
  err = cudaFree(alpha_dev);
  cudaAssert(err);
  err = cudaFree(beta_dev);
  cudaAssert(err);
  err = cudaFree(sbuf_gen_dev);
  cudaAssert(err);
  free(sbuf_tmp);

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

  // Allocate device memory for evaluation
  uint8_t *x_dev;
  err = cudaMalloc(&x_dev, sizeof(uint16_t) * kIterNum);
  cudaAssert(err);
  err = cudaMemcpy(x_dev, x_int, sizeof(uint16_t) * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);

  // Allocate buffer for evaluation
  uint8_t *sbuf_eval_dev;
  err = cudaMalloc(&sbuf_eval_dev, kLambda * 3 * kIterNum);
  cudaAssert(err);

  // Prepare temporary buffer for evaluation
  sbuf_tmp = (uint8_t *)malloc(kLambda * 3 * kIterNum);
  assert(sbuf_tmp != NULL);
  for (int i = 0; i < kIterNum; i++) {
    memcpy(sbuf_tmp + i * kLambda * 3, s0s, kLambda);
  }

  t = get_time();
  // Copy initial seeds to device
  err = cudaMemcpy(sbuf_eval_dev, sbuf_tmp, kLambda * 3 * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);

  // Evaluate at points and measure time
  eval_kernel<<<block_num, kThreadsPerBlock>>>(sbuf_eval_dev, 0, cw_np1_dev, cws_dev, x_dev);
  cudaDeviceSynchronize();
  err = cudaMemcpy(sbuf_tmp, sbuf_eval_dev, kLambda * 3 * kIterNum, cudaMemcpyDeviceToHost);
  cudaAssert(err);
  for (int i = 0; i < kIterNum; i++) {
    memcpy(y0 + i * kLambda, sbuf_tmp + i * kLambda * 3, kLambda);
  }
  printf("dpf_eval (s/%d): %lf\n", kIterNum, get_time() - t);

  // Prepare for second party evaluation
  for (int i = 0; i < kIterNum; i++) {
    memcpy(sbuf_tmp + i * kLambda * 3, s0s + kLambda, kLambda);
  }
  err = cudaMemcpy(sbuf_eval_dev, sbuf_tmp, kLambda * 3 * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);

  // Evaluate second party
  eval_kernel<<<block_num, kThreadsPerBlock>>>(sbuf_eval_dev, 1, cw_np1_dev, cws_dev, x_dev);
  cudaDeviceSynchronize();
  err = cudaMemcpy(sbuf_tmp, sbuf_eval_dev, kLambda * 3 * kIterNum, cudaMemcpyDeviceToHost);
  cudaAssert(err);
  for (int i = 0; i < kIterNum; i++) {
    memcpy(y1 + i * kLambda, sbuf_tmp + i * kLambda * 3, kLambda);
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
  err = cudaFree(cw_np1_dev);
  cudaAssert(err);
  err = cudaFree(cws_dev);
  cudaAssert(err);
  err = cudaFree(sbuf_eval_dev);
  cudaAssert(err);
  err = cudaFree(x_dev);
  cudaAssert(err);
  free(sbuf_tmp);
  free(s0s);
  free(x_int);
  free(y0);
  free(y1);
  return 0;
}
