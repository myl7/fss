#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <dpf.h>

#define kThreadsPerBlock 256
#define kIterNum 1000000

#define kSeed 114514
#define kAlpha 107
#define kAlphaBitlen 16
#define kBeta 604

extern void prg_init(const uint32_t nonce[2]);

static inline double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

__global__ void gen_kernel(
  uint8_t *cw_np1_dev, uint8_t *cws_dev, const uint8_t *alpha_dev, const uint8_t *beta_dev, uint8_t *sbuf_dev) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= kIterNum) return;

  uint8_t *sbuf = sbuf_dev + tid * kLambda * 6;
  uint8_t *cw_np1 = cw_np1_dev + tid * kLambda;
  uint8_t *cws = cws_dev + tid * kCwLen * kAlphaBitlen;

  DpfKey k = {cws, cw_np1};
  PointFunc pf = {{alpha_dev, kAlphaBitlen}, beta_dev};
  dpf_gen(k, pf, sbuf);
}

__global__ void eval_kernel(
  uint8_t *sbuf_dev, uint8_t b, const uint8_t *cw_np1_dev, const uint8_t *cws_dev, const uint8_t *x_dev) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= kIterNum) return;

  uint8_t *sbuf = sbuf_dev + tid * kLambda * 3;
  const uint8_t *x = x_dev + tid * sizeof(uint16_t);
  const uint8_t *cw_np1 = cw_np1_dev + tid * kLambda;
  const uint8_t *cws = cws_dev + tid * kCwLen * kAlphaBitlen;

  DpfKey k = {(uint8_t *)cws, (uint8_t *)cw_np1};
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

  uint32_t nonce[2] = {0, 1};
  prg_init(nonce);

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

  uint8_t *sbuf_dev;
  err = cudaMalloc(&sbuf_dev, kLambda * 6 * kIterNum);
  cudaAssert(err);
  uint8_t *cw_np1_dev;
  err = cudaMalloc(&cw_np1_dev, kLambda * kIterNum);
  cudaAssert(err);
  uint8_t *cws_dev;
  err = cudaMalloc(&cws_dev, kCwLen * kAlphaBitlen * kIterNum);
  cudaAssert(err);

  uint8_t *s0s = (uint8_t *)malloc(kLambda * 2);
  assert(s0s != NULL);
  memset(s0s, 0, kLambda * 2);
  s0s[0] = 114;
  s0s[kLambda - 1] = 51;
  s0s[kLambda] = 4;
  for (int i = 0; i < kIterNum; i++) {
    err = cudaMemcpy(sbuf_dev + i * kLambda * 6, s0s, kLambda * 2, cudaMemcpyHostToDevice);
    cudaAssert(err);
  }

  t = get_time();
  int block_num = (kIterNum + kThreadsPerBlock - 1) / kThreadsPerBlock;
  gen_kernel<<<block_num, kThreadsPerBlock>>>(cw_np1_dev, cws_dev, alpha_dev, beta_dev, sbuf_dev);
  cudaDeviceSynchronize();
  printf("dpf_gen (s/%d): %lf\n", kIterNum, get_time() - t);

  err = cudaFree(alpha_dev);
  cudaAssert(err);
  err = cudaFree(beta_dev);
  cudaAssert(err);
  err = cudaFree(sbuf_dev);
  cudaAssert(err);

  uint16_t *x_int = (uint16_t *)malloc(sizeof(uint16_t) * kIterNum);
  assert(x_int != NULL);
  for (int i = 0; i < kIterNum; i++) {
    if (i == 0) {
      x_int[i] = kAlpha;
    } else {
      x_int[i] = rand() & UINT16_MAX;
    }
  }

  err = cudaMalloc(&sbuf_dev, kLambda * 3 * kIterNum);
  cudaAssert(err);

  uint8_t *x_dev;
  err = cudaMalloc(&x_dev, sizeof(uint16_t) * kIterNum);
  cudaAssert(err);
  err = cudaMemcpy(x_dev, x_int, sizeof(uint16_t) * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);

  __uint128_t *y0_int = (__uint128_t *)malloc(kLambda * kIterNum);
  assert(y0_int != NULL);
  __uint128_t *y1_int = (__uint128_t *)malloc(kLambda * kIterNum);
  assert(y1_int != NULL);

  uint8_t *sbuf_tmp = (uint8_t *)malloc(kLambda * 3 * kIterNum);
  assert(sbuf_tmp != NULL);
  for (int i = 0; i < kIterNum; i++) {
    memcpy(sbuf_tmp + i * kLambda * 3, s0s, kLambda);
  }

  t = get_time();
  err = cudaMemcpy(sbuf_dev, sbuf_tmp, kLambda * 3 * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);
  eval_kernel<<<block_num, kThreadsPerBlock>>>(sbuf_dev, 0, cw_np1_dev, cws_dev, x_dev);
  cudaDeviceSynchronize();
  err = cudaMemcpy(sbuf_tmp, sbuf_dev, kLambda * 3 * kIterNum, cudaMemcpyDeviceToHost);
  cudaAssert(err);
  printf("dpf_eval (s/%d): %lf\n", kIterNum, get_time() - t);

  for (int i = 0; i < kIterNum; i++) {
    y0_int[i] = *(__uint128_t *)(sbuf_tmp + i * kLambda * 3);
  }

  for (int i = 0; i < kIterNum; i++) {
    memcpy(sbuf_tmp + i * kLambda * 3, s0s + kLambda, kLambda);
  }

  err = cudaMemcpy(sbuf_dev, sbuf_tmp, kLambda * 3 * kIterNum, cudaMemcpyHostToDevice);
  cudaAssert(err);
  eval_kernel<<<block_num, kThreadsPerBlock>>>(sbuf_dev, 1, cw_np1_dev, cws_dev, x_dev);
  cudaDeviceSynchronize();
  err = cudaMemcpy(sbuf_tmp, sbuf_dev, kLambda * 3 * kIterNum, cudaMemcpyDeviceToHost);
  cudaAssert(err);
  for (int i = 0; i < kIterNum; i++) {
    y1_int[i] = *(__uint128_t *)(sbuf_tmp + i * kLambda * 3);
  }

  for (int i = 0; i < kIterNum; i++) {
    __uint128_t y_int = *(y0_int + i) + *(y1_int + i);
    if (x_int[i] == kAlpha) {
      assert(y_int == kBeta);
    } else {
      assert(y_int == 0);
    }
  }

  err = cudaFree(cw_np1_dev);
  cudaAssert(err);
  err = cudaFree(cws_dev);
  cudaAssert(err);
  err = cudaFree(sbuf_dev);
  cudaAssert(err);
  err = cudaFree(x_dev);
  cudaAssert(err);
  free(s0s);
  free(x_int);
  free(y0_int);
  free(y1_int);
  free(sbuf_tmp);
  return 0;
}
