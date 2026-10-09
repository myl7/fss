// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"
#include "gpu_stats.h"
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <limits>

// #include <sys/types.h>

cudaMemPool_t mempool;

extern "C" void initGPUMemPool() {
  int device = 0;
  checkCudaErrors(cudaGetDevice(&device));
  int supported = 0;
  checkCudaErrors(cudaDeviceGetAttribute(
      &supported, cudaDevAttrMemoryPoolsSupported, device));
  if (!supported) {
    fprintf(stderr, "memory pools are not supported on CUDA device %d\n", device);
    exit(EXIT_FAILURE);
  }
  checkCudaErrors(cudaDeviceGetDefaultMemPool(&mempool, device));
  uint64_t threshold = UINT64_MAX;
  checkCudaErrors(cudaMemPoolSetAttribute(
      mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

  // Prefill is setup work. Leave at least half the currently free device memory
  // available for benchmark keys and scratch buffers.
  size_t pool_mib = 512;
  if (const char *value = getenv("FSS_EZPC_POOL_MIB")) {
    char *end = nullptr;
    errno = 0;
    unsigned long long requested = strtoull(value, &end, 10);
    if (errno || value == end || *end != '\0' || *value == '-' ||
        requested > std::numeric_limits<size_t>::max() / (1ULL << 20)) {
      fprintf(stderr, "invalid FSS_EZPC_POOL_MIB: %s\n", value);
      exit(EXIT_FAILURE);
    }
    pool_mib = requested;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t bytes = pool_mib * (1ULL << 20);
  if (bytes > free_bytes / 2) bytes = free_bytes / 2;
  if (bytes > 0) {
    void *dummy = nullptr;
    checkCudaErrors(cudaMallocAsync(&dummy, bytes, 0));
    checkCudaErrors(cudaFreeAsync(dummy, 0));
    checkCudaErrors(cudaStreamSynchronize(0));
  }
}

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrors(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}


extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    if (err != 0) {
        fprintf(stderr, "could not allocate host buffer of %zu bytes: error %d\n",
                size_in_bytes, err);
        exit(EXIT_FAILURE);
    }
    if (pin)
        checkCudaErrors(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    return h_a;
}

extern "C" void gpuFree(void *d_a)
{
    checkCudaErrors(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFree(void *h_a, bool pinned)
{
    if (pinned)
        checkCudaErrors(cudaHostUnregister(h_a));
    free(h_a);
}

extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMalloc(size_in_bytes, true);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}
