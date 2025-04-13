// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file fss_prelude.h
 */

#pragma once

#include <stdint.h>

#ifndef kLambda
/**
 * For security param lambda in papers, it is lambda / 8.
 * It is the byte len of output, but see @ref fss_decl.h for details.
 */
#define kLambda 16
#endif

#ifndef kParallelDepth
/**
 * Parallel degree of full domain eval.
 * <= 0 disables parallelization.
 * > 0 gives 2 ** `kParallelDepth` concurrent tasks running on a fixed num of threads set by OpenMP.
 * Higher is not better, particularly when > physical core num with hyper-threading, and users need trials.
 * For @ref kLambda = 16, 1 already results in top performance.
 */
#define kParallelDepth 1
#endif

#if __CUDACC__
#define HOST_DEVICE __host__ __device__
#define DEVICE_CONST __constant__
#else
#define HOST_DEVICE
#define DEVICE_CONST
#endif

/**
 * Support bitlen that is not a multiple of 8
 */
typedef struct {
  /**
   * Little-endian bytes. Its len must >= (`bitlen` + 7) / 8.
   */
  uint8_t *bytes;
  /**
   * Bit len. Left bits whose indexes > `bitlen` are unused.
   */
  int bitlen;
} Bits;
