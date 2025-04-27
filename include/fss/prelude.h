// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

/**
 * @file prelude.h
 */

#pragma once

#include <stdint.h>

// Verify little-endian
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
  #include <stdbit.h>
  #if __STDC_ENDIAN_NATIVE__ != __STDC_ENDIAN_LITTLE__
    #error "only support little-endian"
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    #error "only support little-endian"
  #endif
#else
  #warning "cannot verify little-endian: only verify for c23/gcc/clang"
#endif

#ifndef kLambda
  /**
 * For security param lambda in papers, it is lambda / 8.
 * Must >= 16 to provide enough security level.
 * It is the byte len of output, but see @ref group.h for details.
 */
  #define kLambda 16
#endif
#if kLambda < 16
  #error "kLambda must be >= 16"
#endif

#if __CUDACC__
  #define FSS_CUDA_HOST_DEVICE __host__ __device__
  #define FSS_CUDA_CONSTANT __constant__
#else
  #define FSS_CUDA_HOST_DEVICE
  #define FSS_CUDA_CONSTANT
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

/**
 * Key used by DPF and DCF.
 * key + `s0s[0]` and key + `s0s[1]` are 2 shares given to 2 parties.
 * Designed for easy serialization.
 */
typedef struct {
  /**
   * Correction words whose len = @ref kDpfCwLen * lambda
   */
  uint8_t *cws;
  /**
   * Last correction word whose len = lambda
   */
  uint8_t *cw_np1;
} Key;

/**
 * Point (`alpha`, `beta`)
 */
typedef struct {
  Bits alpha;
  /**
   * Little-endian lambda bytes viewed as a group element.
   * Its MSB is ignored and assumed to be 0. See @ref group.h for details.
   */
  uint8_t *beta;
} Point;

/**
 * Point function.
 * Output = `beta` when input = `alpha`, otherwise output = 0.
 */
typedef struct {
  Point point;
} PointFunc;

enum Bound {
  /**
   * Output = `beta` when input < `alpha`, otherwise output = 0
   */
  kLtAlpha,
  /**
   * Output = `beta` when input > `alpha`, otherwise output = 0
   */
  kGtAlpha,
};

/**
 * Comparison function.
 * See @ref Bound for its def based on `bound`.
 */
typedef struct {
  Point point;
  enum Bound bound;
} CmpFunc;
