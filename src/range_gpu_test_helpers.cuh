// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "core_api_test_helpers.cuh"

namespace fss_test {

template <int bits, typename In, int frontier = 16, int threads = 256, typename Scheme>
void CheckGpuRange(Scheme &scheme, In first, In last) {
  int devices = 0;
  cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) GTEST_SKIP() << "no CUDA device available";
  ASSERT_EQ(status, cudaSuccess);
  const int4 seeds[2] = {{11, 22, 33, 44}, {55, 66, 77, 88}};
  ComponentKey<Scheme, In> keys[2];
  GenPair(scheme, keys[0], keys[1], cuda::std::span<const int4, 2>(seeds, 2),
      first + (last - first) / 2, int4{7, 19, 23, 42});
  const size_t count = static_cast<size_t>(last - first) + 1;
  constexpr bool kHasValues = requires { std::declval<typename Scheme::Cw &>().v; };
  int4 *cws = nullptr;
  int4 *vws = nullptr;
  int4 *outputs = nullptr;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&cws, (bits + 1) * sizeof(int4)), cudaSuccess);
  if constexpr (kHasValues) ASSERT_EQ(cudaMalloc(&vws, (bits + 1) * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&outputs, (count + 1) * sizeof(int4)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(cws, keys[0].cws, (bits + 1) * sizeof(int4),
      cudaMemcpyHostToDevice, stream), cudaSuccess);
  if constexpr (kHasValues)
    ASSERT_EQ(cudaMemcpyAsync(vws, keys[0].vws, (bits + 1) * sizeof(int4),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
  const int4 sentinel = {91, 92, 93, 94};
  std::vector<int4> actual(count + 1, sentinel);
  for (int party = 0; party < 2; ++party) {
    SCOPED_TRACE(party);
    actual.assign(count + 1, sentinel);
    ASSERT_EQ(cudaMemcpyAsync(outputs, actual.data(), actual.size() * sizeof(int4),
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    if constexpr (kHasValues) {
      ASSERT_EQ((scheme.template EvalRangeGpu<frontier, threads>(party, first, last,
          seeds[party], {cws, bits + 1}, {vws, bits + 1}, keys[party].tr,
          {outputs, count + 1}, stream)), cudaSuccess);
    } else {
      ASSERT_EQ((scheme.template EvalRangeGpu<frontier, threads>(party, first, last,
          seeds[party], {cws, bits + 1}, keys[party].tr,
          {outputs, count + 1}, stream)), cudaSuccess);
    }
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), outputs, actual.size() * sizeof(int4),
        cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t index = 0; index < count; ++index) {
      int4 expected = EvalKey(scheme, party, keys[party], static_cast<In>(first + index));
      EXPECT_EQ(actual[index].x, expected.x) << index;
      EXPECT_EQ(actual[index].y, expected.y) << index;
      EXPECT_EQ(actual[index].z, expected.z) << index;
      EXPECT_EQ(actual[index].w, expected.w) << index;
    }
    EXPECT_EQ(actual[count].x, sentinel.x);
    EXPECT_EQ(actual[count].y, sentinel.y);
    EXPECT_EQ(actual[count].z, sentinel.z);
    EXPECT_EQ(actual[count].w, sentinel.w);
  }
  if constexpr (kHasValues) {
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{1}, In{0},
        seeds[0], {cws, bits + 1}, {vws, bits + 1}, keys[0].tr, {outputs, count}, stream)),
        cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits}, {vws, bits}, keys[0].tr, {outputs, count}, stream)),
        cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits + 1}, {vws, bits + 1}, keys[0].tr, {outputs, count - 1}, stream)),
        cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits + 1}, {vws, bits + 1}, keys[0].tr,
        {static_cast<int4 *>(nullptr), count}, stream)), cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits + 1}, {vws, bits + 1}, keys[0].tr, {outputs, count}, stream, 0)),
        cudaErrorInvalidValue);
    if constexpr (bits == 128) {
      EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{0}, ~In{0},
          seeds[0], {cws, bits + 1}, {vws, bits + 1}, keys[0].tr, {outputs, size_t(-1)}, stream)),
          cudaErrorInvalidValue);
      EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{0},
          static_cast<In>(size_t(-1) / sizeof(int4)), seeds[0],
          {cws, bits + 1}, {vws, bits + 1}, keys[0].tr, {outputs, size_t(-1)}, stream)),
          cudaErrorInvalidValue);
    } else if constexpr (bits < sizeof(In) * 8) {
      EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{0}, In{1} << bits,
          seeds[0], {cws, bits + 1}, {vws, bits + 1}, keys[0].tr, {outputs, count}, stream)),
          cudaErrorInvalidValue);
    }
  } else {
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{1}, In{0},
        seeds[0], {cws, bits + 1}, keys[0].tr, {outputs, count}, stream)),
        cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits}, keys[0].tr, {outputs, count}, stream)),
        cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits + 1}, keys[0].tr, {outputs, count - 1}, stream)),
        cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits + 1}, keys[0].tr,
        {static_cast<int4 *>(nullptr), count}, stream)), cudaErrorInvalidValue);
    EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, first, last,
        seeds[0], {cws, bits + 1}, keys[0].tr, {outputs, count}, stream, 0)),
        cudaErrorInvalidValue);
    if constexpr (bits == 128) {
      EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{0}, ~In{0},
          seeds[0], {cws, bits + 1}, keys[0].tr, {outputs, size_t(-1)}, stream)),
          cudaErrorInvalidValue);
      EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{0},
          static_cast<In>(size_t(-1) / sizeof(int4)), seeds[0],
          {cws, bits + 1}, keys[0].tr, {outputs, size_t(-1)}, stream)),
          cudaErrorInvalidValue);
    } else if constexpr (bits < sizeof(In) * 8) {
      EXPECT_EQ((scheme.template EvalRangeGpu<frontier, threads>(false, In{0}, In{1} << bits,
          seeds[0], {cws, bits + 1}, keys[0].tr, {outputs, count}, stream)),
          cudaErrorInvalidValue);
    }
  }
  EXPECT_EQ(cudaFree(outputs), cudaSuccess);
  if constexpr (kHasValues) EXPECT_EQ(cudaFree(vws), cudaSuccess);
  EXPECT_EQ(cudaFree(cws), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace fss_test
