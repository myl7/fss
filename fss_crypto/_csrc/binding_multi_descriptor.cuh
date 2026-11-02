// SPDX-License-Identifier: Apache-2.0
/**
 * @file binding_multi_descriptor.cuh
 * @brief Package-private launch descriptors for multi-point schemes.
 *
 * Each descriptor stores the wide Group non-type template parameter behind a
 * GroupTag, so generated CUDA host stubs name only safe 64-bit pieces.
 */

#pragma once
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>

namespace fss::gpu::detail {

template <int in_bits, int max_points, int bucket_bits, typename GroupTag, typename Prg,
    typename Prp, typename In, int ch_lambda = 80>
struct DmpfLaunchDescriptor {
  using Scheme = Dmpf<in_bits, max_points, bucket_bits, typename GroupTag::type, Prg, Prp, In, ch_lambda>;
  using Group = typename GroupTag::type;
  using Input = In;
  using InputType = In;
  using PrgBackend = Prg;
  using PrpBackend = Prp;
  using Cw = typename Scheme::Cw;
  static constexpr int kInputBits = in_bits;
  static constexpr int kBucketBits = bucket_bits;
  static constexpr int kBucketCount = Scheme::kBucketCount;
};

template <int in_bits, int max_points, int bucket_bits, typename GroupTag, typename Prg,
    typename XorHash, typename Hash, typename Prp, typename In, int ch_lambda = 80>
struct VdmpfLaunchDescriptor {
  using Scheme = Vdmpf<in_bits, max_points, bucket_bits, typename GroupTag::type, Prg, XorHash,
      Hash, Prp, In, ch_lambda>;
  using Group = typename GroupTag::type;
  using Input = In;
  using InputType = In;
  using PrgBackend = Prg;
  using XorHashBackend = XorHash;
  using HashBackend = Hash;
  using PrpBackend = Prp;
  using InnerScheme = Vdpf<bucket_bits, typename GroupTag::type, Prg, XorHash, Hash, __uint128_t>;
  using Cw = typename Scheme::Cw;
  static constexpr int kInputBits = in_bits;
  static constexpr int kBucketBits = bucket_bits;
  static constexpr int kBucketCount = Scheme::kBucketCount;
};

}  // namespace fss::gpu::detail
