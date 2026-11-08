// SPDX-License-Identifier: Apache-2.0
/**
 * @file compat.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 *
 * @brief Single build-time switch for the deprecated v1 Cw-form compatibility entries.
 *
 * FSS_ENABLE_V1_COMPAT is the only definition site of the switch. It is a numeric
 * macro: the default 1 keeps the deprecated Cw-form overloads, the deprecated
 * aggregate-Key entries, and their companion types available, and an explicit 0
 * removes them so only the current split-component APIs remain.
 *
 * Define it to the same value in every translation unit of a program, including
 * ones that only include FSS headers indirectly. The scheme classes gain or lose
 * members with this macro, so mixed settings create distinct class definitions
 * and violate the one-definition rule.
 */

#pragma once

#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <cuda/std/array>

#ifndef FSS_ENABLE_V1_COMPAT
#define FSS_ENABLE_V1_COMPAT 1
#endif

#if FSS_ENABLE_V1_COMPAT

namespace fss::compat::detail {

// Record templates backing the deprecated v1.3.0 aggregate-Key nested names.
// Each scheme header exposes one of these through a deprecated alias, so the
// deprecation warning fires only where the legacy name is spelled. Member
// declarations and definitions use the record types directly: a deprecated
// alias in a member signature would warn on every modern instantiation under
// NVCC. The SchemeTag template parameter keeps the nested types of different
// schemes distinct even when their field shapes coincide.

/** Correction words plus the party seed, for Dcf::Key and GrottoDcf::Key. */
template <typename SchemeTag, typename Cw, int n>
struct CwsKeyRecord {
  Cw cws[n];
  int4 s0;
};

/** Tree correction words, the separate output correction word, and the seed, for HalfTreeDpf::Key. */
template <typename SchemeTag, typename Cw, int n>
struct OcwKeyRecord {
  [[deprecated("the par_depth template parameter is gone; pass par_depth to EvalAll() instead")]]
  static constexpr int kParDepth = -1;
  Cw cws[n];
  int4 ocw;
  int4 s0;
};

/** Tree correction words, the packed-output correction word, and the seed, for PackedHalfTreeDpf::Key. */
template <typename SchemeTag, typename Cw, int n>
struct FcwKeyRecord {
  Cw cws[n];
  int4 fcw;
  int4 s0;
};

/** Tree correction words, the correction seed, the output correction word, and the seed, for Vdpf::Key. */
template <typename SchemeTag, typename Cw, int n>
struct VdpfKeyRecord {
  Cw cws[n];
  cuda::std::array<int4, 4> cs;
  int4 ocw;
  int4 s0;
};

/** Borrowed, party-tagged parity-tree view, for GrottoDcf::ParityTree. */
template <typename SchemeTag>
struct ParityTreeRecord {
  bool *p;
  bool b;
};

/** Host-owned parity tree exposing a borrowed ParityTreeRecord view, for GrottoDcf::PreprocessedTree. */
template <typename SchemeTag>
struct PreprocessedTreeRecord {
  std::unique_ptr<bool[]> p;
  bool b;

  [[deprecated("use PreprocessOwned() and OwnedParityTree::View()")]]
  ParityTreeRecord<SchemeTag> view() const { return ParityTreeRecord<SchemeTag>{p.get(), b}; }
};

/**
 * Checked conversion behind the deprecated legacy int bucket-size constants.
 * The static data member initializers that call it instantiate only where the
 * legacy name is referenced, so the domain check fires exactly there and wide
 * modern configurations that never use the legacy constant still compile.
 */
template <__uint128_t kLegacySize>
constexpr int LegacyBucketSize() {
  static_assert(kLegacySize <= static_cast<__uint128_t>(std::numeric_limits<int>::max()),
      "legacy b_size: kBucketSize does not fit the legacy int domain");
  return static_cast<int>(kLegacySize);
}

/** Output share plus the corrected per-point hash, for Vdpf::EvalResult. */
template <typename SchemeTag>
struct EvalResultRecord {
  int4 y;
  cuda::std::array<int4, 4> pi_tilde;
};

}  // namespace fss::compat::detail

#endif  // FSS_ENABLE_V1_COMPAT
