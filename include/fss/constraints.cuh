// SPDX-License-Identifier: Apache-2.0
/**
 * @file constraints.cuh
 * @copyright Apache License, Version 2.0. Copyright (C) 2026 Yulong Ming <i@myl7.org>.
 * @author Yulong Ming <i@myl7.org>
 */

#pragma once
#include <cstddef>
#include <type_traits>

namespace fss {

/**
 * Stride sentinel selecting the dense level-major batch layout.
 *
 * A batch method of a scheme class takes its correction arrays with key k at
 * layer l stored at l*stride + k. Passing this sentinel in place of a stride
 * means "dense level-major", i.e., the stride is the key count of the call.
 * The sentinel is valid only where the call defines a key count, which is the
 * batch methods of the scheme classes. Every other method documents stride > 0
 * and asserts stride > 0 in debug builds; a release build does not check it, and
 * the value 0 then aliases every layer to index 0.
 * @ingroup fss_api
 */
inline constexpr size_t kDenseStride = 0;

/**
 * Unsigned input storage wide enough for an input domain of in_bits bits.
 *
 * bool is excluded even though standard type traits classify it as unsigned.
 * __uint128_t is accepted explicitly because standard type traits do not
 * recognize it as unsigned in every compiler mode.
 * Schemes impose their additional algorithm-specific bit-size restrictions.
 * This constraint does not require the full domain to fit in memory.
 * @tparam In Input storage type.
 * @tparam in_bits Input domain bit size.
 * @ingroup fss_api
 */
template <typename In, int in_bits>
concept ValidForInputBits =
    !std::is_same_v<std::remove_cv_t<In>, bool> &&
    (std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
    in_bits <= sizeof(In) * 8;

}  // namespace fss
