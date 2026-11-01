// SPDX-License-Identifier: Apache-2.0
// Included twice to keep host-only backends out of device instantiations.
// Operations receive already validated storage; all cryptography stays in Core.
template <typename T, int Kind>
FSS_PRIMITIVE_SPACE void FSS_PRIMITIVE_APPLY(T &backend, const int4 *lhs,
    int4 *out, int mode, __uint128_t x, __uint128_t domain) {
  if constexpr (Kind == 2) {
    auto result = backend.Gen(lhs[0]);
    for (int i = 0; i < kPrgMul; ++i) out[i] = result[i];
  } else if constexpr (Kind == 3) {
    if constexpr (requires { backend.Hash(int4{}); }) out[0] = backend.Hash(lhs[0]);
    else {
      if (mode == 4) {
        auto result = backend.Hash(cuda::std::span<const int4, 4>(lhs, 4));
        for (int i = 0; i < 2; ++i) out[i] = result[i];
      } else {
        auto result = backend.Hash(cuda::std::tuple<int4, const int4>{lhs[0], lhs[1]});
        for (int i = 0; i < 4; ++i) out[i] = result[i];
      }
    }
  } else if constexpr (Kind == 4) {
    __uint128_t value = backend.Permu(lhs[0], x, domain);
    out[0] = {static_cast<int>(value), static_cast<int>(value >> 32),
        static_cast<int>(value >> 64), static_cast<int>(value >> 96)};
  }
}
