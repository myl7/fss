# CUDA usage notes

## Group types as kernel template arguments

The built-in groups are safe to use as `__global__` kernel template arguments.
`fss::group::Uint128<fss::group::Uint128Modulus{.low = ..., .high = ...}>` carries its modulus as one structural non-type template parameter whose two `uint64_t` fields hold the low and high halves, because nvcc host stubs cannot spell a `__uint128_t` non-type template parameter.
`fss::group::Uint<T, mod>` keeps the modulus in `T` itself, so it is limited to `T` up to `uint64_t`.
