# CUDA usage notes

## Group types as kernel template arguments

The built-in groups are safe to use as `__global__` kernel template arguments, and `fss::group::Uint128<fss::group::Uint128Modulus{.low = ..., .high = ...}>` is the form to use for wide moduli: it carries the modulus as one structural non-type template parameter whose two `uint64_t` fields hold the low and high halves.
The default compatibility switch keeps the deprecated `fss::group::Uint<__uint128_t, mod>` spelling for positive moduli up to `2^64 - 1`, and on the probed CUDA 12.6/GCC 11.4 and 13.4/GCC 13.3 toolchains that range compiles through a generic kernel launch, with GCC 11.4 at most warning on the upper half of the range while reading the value back correctly.
The v1.3.0 form with modulus `2^127` fails in the generic-launch probes on those toolchains, even though the old static assertion passes.
In the retained CUDA 12.6 generated files, ordinary host C++ spells `2^127` as a high/low expression, whereas the launch stub spells it as one large decimal literal that GCC reads back as zero.
`fss::group::Uint<T, mod>` keeps the modulus in `T` itself, with `T` from `uint8_t` through `uint64_t` as the primary form.
