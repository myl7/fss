# CUDA usage notes

## Uint as a kernel template argument


nvcc fails to compile the stub file when `fss::group::Uint<__uint128_t, ...>` is used as a template argument to a `__global__` kernel.
Every kernel instantiation gets a host stub, and the stub spells the modulus as unsuffixed decimal literals; a 64-bit half of the modulus at or above 2^63 is not a valid unsuffixed literal for the host compiler.
`__device__` functions are not affected, because no stub is generated for them.
The failure comes from how nvcc generates the host stub, not from a particular toolkit version.

Workaround: wrap the type in a plain aggregate struct that satisfies `Groupable` but has no `__uint128_t` non-type template parameter in its name.
The struct must have no user-declared constructors to remain an aggregate.
See `third_party/fss/bench.cu` for an example.
