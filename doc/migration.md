# Migrating from v1.3.0

This guide is for C++ callers of v1.3.0.
It lists the v1.3.0 declarations that still compile, the ones that are gone, and the current API to move each one to.
Python code is out of scope.

## Keep compiling with the compatibility switch

The numeric macro `FSS_ENABLE_V1_COMPAT` controls a compatibility layer for the v1.3.0 API.
The default 1 keeps the layer on.
Define the macro to the same value in every translation unit of a program, including the ones that only include FSS headers indirectly.
The scheme classes gain or lose members with the macro, so mixed settings create distinct class definitions and violate the one-definition rule.
Define `FSS_ENABLE_V1_COMPAT=0` to compile against the current API only.

With the switch on, the following v1.3.0 declarations are available again.
Every entry carries the standard `[[deprecated]]` attribute naming its current replacement.

- The aggregate Cw-form `Gen`, `Eval`, and `EvalAll` of `Dpf`, `Dcf`, `HalfTreeDpf`, and `Vdpf`, the Cw-form `Gen`, `Preprocess`, and `EvalAll` of `GrottoDcf`, and the aggregate-Key entries of all eight schemes, including `Vdpf::EvalResult` and the Grotto owner forms.
- The nested types `Dcf::Key`, `HalfTreeDpf::Key`, `PackedHalfTreeDpf::Key`, `GrottoDcf::Key`, `GrottoDcf::ParityTree`, `GrottoDcf::PreprocessedTree`, `Vdpf::Key`, and `Vdpf::EvalResult`.
- The constants `m`, `n`, and `b_size` on `Dmpf` and `Vdmpf`, `kBetaMax` and `kTopDepth` on `PackedHalfTreeDpf`, `kMerkleChunkSize` on `Vdmpf`, and `kParDepth` on `Dpf::Key` and the Half-Tree key.
- The group spelling `Uint<__uint128_t, mod>` for `0 < mod <= 2^64 - 1`.

The restored entries convert to the current component API internally, so they follow the current protocol.
In particular, their `EvalAll()` runs sequentially, and generated keys and proofs match this version, not v1.3.0.
The restored `Vdmpf` aggregate-Key `Gen()` is the v1.3.0 const-seed single-attempt form: it maps both current failure codes to the old failure code 1.

The sections below cover the v1.3.0 calls that no longer compile or no longer behave the same with the switch on.

## Regenerate persisted keys and proofs

Blake3 and Sha256 tuple hashing changed: the domain-separation tag moved to the second hashed block, and hashing now preserves every coordinate bit.
VDPF and VDMPF keys, correction seeds, and proofs from v1.3.0 therefore do not verify with this version, and this version's material does not verify with v1.3.0.
Regenerate persisted key and proof material, and use one version and one proof mode, chain or Merkle, on both parties.

## Construct schemes through their constructors

In v1.3.0 the schemes exposed their backends as public fields:

```cpp
// v1.3.0
DpfPrg prg(ctxs);
Dpf<8, Group, DpfPrg, In> dpf{prg};
dpf.prg = prg;  // the backend was a public, assignable field
```

Each scheme now takes its backends through an `explicit` constructor and keeps them private.
Construct a configured scheme, and construct a new scheme to change a backend.
Default construction and aggregate initialization no longer compile.
In v1.3.0, a default-constructed scheme still required a default-constructible backend: the AES-context PRGs behind `DpfPrg` construct from cipher contexts only, so the declare-then-assign line above was already unavailable for them and the aggregate form was the old way.

```cpp
// Now
DpfPrg prg(ctxs);
Dpf<8, Group, DpfPrg, In> dpf{prg};
Vdpf<8, Group, Prg, XorHash, Hash, In> vdpf{prg, xor_hash, hash};
Dmpf<8, 100, 16, Group, Prg, Prp, In> dmpf{prg, prp};
Vdmpf<8, 100, 16, Group, Prg, XorHash, Hash, Prp, In> vdmpf{prg, xor_hash, hash, prp};
```

`HalfTreeDpf` and `PackedHalfTreeDpf` also changed their backend parameter: they take a hash satisfying `CcrHashable` instead of a PRG satisfying `Prgable`, together with the instance seed.
`fss::hash::Aes128Ccr` implements the CCR construction over a single-key AES-MMO backend.
Wrapping a v1.3.0 PRG with `Hash(z) = prg.Gen(z)[0]` fills the expression but does not satisfy the CCR security requirement, so use a real CCR backend.

```cpp
// Now
using Aes = fss::prg::Aes128Mmo<1>;
using Hash = fss::hash::Aes128Ccr<Aes>;
HalfTreeDpf<8, Group, Hash, In> dpf{Hash{Aes{ctxs}}, instance_seed};
PackedHalfTreeDpf<8, 8, Hash, In> packed{Hash{Aes{ctxs}}, instance_seed};
```

`GrottoDcf` keeps its PRG backend and only changes to the explicit constructor.

## Drop removed template parameters

The seven schemes that took a trailing `par_depth` template parameter — `Dpf`, `Dcf`, `HalfTreeDpf`, `PackedHalfTreeDpf`, `GrottoDcf`, `Vdpf`, and `Dmpf` — lost it.
`Vdmpf` never had one.
Host full-domain and preprocessing operations take a nonnegative per-call `par_depth` argument instead, where 0 runs sequentially and a positive value bounds the OpenMP work partitions.
The v1.3.0 default of -1, which picked an automatic split from the thread count, has no equivalent.
The deprecated aggregate entries take no depth argument and always run sequentially; to run in parallel, migrate to a component entry and pass a positive per-call `par_depth`.

`Dmpf` and `Vdmpf` also lost `kappa`.
The Cuckoo candidate count is fixed at 3, and the template parameter after the input type is now `ch_lambda`.
Positional template arguments after the input type shift meaning: an explicit v1.3.0 `kappa` value of 3 in `Dmpf<..., In, 3>` now silently sets `ch_lambda = 3`.
The full v1.3.0 tails `Dmpf<..., In, 3, 80>` and `Dmpf<..., In, 3, 80, -1>` pass too many template arguments and no longer compile.
Check every explicit template argument after the input type when porting.

## Own the generation retry

In v1.3.0, `Vdpf::Gen` and `Dmpf::Gen` over keys retried a failed generation up to 128 times by deterministically perturbing the seeds, and wrote the effective seeds back.
The v1.3.0 `Vdmpf::Gen` overload over mutable sigma and mutable seed pairs ran the same retry loop.
None of these retrying overloads is restored, including with an explicit `retry = false`; the restored `Vdmpf` aggregate-Key `Gen()` is the const-seed single-attempt form listed above.
Generation makes one attempt, returns a status, and never modifies the caller's seeds or sigma.
The caller draws fresh independent uniformly random values from a cryptographically secure generator and calls again.
Incrementing a seed word is not resampling.

The status codes and the value to resample for each failure:

- `Vdpf::Gen` returns 0 on success and 1 when the correction seed or the final control bits are unusable.
  Resample both initial seeds and retry.
- `Dmpf::Gen` returns 0 on success and 1 when Cuckoo placement fails.
  Resample sigma only, because placement does not depend on the inner DPF seeds.
- `Vdmpf::Gen` returns 0 on success, 1 when Cuckoo placement fails, and 2 when an inner VDPF generation fails.
  On 1 resample sigma only.
  On 2 keep sigma and resample every bucket seed pair.
  The restored const-seed aggregate-Key `Gen()` reports both 1 and 2 as the old code 1.

The component `Gen()` forms also changed their seed layout and ownership.
The fences below are schematic: `SampleRandom` stands for a helper you write over a cryptographically secure random generator, and the scheme object, input points, and payload buffers are the ones your program already holds.
`Vdpf::Gen` takes read-only seeds and outputs the packed control bits and the correction seed alongside the correction words:

```cpp
// Now
int4 cws[9];  // in_bits layers plus the terminal
In tr;
cuda::std::array<int4, 4> cs;
int4 s0s[2];
for (;;) {
  SampleRandom(s0s);  // fresh independent seeds per attempt
  if (vdpf.Gen(cuda::std::span<int4>(cws, 9), tr, cuda::std::span<int4, 4>(cs),
      cuda::std::span<const int4, 2>(s0s, 2), alpha, beta) == 0) {
    break;
  }
}
// Party b keeps s0s[b]; cws, tr, and cs are shared key material.
```

`Dmpf::Gen` and `Vdmpf::Gen` take a flat party-major seed array instead of the v1.3.0 bucket-major `array<int4, 2>[m]`: entry `i` is party 0's seed for bucket `i` and entry `kBucketCount + i` is party 1's.
They fill shared correction-word buffers instead of key structs, and return the runtime bucket count and bucket size through output parameters:

```cpp
// Now
constexpr int kBucketBits = 12;
using DmpfT = Dmpf<16, 100, kBucketBits, Group, Prg, Prp, In>;
DmpfT dmpf{prg, prp};
std::vector<DmpfT::Cw> cws(DmpfT::kBucketCount * (kBucketBits + 1));
std::vector<int4> s0s(2 * DmpfT::kBucketCount);  // party-major
int m_rt;
__uint128_t b_size_rt;
int4 sigma;
SampleRandom(sigma);
SampleRandom(s0s.data(), s0s.size());
for (;;) {
  if (dmpf.Gen(cws, m_rt, b_size_rt, sigma, s0s, as, b_bufs, t) == 0) {
    break;
  }
  SampleRandom(sigma);  // placement failure: resample sigma only
}
```

Party `b` keeps the entries `s0s[b * kBucketCount .. (b + 1) * kBucketCount)`, and `cws`, `sigma`, `m_rt`, and `b_size_rt` are the shared key material that `BatchEval` and `EvalAll` take.
The restored aggregate-Key `BatchEval` and `EvalAll` accept keys assembled in the current `Key` layout.

## Multi-point keys store wider data

`Dmpf::Key` and `Vdmpf::Key` keep their names but change their storage:

- `b_size_rt` is `__uint128_t` instead of `int`.
  Check the range before narrowing it into an `int`.
- A `Vdmpf::BucketKey` stores `bucket_bits + 1` correction words.
  The terminal that was the separate `ocw` field is now `cws[bucket_bits].s`, and the correction seed is a plain `int4 cs[4]` instead of a `cuda::std::array<int4, 4>`.
  `bk.ocw` and whole-array `bk.cs = ...` assignments no longer compile.
- `Dmpf::InnerDpf` and `Vdmpf::InnerVdpf` are private, and their input type is `__uint128_t`.
  Use the `BucketKey` and `Cw` nested names, which describe the current storage.
- Field order, padding, and binary serialization of keys are not specified.
  Do not reinterpret old persisted key bytes as the current layout; regenerate keys instead.

`PackedHalfTreeDpf::PackBeta` is private now.
Build the packed beta block yourself: the beta of an input goes into the lane indexed by the low `log2(kLanes)` bits of that input, exactly as `Extract` reads it back.
The public `Extract` keeps its v1.3.0 declaration.

## Move wide groups to Uint128

`fss::group::Uint<T, mod>` with `T` from `uint8_t` through `uint64_t` is the primary form.
Use `fss::group::Uint128<fss::group::Uint128Modulus{.low = ..., .high = ...}>` for wide modular groups.
The default `Uint128Modulus{0, uint64_t{1} << 63}` is 2^127, matching the v1.3.0 `Uint<__uint128_t, (uint128_t)1 << 127>` group.

The restored compatibility spelling `Uint<__uint128_t, mod>` covers `0 < mod <= 2^64 - 1` only.
Wider v1.3.0 moduli do not compile and must move to `Uint128`.
The boundary is deliberate: the v1.3.0 wide type compiled in ordinary host code, and its `static_assert(Groupable<...>)` passed, but on the probed CUDA 12.6/GCC 11.4 and 13.4/GCC 13.3 toolchains a generic kernel launch that takes such a type as a template argument fails to compile, because the generated host stub respells the modulus as one bare decimal literal that GCC reads back as zero.
`Uint128` spells the modulus as two `uint64_t` fields and is the stable form for kernel arguments.

Naming a restored `Uint<__uint128_t, mod>` with `mod >= 2^63` draws an integer-literal warning from the CUDA 12.6/GCC 11.4 host pass.
The value parses correctly, the warning is informational, and it is not a truncation.
The public `val` fields of `Uint` and `Bytes` were removed.
Read a value through `Uint::Value()` or `Bytes::Into()`, and build values with `From`.

## Update primitive backends

- `ChaCha` lost its non-owning pointer-nonce constructor.
  `ChaCha(nonce0, nonce1)` copies the nonce values, so the scheme no longer observes later writes to an external nonce array.
  `rounds` must now be positive in addition to even.
- `Aes128MmoRaw::Gen8`, `Vdpf::Prove`, and `Vdpf::ProveMerkle` take `cuda::std::span` parameters where v1.3.0 took array references.
  Ordinary calls with a `cuda::std::array` lvalue still compile through the implicit conversion.
  Exact member-function-pointer types change.
  `ProveMerkle` also gained a trailing `int par_depth = 0` argument and now throws `std::invalid_argument` on empty input instead of asserting.
- The host-only backends `Aes128Mmo`, `Aes128MmoRaw`, `Sha256`, `Aes128Feistel`, and `Aes128FeistelRaw`, and `Aes128Ccr` over an OpenSSL or AES-NI backend, fail device use at link time with an undefined-symbol error naming `fss_host_only_<Class>_used_in_device_code`.
  v1.3.0 trapped at runtime.
  Use `Aes128Soft` or `Aes128Ccr<Aes128Soft<1>>` for device-side hashing.
- OpenSSL failures now throw `std::runtime_error` instead of asserting, and failed context setup releases its resources.

## Check errors where the library now throws

Host boundary entries validate buffer lengths, ranges, bucket sizes, and point counts, and throw `std::invalid_argument` in Debug and Release builds.
Low-level requirements, such as the split Gen and Eval layouts, terminal group values, and the zero LSB that `Bytes::From` expects, remain caller preconditions: Debug builds abort with a diagnostic on the host and the device, and Release builds have undefined behavior.
Not every host call throws, so keep checking preconditions near the calls that assume them.

## Move off the deprecated entries

The deprecated entries convert through compile-time-sized scratch buffers, so they copy more than the current API.
Migrate a call when convenient, and treat the deprecation message as the map to its replacement.
The single-point DPF pair looks like this:

```cpp
// v1.3.0, still compiles with the switch on
DpfT::Key k0, k1;
dpf.Gen(k0, k1, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
int4 y = dpf.Eval(false, k0, x);
```

```cpp
// Now
int4 cws[9];  // in_bits + 1: the tree layers plus the terminal
In tr = 0;
dpf.Gen(cuda::std::span<int4>(cws, 9), tr, cuda::std::span<const int4, 2>(s0s, 2), alpha, beta);
int4 y = dpf.Eval(false, s0s[0], cuda::std::span<const int4>(cws, 9), tr, x);
```

The correction array now includes the terminal at `cws[in_bits]`, and the per-layer control bits arrive packed in `tr`.
Each scheme's current usage is in `samples/`: `dpf_dcf_cpu.cu`, `half_tree_dpf_cpu.cu`, `packed_half_tree_dpf_cpu.cu`, `grotto_dcf_cpu.cu`, `vdpf_cpu.cu`, `dmpf_cpu.cu`, and `vdmpf_cpu.cu`, with GPU usage in `dpf_dcf_gpu.cu`.
