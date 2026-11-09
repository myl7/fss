# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `fss::group::Uint128<mod>`, the clamped 127-bit modular integer group in 128-bit storage. The modulus is the structural `fss::group::Uint128Modulus{low, high}` non-type template parameter (`(high << 64) | low`, default `2^127`), so the type is usable directly as a CUDA kernel template argument.
- A v1.3.0 compatibility layer behind the numeric macro `FSS_ENABLE_V1_COMPAT`, on by default: the v1.3.0 aggregate Cw-form and aggregate-Key entries, nested types, legacy constants, and `Uint<__uint128_t, mod>` spelling stay available as deprecated declarations that convert to the current component API, without the automatic-retry Key generation overloads. Define the macro to the same value in every translation unit of a program; 0 leaves the current API only. doc/migration.md covers the full scope and the breaks that remain with the switch on.
- Closed-interval host evaluation for DPF, DCF, `HalfTreeDpf`, `PackedHalfTreeDpf`, and VDPF, with CUDA range evaluation for DCF.
- VDPF sparse-batch and range Merkle accumulation on CPU and CUDA.
- Single-path Grotto DCF evaluation without preprocessing, and the experimental `HalfTreeGrottoDcf` scheme with Python bindings.
- The `Aes128Ccr` CCR-hash backend, which implements the CCR-hash contract that `HalfTreeDpf` and `PackedHalfTreeDpf` now require.
- Public optional primitive extensions through `HasGenHalf`, `HasGen8`, and `HasHash4`, with scalar fallbacks.
- DMPF and VDMPF accept `bucket_bits` up to 127 and run their inner schemes on `__uint128_t` inputs, widening the supported per-bucket domains.
- Standalone Python Group, PRG, hash, and PRP operations, and explicit primitive configuration for schemes.
- Python precompilation and scoped compiled-plugin caches, including read-only cache reuse.

### Changed

- Scheme operations take caller-owned component buffers. Correction arrays include the terminal data, a positive stride parameter supports batched correction storage, and host full-domain and preprocessing operations take the OpenMP split depth as a per-call nonnegative `par_depth` argument.
- Scheme classes take their backends through an `explicit` constructor and keep them private. The v1.3.0 public backend fields (`prg`, `xor_hash`, `hash`, `prp`, and `hash_key`) are gone, so code that default-constructed a scheme or reassigned a backend field constructs a configured scheme instead.
- `HalfTreeDpf` and `PackedHalfTreeDpf` run on a CCR-hash backend satisfying `CcrHashable` instead of a PRG backend satisfying `Prgable`, configured together with the instance seed. The v1.3.0 PRG backends do not satisfy the new contract.
- VDPF, DMPF, and VDMPF generation makes a single attempt, never rewrites the caller's seeds or sigma, and leaves retrying to the caller, who checks the returned status and resamples the relevant randomness before calling again. The component `Vdmpf::Gen` reports a new status 2 when an inner VDPF generation fails; the deprecated aggregate-Key `Vdmpf::Gen()` maps both statuses to the old failure code.
- DMPF and VDMPF keys keep their names but store wider data: `b_size_rt` is now `__uint128_t`, and a VDMPF `BucketKey` carries `bucket_bits + 1` correction words with the terminal merged into the last entry and the correction seed as `int4 cs[4]`, so the separate `ocw` field is gone.
- Grotto preprocessing uses explicit parity-tree views and owned trees whose `View()` methods expose borrowed storage.
- Host boundary entries report `std::invalid_argument` for buffer, range, bucket, and count metadata in Debug and Release builds. Low-level algorithms, including split Gen/Eval layouts, terminal group values, PRP input domains, `Bytes::From`, and `Vdmpf::Eval`, treat these requirements as caller preconditions: Debug builds abort with a diagnostic on both the host and the device, and Release builds have undefined behavior. Terminal validation runs before parallel traversal.
- VDPF and VDMPF evaluation and proof entry points document nonzero correction seeds as a precondition; keys from a successful `Gen()` satisfy it.
- Host-only primitive backends, including the AES MMO PRGs, `Sha256`, and the AES Feistel PRPs, fail device use at link time through undefined device symbols instead of trapping at runtime.
- OpenSSL backend failures report `std::runtime_error`, and failed context setup releases allocated resources.
- Python single-point schemes store keys as opaque component bytes: 16-byte correction-component rows plus a packed-control row (DCF adds the value-component rows, changing its row count). Previously persisted native key bytes are not compatible and must be regenerated.
- Python scheme methods fill caller-owned tensors and select CPU or CUDA from their tensor arguments. Mixed devices are rejected instead of copied automatically.
- Python packaging requires setuptools >= 84.1.0, filelock == 3.20.3 on Python 3.10, and filelock >= 4.2.1 on newer Python versions.

### Deprecated

- Under the default `FSS_ENABLE_V1_COMPAT`, the v1.3.0 aggregate Cw-form and aggregate-Key entries of `Dpf`, `Dcf`, `HalfTreeDpf`, `PackedHalfTreeDpf`, `GrottoDcf`, `Vdpf`, `Dmpf`, and `Vdmpf`, together with their v1.3.0 nested types (the aggregate `Key`, `Vdpf::EvalResult`, the Grotto `ParityTree` and `PreprocessedTree`) and the legacy constants `m`, `n`, `b_size`, `kBetaMax`, `kTopDepth`, `kMerkleChunkSize`, and `kParDepth`. Every entry carries the standard deprecation attribute naming its current replacement; doc/migration.md lists the full scope.
- The wide-storage spelling `Uint<__uint128_t, mod>` for `0 < mod <= 2^64 - 1`; use `fss::group::Uint128` instead.

### Fixed

- Preserved every coordinate bit in Blake3 and Sha256 tuple hashing used by VDPF and VDMPF, fixing the loss of bit 96 for 97- to 128-bit coordinates. The domain-separation tag also moved, so keys and proofs generated by v1.3.0 are not compatible with this version. Regenerate persisted key and proof material.
- Copy assignment into a moved-from `Aes128Feistel` now recreates its cipher context and restores the object to a usable state.

### Removed

- The trailing `par_depth` template parameter of DPF, DCF, Half-Tree DPF, packed Half-Tree DPF, Grotto DCF, VDPF, and DMPF, and the `kappa` template parameter of DMPF and VDMPF. Host operations take a nonnegative per-call `par_depth`, and the Cuckoo candidate count is fixed at 3, so positional template arguments after the input type shift meaning or exceed the new parameter list.
- The public `InnerDpf` and `InnerVdpf` aliases of DMPF and VDMPF and the public `PackBeta` helper of packed Half-Tree DPF. Use the current `BucketKey` and `Cw` names and pack betas by the `out_bits` lane rule.
- The public `val` fields of `Uint` and `Bytes`. Read a value through `Value()` or `Into()`.
- The non-owning pointer-nonce constructor of `ChaCha`. The constructor copies the nonce values.
- `Uint<__uint128_t, mod>` moduli above `2^64 - 1`. Use `fss::group::Uint128` for wide moduli such as the 2^127 group.
- Python `DmpfKey` and `VdmpfKey` wrappers.

Migration notes: [v1.3.0 to v2.0.0 (unreleased)](doc/migration.md#v130-to-v200-unreleased).

## [1.3.0] - 2026-10-18

### Added

- Distributed multi-point function (DMPF)
- Merkle-fold proof accumulation for VDPF (`ProveMerkle`, `EvalAllMerkle`, GPU `VdpfEvalAllMerkleGpu`) and for VDMPF (`BatchEvalMerkle`, GPU `VdmpfBatchEvalMerkleGpu`), with the security analysis in doc/vdpf_merkle_fold_security.md
- Packed-output DPF (`PackedHalfTreeDpf`, experimental): Half-Tree expansion with a domain-separated leaf conversion and a batched frontier walk
- Packed point-key CPU fast paths for DPF and Half-Tree point evaluation
- GPU full-domain evaluation for DCF, Grotto DCF, packed HalfTree DPF, and VDPF, and GPU batch and full-domain evaluation for DMPF
- GPU point evaluation over the full input-domain range, for both DCF predicates
- ChaChaFeistel small-domain PRP backend, host and device
- 8-way AES-NI batch generation for the raw MMO PRG (`Gen8`), with an internal `GenBatch` fallback for PRGs without one
- AES-NI raw Feistel PRP backend (`Aes128FeistelRaw`)
- Python bindings for every scheme, including CUDA evaluation and device-side VDPF Merkle proofs
- Keyword-only Python `par_depth` controls, explicit VDMPF `batch_eval_merkle`, and verification without constructing or compiling a scheme
- Static benchmark comparison site (doc/site), scheme-family comparison figures, and benchmark coverage page (doc/benchmarks.md)
- CPU benchmark curves for the newer schemes, with the Servan-Schreiber VDPF reference and the Mastic, PLASMA, sycret, dpf-cpp, and dpfplusplus (fetched on demand) baselines
- README figures render on the Doxygen pages
- Record of the attempt to replace the VDPF/VDMPF proof hash with an AES or ChaCha construction (doc/vdpf_proof_hash_exploration.md): the constructions were reviewed and measured, both approved candidates lost the performance comparison, and Blake3 stays

### Changed

- Added aggregate `Key` APIs, owning Grotto `PreprocessedTree`, and owning ChaCha nonce construction alongside the v1.2.0 raw-buffer and pointer APIs
- Mutable generation APIs can retry and return effective seeds; released const-input VDPF and VDMPF calls retain fail-fast behavior. The mutable VDMPF overload requires explicit retry arguments
- `Vdpf::Gen` also rejects an all-zero correction seed (`cs`) with the existing failure code, including through VDMPF's inner generation
- Added `*RepackGpu` preparation names alongside the released `*RelayoutGpu` entry points
- The DPF, Half-Tree, and DCF CPU full-domain evaluations walk a batched frontier, honoring `par_depth` throughout
- `par_depth=0` executes on the calling thread; positive depths bound work partitions. DMPF forwards the option to its inner DPF
- The `fss-crypto` package supports Python >= 3.10 instead of >= 3.13
- Historical FSS curves moved from the figures to the comparison site, and the GPU benchmark curves moved to the cs11369a reference host
- Benchmark figures separate five scheme families and prepared-key, proof, PRP, and output-group comparisons, with distinct labels and styles
- Benchmark records distinguish independent keys, single-key query batches, and logical domain outputs; setup preparation is measured separately

### Fixed

- The Python bindings' CUDA paths no longer dereference device seed tensors on the host before kernel launches
- Define ChaCha and BLAKE3 round additions as unsigned arithmetic modulo 2^32
- Preserve all 64/128-bit scalar input bits across Python's signed native argument boundary
- Python full-domain evaluation rejects domain sizes whose outputs would not fit before allocating
- Reject overflowing VDMPF bucket capacities before computing sizes
- Installed CMake packages discover their OpenMP dependency
- `Vdmpf::BatchEval` no longer scans every collected record for duplicates, a quadratic pass the Cuckoo PRP's permutation property can never trigger

Migration notes: [v1.2.0 to v1.3.0](doc/migration.md#v120-to-v130).

## [1.2.0] - 2026-08-17

### Added

- Verifiable distributed multi-point function (VDMPF)
- PRP interface, PRP-based Cuckoo hashing, and PRP based on AES-128 and Feistel networks
- Raw AES-128 MMO PRG
- Python bindings for DPF and DCF
- GPU full-domain evaluation kernels for DPF and Half-Tree DPF
- Level-major GPU point evaluation kernels for DPF, DCF, VDPF, and Half-Tree DPF

### Fixed

- GPU key generation for DPF, VDPF, and Half-Tree DPF wrote only part of each correction-word slot, leaving the padding undefined and forcing NVCC to emit a masked 256-bit store; the partial-sector write triggered a write-allocate round trip per level (~2x Gen time). Both 16B halves are now written explicitly, roughly halving GPU Gen time.

## [1.1.0] - 2026-03-10

### Added

- Verifiable DPF (VDPF).
- Hash interface with SHA-256 and BLAKE3 impl, used by VDPF.
- DPF based on Half-Tree.
- DCF based on Grotto.
- Soft AES-128 with Matyas-Meyer-Oseas, pre-initialized cipher contexts, and T-table in GPU shared memory as a PRG impl.

## [1.0.0] - 2026-03-05

[unreleased]: https://github.com/myl7/fss/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/myl7/fss/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/myl7/fss/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/myl7/fss/releases/tag/v1.1.0
[1.0.0]: https://github.com/myl7/fss/releases/tag/v1.0.0
