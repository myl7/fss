# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Chart merges can drop a library-platform combination whose rows would otherwise survive every refresh
- Endpoint CI matrix covering the oldest and newest CUDA toolkits and Python/torch pairs
- README figures render on the Doxygen pages
- Record of the attempt to replace the VDPF/VDMPF proof hash with an AES or ChaCha construction (doc/vdpf_proof_hash_exploration.md): the constructions were reviewed and measured, both approved candidates lost the performance comparison, and Blake3 stays

### Changed

- Added aggregate `Key` APIs, owning Grotto `PreprocessedTree`, and owning ChaCha nonce construction alongside the v1.2.0 raw-buffer and pointer APIs
- Mutable generation APIs can retry and return effective seeds; released const-input VDPF and VDMPF calls retain fail-fast behavior. The mutable VDMPF overload requires explicit retry arguments
- Added `*RepackGpu` preparation names alongside the released `*RelayoutGpu` entry points
- The DPF, Half-Tree, and DCF CPU full-domain evaluations walk a batched frontier, honoring `par_depth` throughout
- `par_depth=0` executes on the calling thread; positive depths bound work partitions. DMPF forwards the option to its inner DPF
- The `fss-crypto` package requires Python >= 3.10
- Historical FSS curves moved from the figures to the comparison site, and the GPU benchmark curves moved to the cs11369a reference host
- Benchmark figures separate five scheme families and prepared-key, proof, PRP, and output-group comparisons, with distinct labels and styles
- Benchmark records distinguish independent keys, single-key query batches, and logical domain outputs; setup preparation is measured separately

### Fixed

- The Python bindings' CUDA paths no longer dereference device seed tensors on the host before kernel launches
- Python CUDA scalar and metadata copies follow the current PyTorch stream before evaluation
- Restored v1.2.0 overloads, VDMPF `BucketKey` layout, legacy GPU query strides, and deferred ChaCha nonce-pointer reads
- Configured CPU and GPU helpers accept nondefault `par_depth` and Cuckoo security parameters
- Preserve DPF parallel-root initialization on affected GCC 12 toolchains
- Python AES uses the OpenSSL MMO implementation with independent thread-local contexts; JIT cache fingerprints include source, compiler, ABI, architecture, and options
- Python backend and domain validation rejects unsupported device combinations, invalid queries, oversized bucket domains, and unsafe full-domain allocations before evaluation
- Reject overflowing DMPF and VDMPF compile-time bucket capacities before multiplication
- Complete 64-bit beta handling and defined unsigned retry increments
- Define ChaCha and BLAKE3 round additions as unsigned arithmetic modulo 2^32
- Preserve all 64/128-bit scalar input bits across Python's signed native argument boundary
- GPU scratch allocations have per-device ownership and explicit release; empty multi-point batches preserve proof semantics without invalid kernel launches
- Installed CMake packages discover their OpenMP dependency, and Python wheels include internal public-header dependencies
- `Vdmpf::BatchEval` no longer scans every collected record for duplicates, a quadratic pass the Cuckoo PRP's permutation property can never trigger (`Dmpf::BatchEval` carried the same scan)

### Removed

- Repository pre-commit hook

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

[unreleased]: https://github.com/myl7/fss/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/myl7/fss/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/myl7/fss/releases/tag/v1.1.0
[1.0.0]: https://github.com/myl7/fss/releases/tag/v1.0.0
