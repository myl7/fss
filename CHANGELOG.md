# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Distributed multi-point function (DMPF)
- Merkle-fold proof accumulation for VDPF (`ProveMerkle`, `EvalAllMerkle`, GPU `VdpfEvalAllMerkleGpu`) and for VDMPF (`BatchEvalMerkle`, GPU `VdmpfBatchEvalGpu`), with the security analysis in doc/vdpf_merkle_fold_security.md
- Packed-output DPF (`PackedHalfTreeDpf`, experimental): Half-Tree expansion with a domain-separated leaf conversion and a batched frontier walk
- Packed point-key CPU fast paths for DPF and Half-Tree point evaluation
- GPU full-domain evaluation for DCF, Grotto DCF, packed HalfTree DPF, and VDPF, and GPU batch and full-domain evaluation for DMPF
- GPU point evaluation over the full input-domain range, for both DCF predicates
- ChaChaFeistel small-domain PRP backend, host and device
- 8-way AES-NI batch generation for the raw MMO PRG (`Gen8`), with an internal `GenBatch` fallback for PRGs without one
- AES-NI raw Feistel PRP backend (`Aes128FeistelRaw`)
- Python bindings for every scheme, including CUDA evaluation and device-side VDPF Merkle proofs
- Static benchmark comparison site (doc/site), scheme-family comparison figures, and benchmark coverage page (doc/benchmarks.md)
- CPU benchmark curves for the newer schemes, with the Servan-Schreiber VDPF reference and the Mastic, PLASMA, sycret, dpf-cpp, and dpfplusplus (fetched on demand) baselines
- Chart merges can drop a library-platform combination whose rows would otherwise survive every refresh
- Endpoint CI matrix covering the oldest and newest CUDA toolkits and Python/torch pairs
- README figures render on the Doxygen pages
- Record of the attempt to replace the VDPF/VDMPF proof hash with an AES or ChaCha construction (doc/vdpf_proof_hash_exploration.md): the constructions were reviewed and measured, both approved candidates lost the performance comparison, and Blake3 stays

### Changed

- **Breaking** (0.x): scheme keys are `Key` aggregates; `Gen` fills one per party, `Eval` and `EvalAll` take the party's key, public array parameters are `cuda::std::span`, `Vdpf::Eval` returns `{y, pi_tilde}`, and `GrottoDcf::Preprocess` returns an owning `PreprocessedTree`
- **Breaking** (0.x): `ChaCha` stores its nonce by value; the pointer constructor and the host/device dual-instance pattern are gone
- **Breaking** (0.x): `Vdpf`, `Dmpf`, and `Vdmpf` `Gen` retries internally by default and writes the effective seeds and sigma back; `retry = false` restores the fail-fast return
- Renamed the GPU point-eval key-preparation entry points from `*RelayoutGpu` to `*RepackGpu`
- The DPF, Half-Tree, and DCF CPU full-domain evaluations walk a batched frontier, honoring `par_depth` throughout
- The `fss-crypto` package requires Python >= 3.10
- Historical FSS curves moved from the figures to the comparison site, and the GPU benchmark curves moved to the cs11369a reference host

### Fixed

- The Python bindings' CUDA paths no longer dereference device seed tensors on the host before kernel launches
- `Vdmpf::BatchEval` no longer scans every collected record for duplicates, a quadratic pass the Cuckoo PRP's permutation property can never trigger (`Dmpf::BatchEval` carried the same scan)

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
