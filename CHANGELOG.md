# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Merkle-fold proof accumulation for VDPF (`ProveMerkle`, `EvalAllMerkle`,
  `VdpfEvalAllMerkleGpu`): an order-independent alternative to the paper's sequential Prove chain
  that hashes the corrected per-point values into the leaves of a binary Merkle tree and folds
  adjacent nodes level by level. `Gen`, `Eval`, and `Verify` are unchanged and the proof keeps the
  64B shape, so both parties only need to switch accumulators jointly. The GPU path computes the
  whole proof on device (748 us vs the sequential chain's 185 ms host-bound pipeline at n=2^20) and
  the CPU `EvalAllMerkle` phase 2 is OpenMP-parallel, removing the chain's 8-thread scaling cap.
  The paper-faithful chain `Prove`/`EvalAll` remain implemented as the reference. Security
  analysis in doc/vdpf_merkle_fold_security.md (soundness reduces to pairwise collision
  resistance of the fold hash; documents the `cs != 0` key-validation assumption that the
  library's control-bit bookkeeping needs, with a matching Gen retry). Fixed the eprint citation
  of the VDPF/VDMPF scheme in the vdpf/cuckoo_hash/vdmpf headers (2024/677 -> 2021/580).
* GPU full-domain evaluation for the schemes that had no GPU `EvalAll`: `DcfEvalAllGpu` and
  `DcfEvalAllGpuBatch` (frontier kernel carrying the DCF group state), `PackedHalfTreeDpfEvalAllGpu`
  and `PackedHalfTreeDpfEvalAllGpuBatch` (frontier stops at `kDepth`, one packed block per leaf
  node), and `GrottoDcfPreprocessGpu` and `GrottoDcfEvalAllGpu` (parity segment tree built per
  level plus a CUB inclusive XOR scan). All bit-exact against the CPU schemes on both parties;
  measured 102x to 4693x over the same-host CPU at 32 threads at n=2^20 on an RTX A6000 (see the
  README GPU results).
* CPU `EvalAll` benchmarks with the ChaCha PRG for DCF, VDPF (with BLAKE3), packed HalfTree DPF,
  and Grotto DCF, for same-PRG comparison against the GPU benchmarks.
* Distributed multi-point function (DMPF), the non-verifiable counterpart of VDMPF.
* Packed-output DPF (`PackedHalfTreeDpf`, experimental): based on Half-Tree expansion with a
  domain-separated leaf conversion, `EvalAll` returns `2^(in_bits-log2(128/out_bits))` blocks that
  each pack `128/out_bits` consecutive XOR-shared outputs of `out_bits` bits,
  `out_bits ∈ {1, 2, 4, 8, 16, 32, 64}`. No published paper describes the composition.
* CPU benchmark curves for HalfTree DPF, packed HalfTree DPF (one-bit), Grotto DCF, DMPF, and
  VDMPF (`t = 64` points). Each platform comparison figure is now a scheme-family grid — one
  row each for DCF (with Grotto DCF), DPF with its verifiable VDPF counterparts, and
  DMPF/VDMPF, one column each for `Gen`, `Eval`, and `EvalAll` — with one compact legend per
  family row. The threads-per-block sweep moved from the README to the benchmark methodology
  page.
* Static benchmark comparison site (`doc/site`, deployed under `/bench/`): pick two
  implementations and compare median per-key times over the domain sweep `N = 2^8 … 2^20`,
  with exact per-domain values, ratios, native configurations, and provenance. It replaces
  the per-benchmark result tables in the README.
* Benchmark coverage page (`doc/benchmarks.md`): what the `bench_cpu`/`bench_gpu` binaries
  measure, the benchmark naming scheme, and the GPU kernel register usage table moved out
  of the README.
* Endpoint CI matrix: the C++/CUDA core builds and tests on the oldest and newest CUDA
  toolkits available for the runner (12.6 and 13.4), and the Python bindings test the
  oldest and newest supported pairs (Python 3.10 with torch 2.6/cu126, Python 3.13 with
  torch 2.13/cu131) in separate jobs.
* VDPF CPU curves on the DPF comparison axes: FSS VDPF beside the Servan-Schreiber VDPF
  reference implementation (sachaservan/vdpf, eprint 2021/580, pinned submodule with a
  benchmark adapter). Both include proof generation in the timed evaluation. Verification is
  an equality check outside timing.

### Changed

* The comparison figures show current implementations: the historical FSS 0.6.0 and
  0.7.0 curves moved to the benchmark comparison site (with full provenance), and the
  figure notes state that an implementation appears in a panel only where it provides
  that method on that platform.
* The GPU benchmark curves moved to a new GPU reference host: `cs11369a`
  (Intel Xeon Platinum 8352V, NVIDIA A6000, SM 86, CUDA 12.6, GPU 0, launch pinned
  to CPU 100, shared-host CPU governor left unchanged). CPU curves stay on `cs659b`.
  All GPU libraries were re-collected there; the historical FSS 0.7 has no GPU curves
  anymore because its upstream benchmark crashes on SM 86 (misaligned address), and
  the merge records the dropped rows with that reason instead of keeping SM 120
  measurements. The GPU register table was re-collected for sm_86 and the block-size
  figure now covers the `EvalAll` schemes and point `Eval` of the whole GPU scheme set.
* The `fss-crypto` package requires Python >= 3.10, the bindings' actual floor
  (`X | Y` annotations), instead of the development interpreter's 3.13. The torch floor
  stays 2.6.

### Fixed

* The packed GPU `EvalAll` benchmark compiles on domains whose frontier is smaller than
  the configured block (`n = 2^8 … 2^12`): those launches now use one block of `2^z`
  threads instead of a negative block-root depth, and the VDPF benchmark is excluded by
  preprocessor when the block size admits no feasible tiling, so block sweeps build
  everywhere.
* Curve collection accepts the shrunken packed block size instead of failing the whole
  domain case, and chart merges can drop a library-platform combination
  (`--drop library:platform:reason`) whose rows would otherwise survive every refresh.

* README figures now render on the Doxygen GitHub Pages site: the Doxygen input filter
  rewrites figures-directory Markdown images to `\image` commands resolved through
  `IMAGE_PATH`, so the SVGs are copied into the HTML output with a responsive width.
* Comparison figures label narrow logarithmic time ranges with minor log ticks. Nearly
  flat curves (such as DMPF point evaluation) previously rendered with as few as two
  axis ticks, which hid the logarithmic scale.

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
