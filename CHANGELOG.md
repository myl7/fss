# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

* The `fss-crypto` package requires Python >= 3.10, the bindings' actual floor
  (`X | Y` annotations), instead of the development interpreter's 3.13. The torch floor
  stays 2.6.

### Fixed

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
