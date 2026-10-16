# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Merkle-fold proof accumulation for VDMPF (`BatchEvalMerkle`, GPU `VdmpfBatchEvalGpu`): extends
  the VDPF fold to the multi-point scheme across the Cuckoo buckets. Per bucket the corrected
  per-point hashes fold pairwise in fixed 256-leaf chunks (odd trailing node carries), the chunk
  roots combine from the bucket's hashed cs, and the m bucket nodes fold into the 64B proof.
  `Gen`, `Verify`, and the output shares are unchanged; the paper-faithful chains `BatchEval`
  remain the reference, and both parties must pick the same accumulator. The GPU path builds the
  canonical per-bucket order from Locate records plus a stable radix sort, then runs the
  evaluation and fold passes fully parallel, bit-exact with the host fold; security analysis
  extended in doc/vdpf_merkle_fold_security.md.
* `Aes128MmoRaw::Gen8`: the AES-NI MMO backend applies its round-key schedules to eight
  seeds at once through the classic eight-stream `aesenc` interleave, bit-exact with eight
  `Gen` calls. The internal `fss::detail::GenBatch` helper exposes the batch to callers:
  it forwards to `Gen8` on PRGs that provide one and loops `Gen` otherwise.
* GPU benchmark curves for DMPF batch `Eval` and `EvalAll`: the GPU Cuckoo `Locate` runs on the
  `ChaChaFeistel` PRP while the CPU curves keep OpenSSL-backed `Aes128Feistel` — each platform's
  own optimal backend, and the figures compare implementations within one platform. The refreshed
  `gpu-operations` grid gains the DMPF / VDMPF row (Gen cell empty: DMPF Gen stays host-side).
  `figure_bench.py --scheme` restricts a native collection to one scheme, which collected the new
  curves without re-running the existing GPU rows. Also restores the `bench_fss` build, which the
  span-taking public-API refactor had broken (stale `std::span` call sites and a missing `Cw`
  template argument in the occupancy check).
* GPU evaluation for DMPF: `DmpfBatchEvalGpu` (one thread per input point runs its kappa
  Cuckoo `Locate` calls and the located bucket descents itself, so no atomics or synchronization;
  level-major bucket keys from `DpfRepackGpu` serve both parties) and `DmpfEvalAllGpu`
  (full-domain-evaluates every active bucket key through `DpfEvalAllGpuBatch` into caller scratch,
  then one gather thread per domain point). `PrpHash::Locate` is now `__host__ __device__` and
  returns a plain `LocateResult` aggregate (std::pair's constructor is host-only under nvcc), so
  the ChaChaFeistel PRP serves host and device from one code path. The phase-1 tiling defaults
  adapt to small bucket domains. Bit-exact against the host `BatchEval`/`EvalAll` on both parties,
  tested with the Bytes and Uint127 groups.
* `ChaChaFeistel` small-domain PRP backend: the 4-round Feistel + cycle-walking construction
  of `Aes128Feistel` with the round PRF replaced by the keyed ChaCha block function the
  library PRG already uses (round key in the ChaCha key, the <=64-bit Feistel half-word in
  the nonce slot), so no new assumption is introduced. Stateless and OpenSSL-free: `Permu`
  is `__host__ __device__` and instances are trivially copyable, preparing a device-side
  Cuckoo `Locate` for DMPF/VDMPF. Tests cover bijection (exhaustive to 2^20 plus a sampled
  128-bit domain), determinism, seed variation, bucket balance, and DMPF + Cuckoo `Compact`
  end-to-end on the new backend.
* Five third-party benchmark baselines in `third_party/bench.py`: the Mastic and PLASMA
  verifiable-incremental DPF prototypes (Rust wrappers over the pinned submodules; Gen,
  proof-chained Eval, and for Mastic a both-party full-domain `eval_tree` whose timing
  includes the BLAKE3 digest comparison), sycret's equality/comparison keygen and batched
  point eval through its C ABI (batch sizes 1024 to 65536), the dkales/dpf-cpp DPF entry
  (Gen, Eval, and the `EvalFull`/`EvalFull8` full-domain paths), and dpfplusplus fetched
  and patched at bench time — its anonymous pre-release headers forbid redistribution, so
  nothing is vendored and `doc/bench-dpfplusplus.md` documents provenance, license status,
  the three scripted fixes, and reproduction. The runner parses per-operation domain and
  batch suffixes from the new benchmark IDs for correct work-unit accounting, smoke
  filters match suffixed criterion IDs, and `figure_bench.py` sweeps the four domain-swept
  additions with per-library operation and domain guards (sycret stays `bench.py`-only).
* GPU point evaluation now covers the same input-domain widths as the host `Eval`, through
  `__uint128_t`: the query-point kernels take `In`-typed points (`uint8_t` to `__uint128_t`),
  and the per-level tr bits packed by the repack kernels widen to `uint64_t` or `__uint128_t`
  once `in_bits` passes 32 or 64 (`fss::gpu::ExtBits`, `uint32_t` layout unchanged for common
  `in_bits <= 32` keys). The DCF repack and point-evaluation wrappers also take the
  predicate of the `Dcf` instance (`lt` default, `gt` explicit). Domains beyond 32 bits
  used to silently wrap the tr-bit word. New
  `point_eval_gpu_test` suite covers repack + point evaluation of every scheme against the
  host `Eval` oracle, including the VDPF per-point proof and `Verify`, the `in_bits` 32/33
  tr-bit boundary, and a wide-`In` (`__uint128_t`, n=2^10) GPU `EvalAllMerkle` smoke test.
* Python bindings for every scheme: `HalfTreeDpf`, `PackedHalfTreeDpf` (with a pure-Python
  lane `extract`), `GrottoDcf` (preprocess and full-domain evaluation on CPU or CUDA),
  `Vdpf` (chain and Merkle accumulators, device-side Merkle full-domain evaluation with
  the proof, `prove`/`prove_merkle`/`verify`), and the CPU-only `Dmpf`/`Vdmpf` with their
  per-bucket key tensors and batch proofs. `Dpf`/`Dcf` `eval_all` now accepts CUDA tensors
  with either DCF comparison predicate and routes to the frontier full-domain kernels on
  the current torch stream, and every CUDA path launches on the current stream and returns
  asynchronously. The JIT layer is registry-driven and compiles one module per scheme
  configuration with full input validation.
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
  `DcfEvalAllGpuBatch` (frontier kernel carrying the DCF group state, for both `lt` and `gt`
  predicates), `PackedHalfTreeDpfEvalAllGpu`
  and `PackedHalfTreeDpfEvalAllGpuBatch` (frontier stops at `kDepth`, one packed block per leaf
  node), and `GrottoDcfPreprocessGpu` and `GrottoDcfEvalAllGpu` (parity segment tree built per
  level plus a CUB inclusive XOR scan). All bit-exact against the CPU schemes on both parties;
  measured 102x to 4693x over the same-host CPU at 32 threads at n=2^20 on an RTX A6000 (see the
  benchmark comparison site for the collected curves).
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

* Renamed the GPU point-eval key-preparation entry points from `*RelayoutGpu` to `*RepackGpu`
  (and the internal `*RelayoutKernel` to `*RepackKernel`): the operation repacks per-key
  correction words into the level-major layout and extracts the tr bits and output correction
  word, which "repack" names more directly than "relayout". Mechanical rename, no behavior
  change.
* **Breaking** (0.x): `ChaCha` stores its nonce by value (`ChaCha(int, int)`); the pointer
  constructor and the host/device dual-instance pattern are gone — one instance built
  anywhere works on both sides.
* **Breaking** (0.x): every scheme exposes a `Key` aggregate (correction words plus the
  party seed, plus `ocw`/`fcw`/`cs` where the scheme has them); `Gen` fills one `Key` per
  party, `Eval`/`EvalAll` take the party's `Key`, all public array parameters are
  `cuda::std::span`, `Vdpf::Eval` returns `{y, pi_tilde}`, and `GrottoDcf::Preprocess`
  returns an owning `PreprocessedTree` (the non-owning `ParityTree` view remains for
  device-side point evaluation).
* **Breaking** (0.x): `Vdpf`/`Dmpf`/`Vdmpf` `Gen` retries internally by default
  (`bool retry = true`) and writes the effective seeds and sigma back; `retry = false`
  restores the fail-fast return.
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

* The grid figures drew a rotated y-axis label on every panel, reaching past the 34px column
  gap into the neighboring plot. Each row (and block-size band) now carries one label on its
  leftmost existing panel, and a panel whose time unit differs from that label notes its unit
  in the panel caption (`(ms/key)`, `(ns/key)`).
* The Python bindings' CUDA paths no longer dereference device tensors on the host when
  reading seeds and check values for kernel launches (the single-point paths passed the
  seed by value after a host read, which faults over PCIe; seeds now travel as kernel
  argument pointers and scalars through explicit device-to-host copies).
* The supplementary block-size figure wraps its panels across two-column bands instead of
  rendering every panel beyond the second outside the SVG canvas.
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

### Changed

* The packed Half-Tree CPU `EvalAll` walks a level-order frontier below three scalar
  levels, expanding eight parents per batched expansion and fusing a branchless final
  conversion; outputs are bit-exact with the previous depth-first walk and the 2^20
  full-domain evaluation moves from 4.8 to 16.5-32.9 G outputs/s.
* The DPF and Half-Tree CPU `EvalAll` paths grow the same batched frontier for every PRG:
  one batched expansion per eight nodes per level with branchless corrections, 64-leaf
  conversion groups for the DPF, and a fused two-sigma-set last level for the Half-Tree;
  `fss::detail::GenBatch` pipelines the batch through `Gen8` when the PRG offers one and
  loops `Gen` otherwise. At 2^20 the DPF bytes evaluation drops from 34 to 12 ms, the
  integer group from 22 to 7.6 ms, and the Half-Tree from 20 to 5 ms, bit-exact before
  and after.

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
