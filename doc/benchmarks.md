# Benchmark coverage

What the `bench_cpu` and `bench_gpu` binaries built by `BUILD_BENCH=ON`
measure, and the GPU kernel resource usage behind them. Build and run
instructions are in the README. Measurement methodology, timing boundaries,
and the cross-library comparison curves live in
[benchmark curve methodology](bench_figures.md). Exact measured values are on
the comparison site.

## Coverage

- Schemes: DPF, DCF, VDPF, Half-Tree DPF, packed HalfTree DPF, and Grotto DCF.
- Operations: `Gen`, `Eval`, host `EvalAll`, VDPF `Prove` and the Merkle-fold
  `ProveMerkle`/`EvalAllMerkle`, Grotto DCF `Preprocess` and `PreprocessEvalAll`,
  GPU point eval, and full-domain GPU `EvalAll` for DPF, HalfTree DPF, packed
  HalfTree DPF, DCF, and VDPF, plus GPU Grotto DCF `Preprocess` and `EvalAll`.
- PRGs: AES-128 MMO over OpenSSL, AES-128 MMO over AES-NI intrinsics,
  software AES-128 MMO, and ChaCha. The CPU benchmarks cover all four. The
  GPU benchmarks cover ChaCha and software AES-128 MMO, the two that run on
  device.
- Output groups: `Uint` and `Bytes`.
- VDPF hashes: SHA-256 and BLAKE3.
- Input domain sizes: 2^20 by default, plus 2^14 and 2^17 for DPF `Eval`.

Benchmark names follow `BM_<Scheme><Operation>_<Group>_<Prg>/<bits>`: for
example, `BM_DcfEval_Uint_AesRaw/17` times one DCF evaluation in the `Uint`
group with the AES-NI-intrinsics PRG over a 2^17 domain. The PRG suffixes are
`Aes` (OpenSSL MMO), `AesRaw` (AES-NI intrinsics MMO), `AesSoft` (software
AES-128 MMO), and `ChaCha`. Grotto DCF and VDPF benchmarks omit the group
because they have one native output shape. Names without a suffix are
template instantiations that differ in input domain bits through the
benchmark argument.

## GPU kernel register usage

Compiled for sm_86 with `--ptxas-options=-v` (`make ptx_info` collects the
log):

| Kernel               | Group | PRG             | Registers | Stack | Smem  |
| -------------------- | ----- | --------------- | --------- | ----- | ----- |
| DpfEval              | Uint  | `ChaCha<2>`     | 39 |      |       |
| DpfEval              | Bytes | `ChaCha<2>`     | 39 |      |       |
| DpfGen               | Uint  | `ChaCha<2>`     | 48 |      |       |
| DpfGen               | Bytes | `ChaCha<2>`     | 48 |      |       |
| DpfEval              | Uint  | `Aes128Soft<2>` | 72 |      | 1280B |
| DpfGen               | Uint  | `Aes128Soft<2>` | 72 |      | 1280B |
| HalfTreeDpfEval      | Uint  | `ChaCha<1>`     | 40 |      |       |
| HalfTreeDpfGen       | Uint  | `ChaCha<1>`     | 52 |      |       |
| VdpfEval             | Uint  | `ChaCha<2>`     | 40 |      |       |
| VdpfGen              | Uint  | `ChaCha<2>`     | 72 |      |       |
| DcfEval              | Uint  | `ChaCha<4>`     | 41 |      |       |
| DcfGen               | Uint  | `ChaCha<4>`     | 56 |      |       |
| DpfEvalPoint         | Uint  | `ChaCha<2>`     | 40 |      |       |
| DcfEvalPoint         | Uint  | `ChaCha<4>`     | 44 |      |       |
| VdpfEvalPoint        | Uint  | `ChaCha<2>`     | 46 |      |       |
| HalfTreeDpfEvalPoint | Uint  | `ChaCha<1>`     | 39 |      |       |
| DcfEvalAll           | Uint  | `ChaCha<4>`     | 124 |     | 6144B |
| PackedHalfTreeDpfEvalAll | -  | `ChaCha<1>`     | 47 |      | 4096B |
| VdpfEvalAllMerkle    | Uint  | `ChaCha<2>`     | 128 |      | 12288B |
| VdpfMerkleRoots      | -     | Blake3          | 40 |      | 8192B |
| GrottoDcfExpandLeafBits | -  | `ChaCha<2>`     | 63 |      | 4096B |
| HalfTreeDpfEvalAll (z=20) | Bytes | `ChaCha<1>`  | 105 |      | 2048B |
| PackedHalfTreeDpfEvalAll (z=17) | - | `ChaCha<1>` | 94 |      | 2048B |
| GrottoDcfExpandLeafBits (z=21) | - | `ChaCha<2>` | 61 | 256B | 2048B |

The PRG drives most of the difference. Software AES-128 MMO costs about 1.8x
the registers of ChaCha for the same scheme and group and is the only backend
here with a T-table in shared memory (see
[the software AES study](bench_aes128_soft.md)). The `mul` parameter is part
of the PRG type because it sets how many 16B blocks one `Gen` call produces.
All kernels shown have zero spill stores and zero spill loads.
The `(z=...)` rows are the deep-frontier tilings the Python bindings dispatch
to for domains above a 16-bit frontier (25-bit domain shown, per-thread tail
depth 4).
