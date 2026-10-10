# Benchmark curve methodology

Use `third_party/figure_bench.py` to reproduce the time curves. Run from the
repository root on Linux after following the
[pinned dependency and toolchain setup](bench_third_parties.md#running).
The reference host is `cs659b`: AMD EPYC 9115, NVIDIA RTX PRO 5000, SM 120,
and CUDA 13.2. The experiment pins CPU operations to CPU 24, uses one CPU
thread, and verifies the `performance` governor. GPU runs use GPU 1 and pin
the launch process to CPU 16. The run manifest records the actual hardware,
toolchain versions, affinity arguments, and governor.

## Work and timing

The domain sweep uses `N = 2^n`, with
`n = 8, 10, 12, 14, 16, 18, 20`. CPU libfss and GPU-DPF materialized point-loop
EvalAll cases use `n = 8, 12, 14, 18, 20` because each iteration calls point
Eval at all N inputs. The other supported cases use seven domain sizes.
The separate current-FSS block sweep fixes `n = 20` and uses
`T = 16, 32, 64, 128, 256, 512, 1024` threads per block.

CPU Gen measures one two-party key generation, and CPU Eval measures one
party at one query. Full-domain EvalAll measures one party over N outputs
with `K = 1`. GPU Gen and point Eval use `K = 262144` keys per invocation,
and the figure divides invocation time by K. GPU full evaluation also uses
`K = 1`. Domain runs use T=256 for point operations and T=128 for full
evaluation where configurable. EzPC uses its fixed T=256 full adapter.
Read each record's `num_keys`, `domain_size`, and `threads_per_block` fields
before comparing an operation or interpreting a native items counter.

Current-FSS specialty schemes extend these boundaries. Grotto DCF point Eval
times the parity-tree query only, with Preprocess outside timing, and its
EvalAll times Preprocess plus the scan as one full-domain operation.
DMPF and VDMPF use `t = 64` points with `m = ChBucket(64, 80) = 112` Cuckoo
buckets and bucket domains of `2^(n-5)`; DMPF EvalAll evaluates every padded
bucket domain, and VDMPF EvalAll materializes through BatchEval over all N
inputs, including proof generation, as a point loop. Packed HalfTree DPF
EvalAll emits `N/128` packed blocks for one-bit outputs. CPU benchmarks pin
one CPU, which also pins the OpenMP team to one thread; DMPF EvalAll would
otherwise pay one parallel-region spawn per bucket.

Current FSS GPU FullEval sets `z = min(17, n)` for DPF,
`z = min(17, n - 1)` for HalfTreeDPF, and `b1 = z - log2(T)`.
The T sweep therefore adjusts the block root depth to satisfy the public
kernel constraint `2^(z - b1) = T`. The timed kernel includes all tree
traversal and excludes key generation, allocation, and upload.

Google Benchmark runs use five repetitions, at least 0.1 seconds of
measurement, and 0.1 seconds of warmup. The figure uses the median.
Rust cases use five separate Criterion processes, each with 20 samples,
0.2 seconds of warmup, and a 0.2-second requested measurement interval.
The figure uses the median of the five process medians. Criterion can extend
an interval for a slow operation. Setup and correctness checks outside the
measured iteration do not contribute to operation time.

CPU timings use framework wall time. Current FSS and historical FSS GPU
harnesses use CUDA events around operation kernel launches, with allocation
and input copies outside timing. EzPC Gen/Eval events surround the native GPU
API calls and include GPU work and transfers within that interval. EzPC full
evaluation times its packed output adapter kernel. GPU-DPF full evaluation
places events around the native `dpf_hybrid` call, with key upload and result
download outside timing. These boundaries appear in each record.

GPU-DPF full evaluation uses `FUSES_MATMUL=0`, one native block, and K=1. It
materializes uint128 outputs in native bit-reversed domain order; the checker
maps that order to input x outside timing. The EzPC full adapter writes packed
bits using the upstream DFS expansion, launches 256 threads, and has one
active DFS thread. Label these implementations and their timing boundaries
when comparing them with current FSS output materialization.

## Native configurations and validation

The primary current-FSS CPU curves use AES-NI and the byte group: 127 logical
output bits in 128-bit storage. Earlier tables used OpenSSL AES for current
FSS CPU operations. That PRG configuration differs, so the new figure values
can differ from those tables.

| Implementation | Native output configuration |
| --- | --- |
| Current FSS and historical FSS 0.7 | 127-bit XOR byte group, 128-bit storage; optional integer group modulo 2^127 |
| Current FSS Grotto DCF | One logical comparison bit; EvalAll returns N bool shares |
| Current FSS packed HalfTree DPF (experimental) | One logical bit; EvalAll returns N/128 packed 128-bit blocks |
| Current FSS DMPF / VDMPF | 127-bit XOR byte group per point; t=64 points over m=112 Cuckoo buckets |
| Rust FSS 0.6 | 128-bit XOR bytes; optional integer group modulo 2^128 |
| Google DPF / DCF | XOR128 / additive integers modulo 2^128 |
| GPU-DPF CPU and GPU | uint128 scalar output |
| libdpf | One logical bit; Eval returns a packed 128-bit leaf, EvalAll returns N/128 leaves |
| libfss DPF / DCF | GMP scalar modulo nextprime(2^32), a 33-bit field / uint64 modulo 2^64 |
| EzPC full adapter | Packed binary output |

The legends and records retain native PRGs: current-FSS CPU AES-NI, its GPU
ChaCha variants and software-AES point variant, GPU-DPF GPU ChaCha12, EzPC
and Google AES128, and historical FSS CPU AES-MMO-NI and GPU Salsa12.
libdpf uses `AES128-MMO/RustCrypto`, with runtime AES-NI dispatch from the
pinned `aes 0.8.4` crate. libfss uses `AES128-MMO/OpenSSL` through the low-level
`AES_encrypt` API. Output groups, packing, PRGs,
and APIs differ across implementations. The experiment reports their native
operation times for the configurations listed in each record.

Current FSS GPU EvalAll uses the corrected shared-frontier barriers and depth
bounds from commit `5ce7740`. The checks cover both party shares and their
reconstruction outside timing. Google DCF uses supported additive uint128;
the earlier XOR adapter configuration failed reconstruction because the
native DCF zero-value conversion does not handle XOR wrappers. Rust FSS 0.6
additive DCF fails equal-alpha reconstruction and stays outside the primary
byte-group curves. Preserve that failure if collecting optional integer
variants with `--include-uint`. EzPC's strict DCF API rejects n=8. Historical
FSS 0.7 has no native full-domain API. Keep these statuses in the dataset.

Main figures show logarithmic N and logarithmic time per key: CPU point
operations in microseconds, GPU point operations in nanoseconds, and full
evaluation in milliseconds. The block-size figure shows actual T and a
logarithmic time axis. Render only successful positive measurements; leave
unsupported, failed, and unmeasured points absent. Captions must follow the
collected raw data.

Every current-FSS curve carries its scheme in the legend (FSS DPF, FSS DCF,
FSS HalfTreeDPF, FSS PackedHalfTreeDPF with an experimental marker, and the
specialty schemes below), so readers constrained to one scheme can read its
own curve. Grotto DCF and the DMPF family have no like-for-like third-party
counterpart on the comparison axes, so they render on dedicated figures
rather than the main comparison figures: `cpu-grotto` for Grotto DCF with its
one-bit comparison output, and `cpu-dmpf` for DMPF and VDMPF together, since
adding verifiability changes functionality but little time. VDMPF BatchEval
timings include proof generation; verification itself is a comparison
outside timing.

## Run and export

Inspect the requested cases before starting. Each `run` directory must be
new; use a new suffix for a repeated experiment. The examples use the defaults
of five repetitions and the measurement settings above.

```sh
python3 third_party/figure_bench.py list --platform cpu --sweep domain
python3 third_party/figure_bench.py run --platform cpu --sweep domain \
  --cpu 24 --cpu-prg aes-ni --governor performance --jobs 2 \
  --directory build/third_party/sweep/results/cpu-domain
python3 third_party/figure_bench.py run --platform gpu --sweep domain \
  --cpu 16 --gpu 1 --cuda-arch 120 --governor performance --jobs 2 \
  --directory build/third_party/sweep/results/gpu-domain
python3 third_party/figure_bench.py run --platform gpu --libraries fss \
  --sweep block --operations Eval EvalAll --domain-bits 20 \
  --threads-per-block 16 32 64 128 256 512 1024 \
  --num-keys 262144 --cpu 16 --gpu 1 --cuda-arch 120 \
  --governor performance --jobs 2 \
  --directory build/third_party/sweep/results/gpu-block
python3 third_party/figure_bench.py export \
  --directory build/third_party/sweep/results/cpu-domain
python3 third_party/figure_bench.py merge \
  --inputs build/third_party/sweep/results/cpu-domain \
    build/third_party/sweep/results/gpu-domain \
    build/third_party/sweep/results/gpu-block \
  --directory build/third_party/sweep/results/merged
```

`run` exports its `chart-data.json` and CSV. Use `export` on an individual run
to rebuild them from saved `case.json` files; `merge` combines completed run
exports. For an explicit retry, pass its directory to `merge --retry-inputs`
and retain the superseded attempt provenance. A nonzero exit status can
accompany a completed dataset containing recorded failures; inspect the
records and logs before continuing.

Build caches live under `build/third_party/sweep/<library>/<platform>`.
Each result directory contains the run manifest, source, lockfile, and raw-result hashes,
per-case configuration and statuses, exact commands, logs, and native JSON or
Criterion estimates. Keep these files with the merged data.

Use the offline D3/Playwright setup in
[the figure authoring guide](figures/authoring/README.md), then export and inspect
`build/figures` before publishing. Install D3 under `build/figure-tools` and
configure the local Playwright runtime through `FSS_FIGURE_NODE_MODULES` when
the exporter's default runtime path is unavailable. Keep browser downloads
under `build/figure-tools/browsers` as described in that guide:

```sh
node doc/figures/authoring/export.mjs \
  build/third_party/sweep/results/merged/chart-data.json
node doc/figures/authoring/export.mjs \
  build/third_party/sweep/results/merged/chart-data.json --publish
```

The exporter writes authoring HTML, SVG, 2x PNG, and a source manifest under
`build/figures`. `--publish` copies the reviewed SVG and PNG assets to
`doc/figures`.
