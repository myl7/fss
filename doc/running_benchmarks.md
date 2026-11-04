# Run benchmarks and profiles

See [benchmark methodology and comparison curves](bench_figures.md) for the full-output adapters and domain sweeps.
The binaries below use [Google Benchmark](https://github.com/google/benchmark).
Each row times its named operation; batch and full-domain operations process multiple outputs per iteration.
The [benchmark coverage](benchmarks.md) page lists what they measure and the GPU kernel register usage.

Configure with `BUILD_BENCH=ON` and build the targets:

```bash
cmake -B build -DBUILD_BENCH=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target bench_cpu bench_gpu
```

Run all benchmarks:

```bash
./build/bench_cpu
./build/bench_gpu
```

Run a subset using `--benchmark_filter` (regex):

```bash
./build/bench_cpu --benchmark_filter=BM_DcfGen
./build/bench_cpu --benchmark_filter=BM_DpfEval_Uint_Aes/20
```

The `Makefile` runs these sources through `third_party/bench.py`.
Both CPU and GPU runs pin the host process to `CPU_ID`, verify the performance governor, and restore its previous value on exit.
A governor change may require noninteractive `sudo`.
`make bench_gpu` also selects `GPU_ID`.
Builds, raw measurements, environment metadata, and reports are saved under `build/third_party/`.

```bash
CPU_ID=0 make bench_cpu
GPU_ID=1 CUDA_ARCH=86 make bench_gpu
```

The [third-party comparison guide](bench_third_parties.md) covers fixed dependency versions, selecting libraries, and generating reports.
The `main` selection runs this library's benchmarks:

```bash
python3 third_party/bench.py run --libraries main --platform cpu --cpu 24 \
  --repetitions 5 --min-time 1 --run-id readme-cpu
python3 third_party/bench.py run --libraries main --platform gpu --cpu 8 --gpu 1 \
  --cuda-arch 120 --repetitions 5 --min-time 1 --run-id readme-gpu
```

Choose idle, allowed CPU and GPU indices on your host.
Direct binary invocations above leave affinity and the governor to the caller.

`CUDA_ARCH` is only needed when CMake cannot infer the architecture.
Two more targets support the sections below: `make ptx_info` rebuilds with `--ptxas-options=-v` and collects the register usage into `build/ptx_info.log`, and `make profile_gpu` records an Nsight Systems profile of one benchmark selected by `GPU_PROFILE_BENCH`.

## Comparison Curves

The domain sweep measures domain sizes `N = 2^8, 2^10, ..., 2^20`.
Each figure shows one platform, with five rows: DCF, full-width DPF, packed DPF, VDPF, and DMPF/VDMPF. Columns show `Gen`, `Eval`, and `EvalAll`.
Logarithmic time axes retain the full measured range. Legends distinguish the construction, output shape, PRG, and proof accumulator; colors, markers, and line styles identify each method.
Separate figures cover [prepared-key costs](figures/cpu-prepared.svg), [proof accumulators](figures/cpu-proof.svg), [PRP backends](figures/cpu-prp.svg), and [output groups](figures/cpu-output-groups.svg). The comparison site supports selecting individual methods to inspect overlapping curves.
Hardware, sampling, timing boundaries, and reproduction commands are in the [curve methodology](bench_figures.md).
The exact measured medians behind the curves, with per-domain values and pairwise library comparison, are on the [benchmark comparison site](https://myl7.github.io/fss/bench/).
An implementation appears in a panel only where it provides that method on that platform.
Gen timings count one call producing both party keys.

![CPU benchmark operations](figures/cpu-operations.svg)

CPU point evaluation uses one key and one thread, in µs/query; generation uses µs/key.
`EvalAll` produces all N outputs for one key, in ms/key.
The slow materialized point-loop cases use five domain sizes.
The verifiable VDPF curves (FSS VDPF and the Servan-Schreiber reference, eprint 2021/580, single point) include proof generation in the timed evaluation.
Verification stays outside timing.
Grotto DCF point `Eval` times the parity-tree query with Preprocess outside timing.
DMPF and VDMPF use `t = 64` points over `m = 112` Cuckoo buckets.
The VDMPF `EvalAll` benchmark materializes all inputs through batch evaluation, including proof generation.

![GPU benchmark operations](figures/gpu-operations.svg)

GPU `Gen` reports ns/key. Point `Eval` reports ns/query: ordinary point kernels use independent keys, while DMPF/VDMPF evaluate many queries against one key.
`EvalAll` uses one key, in ms/key.
The native GPU-DPF and packed EzPC adapters keep their native output configurations.
GPU `EvalAll` covers DPF, HalfTree DPF, DCF, the one-bit packed HalfTree DPF, VDPF and VDMPF with Merkle-fold proofs computed on device, Grotto DCF, and DMPF.
The threads-per-block sweep lives in the [curve methodology](bench_figures.md).

## Flamegraph

Generate a CPU flamegraph with `perf` and [FlameGraph](https://github.com/brendangregg/FlameGraph):

```bash
perf record -g ./build/bench_cpu --benchmark_filter=BM_DpfEval_Uint_Aes/20
perf script | /path/to/FlameGraph/stackcollapse-perf.pl | /path/to/FlameGraph/flamegraph.pl > build/flamegraph.svg
```

`make flamegraph` runs the same steps, taking the FlameGraph checkout from `FLAMEGRAPH_DIR` (default `../FlameGraph`) and the benchmark from `FLAMEGRAPH_BENCH`:

```bash
FLAMEGRAPH_DIR=/path/to/FlameGraph FLAMEGRAPH_BENCH=BM_DcfGen_Uint_Aes/20 make flamegraph
```

Open `build/flamegraph.svg` in a browser.
The graph is interactive: click a frame to zoom in.
