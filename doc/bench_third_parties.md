# Third-party DPF/DCF Library Benchmarks

All benchmarks use in_bits=20 (domain size 2^20 = 1,048,576).

## Libraries

| Library      | Source                                                                                      | Language        | Platform | Script                                           |
| ------------ | ------------------------------------------------------------------------------------------- | --------------- | -------- | ------------------------------------------------ |
| libdpf       | [weikengchen/libdpf](https://github.com/weikengchen/libdpf)                                 | Rust            | CPU      | `bench_libdpf.sh` (runs `libdpf-bench/` wrapper) |
| libfss       | [frankw2/libfss](https://github.com/frankw2/libfss)                                         | C++             | CPU      | `bench_libfss.sh`                                |
| google_dpf   | [google/distributed_point_functions](https://github.com/google/distributed_point_functions) | C++             | CPU      | `bench_google_dpf.sh`                            |
| GPU-DPF      | [facebookresearch/GPU-DPF](https://github.com/facebookresearch/GPU-DPF)                     | C++/CUDA/Python | CPU+GPU  | `bench_gpu_dpf_cpu.sh`, `bench_gpu_dpf_gpu.sh`   |
| EzPC         | [mpc-msri/EzPC](https://github.com/mpc-msri/EzPC)                                           | CUDA C++        | GPU      | `bench_ezpc.sh`                                  |
| fss-rs 0.6.0 | [pado-labs/fss-rs](https://github.com/pado-labs/fss-rs)                                     | Rust            | CPU      | `bench_fss_v060.sh`                              |
| fss 0.7.0    | (internal)                                                                                  | C               | CPU+GPU  | `bench_fss_v070_cpu.sh`, `bench_fss_v070_gpu.sh` |
| fss          | (this repo)                                                                                 | C++/CUDA        | CPU+GPU  | `bench_fss_cpu.sh`, `bench_fss_gpu.sh`           |
| torchcsprng  | [meta-pytorch/csprng](https://github.com/meta-pytorch/csprng)                               | C++/CUDA        | CPU+GPU  | `bench.py --libraries torchcsprng`                |

## Benchmarked Operations

| Library         | DPF Gen | DPF Eval  | DPF EvalAll | DCF Gen | DCF Eval | DCF EvalAll | AesSoft[^1] |
| --------------- | ------- | --------- | ----------- | ------- | -------- | ----------- | ----------- |
| libdpf          | x       | x         | x           |         |          |             |             |
| libfss          | x       | x         | x           |         |          |             |             |
| google_dpf      | x       | x         | x           | x       | x        |             |             |
| GPU-DPF (CPU)   | x       | x         | x           |         |          |             |             |
| GPU-DPF (GPU)   | x (CPU) |           |             |         |          |             |             |
| EzPC            | x       | x         | x           | x       | x        |             |             |
| fss-rs 0.6.0    | x       | x         | x           | x       | x        | x           |             |
| fss 0.7.0 (CPU) | x       | x         |             | x       | x        |             |             |
| fss 0.7.0 (GPU) | x       | x         |             | x       | x        |             |             |
| fss (CPU)       | x       | x         | x           | x       | x        |             | x           |
| fss (GPU)       | x       | x         |             | x       | x        |             | x           |
| torchcsprng     |         |           |             |         |          |             | x           |

[^1]: See `doc/bench_aes128_soft.md` for implementation details and results.

GPU-DPF's GPU operation is a full-domain table reduction. EzPC's `EvalAll`
traverses the domain and returns a packed reduction bit per key. These operations
do not materialize all domain outputs. The report keeps their names and counts
their throughput in keys. The raw AES comparison is built by the `torchcsprng`
wrapper, which includes both the vendored textbook implementation and fss's
T-table implementation. The fss wrapper's software-AES DPF evaluation remains a
separate operation.

## Settings

### libdpf (Rust)

| Setting              | Value                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| in_bits              | 20                                                                                             |
| out_bits             | 1 (packed in 128-bit blocks)                                                                   |
| Tree depth           | 13 (n-7; packs 128 points per block)                                                           |
| PRG                  | AES-128 MMO                                                                                    |
| PRG acceleration     | AES-NI (`aes` crate 0.8, auto-detected)                                                        |
| AES batch pipelining | Yes (~8 blocks per AES-NI fill)                                                                |
| Output group         | 1-bit XOR (packed 128-bit blocks)                                                              |
| Threading            | Single-thread (`RAYON_NUM_THREADS=1`); rayon multi-thread available (threshold >= 512 parents) |
| Build                | Cargo, release profile: opt-level=3, LTO, codegen-units=1                                      |
| Toolchain            | Rust nightly-2025-09-01                                                                         |
| Bench framework      | Criterion 0.5 (default: 5s measurement, 100 samples, 3s warm-up)                               |

### libfss (C++)

| Setting          | Value                                                 |
| ---------------- | ----------------------------------------------------- |
| in_bits          | 20                                                    |
| out_bits         | ~32 (prime field mod 2^32+15, GMP mpz_class)          |
| PRG              | AES-128                                               |
| PRG implementation | OpenSSL `AES_encrypt` compatibility path             |
| Output group     | Z_p (p = next prime > 2^32 = 4294967311)              |
| Threading        | Single-thread (`OMP_NUM_THREADS=1`); OpenMP available |
| Build            | CMake, Release; `-maes -msse4.2`                      |
| Toolchain        | g++ (C++11)                                           |
| Bench framework  | Google Benchmark 1.9.5                                |

### google_dpf (C++)

| Setting          | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| in_bits          | 20                                                                 |
| out_bits         | 128 (XorWrapper<uint128>)                                          |
| PRG              | AES-128 MMO (Matyas-Meyer-Oseas), 3 instances (left, right, value) |
| PRG acceleration | AES-NI (BoringSSL) + SIMD (Highway library), batch size 64         |
| Output group     | XOR<uint128>                                                       |
| Threading        | Single-thread (no rayon/OpenMP); SIMD parallelism via Highway      |
| Build            | Bazel, `-c opt`                                                    |
| Toolchain        | Bazel's configured C++ compiler                                    |
| Bench framework  | Google Benchmark 1.8.3 (Bazel module)                              |

### GPU-DPF CPU (C++)

| Setting          | Value                                         |
| ---------------- | --------------------------------------------- |
| in_bits          | 20                                            |
| out_bits         | 128 (uint128_t)                               |
| PRG              | AES-128                                       |
| PRG acceleration | Software AES (table-based, from `aes_core.h`) |
| Output group     | Modular integer (uint128_t)                   |
| EvalAll method   | Sequential loop of N point evals              |
| Threading        | Single-thread                                 |
| Build            | CMake, Release, `-O3`                         |
| Toolchain        | g++ (C++17)                                   |
| Bench framework  | Google Benchmark 1.9.5                        |

### GPU-DPF GPU (Python/CUDA)

| Setting          | Value                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------- |
| in_bits          | 20                                                                                       |
| out_bits         | 128 (uint128_t)                                                                          |
| PRG              | ChaCha20                                                                                 |
| PRG acceleration | GPU ChaCha20 (12 rounds)                                                                 |
| Output group     | Modular integer (uint128_t)                                                              |
| Batch size       | 512 (BATCH_SIZE)                                                                         |
| GPU strategy     | Hybrid (dpf_hybrid.cu, Z=128)                                                            |
| Entry size       | 16 x 32-bit values per DPF entry                                                         |
| Threading        | CUDA (128 threads/block)                                                                 |
| Build            | PyTorch CUDA extension (`uv pip install torch && CC=g++ uv run python setup.py install`) |
| Toolchain        | nvcc + PyTorch (managed via uv)                                                          |
| Bench framework  | Python `perf_counter_ns`, median and raw samples, synchronized end-to-end GPU reduction |

### EzPC (CUDA C++)

| Setting          | Value                                                                           |
| ---------------- | ------------------------------------------------------------------------------- |
| in_bits          | 20                                                                              |
| out_bits         | 1 (DPF), 1 (DCF)                                                                |
| PRG              | AES-128                                                                         |
| PRG acceleration | Software AES (GPU shared memory S-box lookup, `gpu_aes_shm.cu`)                 |
| Output group     | u64 (modular integer)                                                           |
| Batch size       | 1024 (original); 2^18 (262144) for tuned run                                    |
| GPU memory pool  | 512 MiB prefill by default, capped at half the free memory. Configure `FSS_EZPC_POOL_MIB`. |
| Threading        | CUDA (256 threads/block)                                                        |
| Build            | CMake, Release; an out-of-source wrapper builds the pinned benchmark dependencies |
| Toolchain        | nvcc + g++                                                                      |
| Bench framework  | Google Benchmark 1.9.5 (UseManualTime, CUDA events)                             |

### fss-rs 0.6.0 (Rust)

| Setting          | Value                                                                            |
| ---------------- | -------------------------------------------------------------------------------- |
| in_bits          | 20 (FILTER_BITN=20, IN_BLEN=3 bytes)                                             |
| out_bits         | 128 (OUT_BLEN=16 bytes)                                                          |
| PRG              | AES-128 MMO (Matyas-Meyer-Oseas); DPF uses 2 AES instances, DCF uses 4           |
| PRG acceleration | AES-NI (`aes` crate, auto-detected x86_64/ARM)                                   |
| Output group     | ByteGroup (XOR) and U128Group (wrapping add)                                     |
| Threading        | Single-thread (`RAYON_NUM_THREADS=1`); rayon multi-thread compiled in by default |
| Build            | Cargo, release profile                                                           |
| Toolchain        | Rust nightly-2025-09-01 (the default portable SIMD backend requires nightly)      |
| Bench framework  | Criterion 0.5.1 (default: 5s measurement, 100 samples)                           |

### fss 0.7.0 CPU (C)

| Setting          | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| in_bits          | 20                                                                 |
| out_bits         | 128 (kLambda=16 bytes; 127 effective, MSB truncated)               |
| PRG              | AES-128 MMO (Matyas-Meyer-Oseas), AES-NI hardware (`aes_mmo_ni.c`) |
| PRG acceleration | AES-NI (`-msse2 -maes`, `_mm_aesenc_si128`)                        |
| BLOCK_NUM        | DPF: 2 (32-byte output and key state); DCF: 4 (64 bytes)            |
| Output group     | u128_le (wrapping add) and bytes (XOR)                             |
| Threading        | Single-thread                                                      |
| Build            | CMake, Release                                                     |
| Toolchain        | gcc (C11) + g++ (C++17 for bench harness)                          |
| Bench framework  | Google Benchmark 1.9.5                                             |

### fss 0.7.0 GPU (C/CUDA)

| Setting          | Value                                               |
| ---------------- | --------------------------------------------------- |
| in_bits          | 20                                                  |
| out_bits         | 128 (kLambda=16 bytes; 127 effective)               |
| PRG              | Salsa20 (12 rounds)                                 |
| PRG acceleration | GPU Salsa20 kernel                                  |
| BLOCK_NUM        | DPF: 1 (32-byte output, 8-byte nonce); DCF: 2 (64-byte output, 16-byte nonce) |
| GPU instances    | 2^20 parallel gen/eval instances (1 thread each)    |
| Output group     | u128_le (wrapping add) and bytes (XOR)              |
| Threading        | CUDA (256 threads/block, 2^20 total threads)        |
| Build            | CMake, Release                                      |
| Toolchain        | nvcc + gcc                                          |
| Bench framework  | Google Benchmark 1.9.5 (UseManualTime, CUDA events) |

### fss CPU (C++/CUDA)

| Setting         | Value                                                                    |
| --------------- | ------------------------------------------------------------------------ |
| in_bits         | 20                                                                       |
| out_bits        | 128 (BytesGroup XOR, UintGroup Z\_{2^127})                               |
| PRG (DPF)       | AES-128 MMO (Matyas-Meyer-Oseas), mul=2, AES-NI (OpenSSL EVP_CIPHER_CTX) |
| PRG (DCF)       | AES-128 MMO, mul=4, AES-NI (OpenSSL EVP_CIPHER_CTX)                      |
| PRG (AesSoft)   | `Aes128Soft<2>` — see `doc/bench_aes128_soft.md`                         |
| Output groups   | BytesGroup (XOR, 128-bit), UintGroup (Z\_{2^127})                        |
| Threading       | Single-thread                                                            |
| Build           | CMake, Release; requires OpenSSL                                         |
| Toolchain       | nvcc + g++ (C++20)                                                       |
| Bench framework | Google Benchmark 1.9.5                                                   |

### fss GPU (C++/CUDA)

| Setting         | Value                                                                  |
| --------------- | ---------------------------------------------------------------------- |
| in_bits         | 20                                                                     |
| out_bits        | 128 (BytesGroup XOR, UintGroup Z\_{2^127})                             |
| PRG (DPF)       | ChaCha<2> (ChaCha20, 12 rounds)                                        |
| PRG (DCF)       | ChaCha<4> (ChaCha20, 12 rounds)                                        |
| PRG (AesSoft)   | `Aes128Soft<2>` (shared-mem Te0+sbox) — see `doc/bench_aes128_soft.md` |
| Output groups   | BytesGroup (XOR, 128-bit), UintGroup (Z\_{2^127})                      |
| GPU instances   | 2^20 parallel gen/eval instances (1 thread each)                       |
| Threading       | CUDA (256 threads/block, 2^20 total threads)                           |
| Build           | CMake, Release; requires OpenSSL                                       |
| Toolchain       | nvcc + g++ (C++20)                                                     |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events)                    |

### torchcsprng (C++/CUDA)

See `doc/bench_aes128_soft.md` for implementation details and results.

| Setting         | Value                                                       |
| --------------- | ----------------------------------------------------------- |
| Build           | CMake, Release (standalone in `third_party/torchcsprng/`)   |
| Toolchain       | nvcc + g++ (C++14)                                          |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events for GPU) |

## Running

The reproducible runner is `third_party/bench.py`. It supports `list`, `prepare`,
`build`, `run`, and `summarize`. All build trees, environments, caches, logs, and
results are under the repository's `build/third_party/` directory.

The supported environment is Linux x86_64 with an NVIDIA GPU for GPU runs.
Select an idle CPU and GPU on a shared host. The fss and raw AES CPU wrappers also
require a CUDA toolkit because their sources include CUDA headers and device
code. Rust, libfss, Google DPF, GPU-DPF CPU, and fss 0.7.0 CPU can be selected
independently. CPU AES-NI cases require an AES-capable processor.

Install a C++ compiler, CMake, OpenSSL, GMP, MPFR, Eigen3, `taskset`, Rustup,
Bazelisk, and uv. The full suite uses the CUDA toolkit's runtime, cuRAND, and
headers. For Ubuntu 24.04, the development packages are:

```bash
sudo apt-get install build-essential cmake libssl-dev libgmp-dev libmpfr-dev libeigen3-dev util-linux
```

GPU-DPF pins Python 3.11.16, PyTorch 2.10.0+cu130, NumPy 1.26.4, setuptools
80.9.0, Ninja 1.13.0, and uv 0.12.24 in its project and lockfile. It requires a
CUDA 13 toolkit and a driver compatible with that PyTorch wheel. The runner
bootstraps the pinned uv version into `build` when required. It builds the
extension from a copy of the pinned source and applies any checked-in compatibility
patches to that copy. The submodule stays at its recorded commit.

The other source versions are fixed by Git submodule entries, Cargo lockfiles,
Rust toolchain files, and Bazel's module lockfile. Google DPF uses Bazel 7.6.1.
Each CMake wrapper uses Google Benchmark 1.9.5, while Google DPF uses 1.8.3.
`prepare` initializes selected submodules recursively and installs the pinned
Rust toolchains and GPU-DPF environment. It does not run EzPC's setup script,
which also builds unrelated applications and downloads datasets. The EzPC
dependency wrapper uses upstream `sytorch/random.cpp`, cryptoTools, bitpack,
and LLAMA. It preserves their crypto and random-number implementations while
omitting unused SCI/SEAL and neural-network backends. GPU RNG and AES remain
the pinned EzPC CUDA implementations. Its compile commands and link map are
saved with the build.

From a fresh checkout, run:

```bash
git clone https://github.com/myl7/fss.git
cd fss
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

python3 third_party/bench.py list
python3 third_party/bench.py prepare --platform all
python3 third_party/bench.py build --platform all --cuda-arch 120 --jobs 4

# Short runs validate initialization and output collection. They omit long EvalAll cases.
python3 third_party/bench.py run --platform all --cuda-arch 120 --cpu 24 --gpu 0 \
  --no-build --smoke --run-id smoke

# Full comparison runs, using five repetitions for Google Benchmark and Python.
python3 third_party/bench.py run --platform cpu --cpu 24 --no-build --run-id comparison-cpu
python3 third_party/bench.py run --platform gpu --cpu 24 --gpu 0 --no-build --run-id comparison-gpu
```

Replace `120`, `24`, and `0` with the GPU architecture, an idle allowed CPU, and
the GPU index on your host. `--jobs` defaults to four to limit build load.
`--libraries` accepts comma-separated names from `list`, so a single library can
be prepared, built, and run without the rest of the suite:

```bash
python3 third_party/bench.py prepare --libraries libfss --platform cpu
python3 third_party/bench.py run --libraries libfss --platform cpu --cpu 24 \
  --filter 'DPF/(Gen|Eval)$' --run-id libfss-point
```

Every run pins the host process with `taskset` and sets `OMP_NUM_THREADS=1` and
`RAYON_NUM_THREADS=1`. The default governor policy is `performance`. The runner
reads the previous governor, switches it if required, verifies it before
measurement, and restores it on success, failure, or handled termination. If
writing the governor requires privilege, configure noninteractive `sudo` for
that write. A missing governor interface or failed switch stops the run.
`--governor keep` is an explicit alternative for smoke checks or hosts with
externally managed clocks. Its actual governor is recorded with the results.

The compatibility `bench_*.sh` scripts use the same runner and accept native
benchmark arguments. `CPU_ID`, `GPU_ID`, `CUDA_ARCH`, and `JOBS` set defaults:

```bash
CPU_ID=24 bash third_party/bench_libfss.sh --benchmark_filter='DPF/Gen'
```

`--min-time` controls Google Benchmark's measurement window and Criterion's
measurement time. Criterion reports its median estimate from 100 samples,
or ten in smoke mode. `--repetitions` applies to Google Benchmark and Python.
Do not compare a smoke result with a full measurement.

Each run creates a new `build/third_party/results/<run-id>/` containing:

- `run.json`: hardware, source and lockfile hashes, compiler and tool versions,
  governor, device selection, timing parameters, commands, and process status.
- Per-library logs and raw Google Benchmark JSON, Criterion estimates and
  samples, or Python timings and validation results.
- `summary.csv` and `summary.md`: normalized times, batch size, throughput,
  measurement boundaries, and any failed processes.

Rebuild a report from copied results with:

```bash
python3 third_party/bench.py summarize build/third_party/results/comparison-gpu
```

The runner preserves partial results and returns nonzero if any process fails.
Historical fss 0.7.0 uses separate binaries and PRG providers for each scheme
and group on both CPU and GPU. DPF requires a 32-byte PRG output, while DCF
requires 64 bytes. Sharing the DCF provider with DPF would overrun its scratch
buffer. The wrappers check both-party reconstruction and guarded scratch
buffers before timing. GPU operations run in separate processes so the pinned
uint DCF alignment failure does not prevent other measurements. That failure
remains visible in the report. EzPC registers 1024 and 262144-key Gen/Eval cases, but
only the 1024-key full-domain reduction to keep the traversal bounded.

The `main` selection runs `src/bench_cpu.cu` and `src/bench_gpu.cu` for the README.
It is excluded from `--libraries all` because its groups and operations differ
from the comparison harness. Use `--libraries main` to refresh that separate
set of results.

## Results

### Hardware

Measured on cs659b on 2026-10-10 (Asia/Hong_Kong).

| Component | Value |
| --- | --- |
| CPU | 2 × AMD EPYC 9115, 16 cores/socket, 64 hardware threads total |
| CPU placement | CPU cases: `taskset -c 24`, performance governor, one thread |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell, sm_120, GPU 0, driver 595.71.05 |
| GPU host placement | `taskset -c 16`, performance governor |
| CUDA | 13.2, nvcc V13.2.78 |
| Build | Release, CUDA architecture 120 |

The tables use medians: five repetitions for Google Benchmark and Python, and
Criterion's median estimate from 100 samples with a 3-second warm-up and at
least a 1-second measurement window. Each input domain has 2^20 points.

### CPU

Times are per key. `EvalAll` and Rust `FullEval` materialize the full domain.
Groups, output widths, and PRGs differ as listed in Settings: libdpf packs
1-bit XOR outputs, libfss uses a prime field, and the other CPU rows use their
128-bit XOR or modular groups. GPU-DPF's native CPU row uses software AES.

| Library | DPF Gen | DPF Eval | DPF EvalAll | DCF Gen | DCF Eval | DCF EvalAll |
| --- | --- | --- | --- | --- | --- | --- |
| libdpf | 1.12 µs | 437 ns | 92.2 µs | — | — | — |
| libfss | 59.1 µs | 5.54 µs | 5.8 s | — | — | — |
| google_dpf | 3.15 µs | 857 ns | 37.5 ms | 8.28 µs | 4 µs | — |
| GPU-DPF (software AES) | 454 µs | 19.7 µs | 20.6 s | — | — | — |
| fss-rs 0.6.0 (bytes) | 573 ns | 431 ns | 45.9 ms | 684 ns | 507 ns | 57 ms |
| fss-rs 0.6.0 (uint) | 580 ns | 430 ns | 45.2 ms | 698 ns | 499 ns | 51.3 ms |
| fss 0.7.0 (bytes) | 1 µs | 1.09 µs | — | 1.43 µs | 926 ns | — |
| fss 0.7.0 (uint) | 990 ns | 1.1 µs | — | 1.91 µs | 996 ns | — |
| fss (bytes) | 1.68 µs | 501 ns | 38.2 ms | 1.71 µs | 928 ns | — |
| fss (uint) | 978 ns | 501 ns | 39.8 ms | 2.02 µs | 943 ns | — |

### GPU

Times below are the median batch duration divided by the batch size, except
GPU-DPF generation, which is a separately measured CPU call with batch 1.
CUDA events time the native GPU kernels. GPU-DPF uses synchronized Python
wall time including key conversion, host/device transfers, table reduction,
and CPU result construction.

| Library | Batch | DPF Gen/key | DPF Eval/key | Materialized DPF EvalAll/key | DPF full-domain reduction/key | DCF Gen/key | DCF Eval/key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPU-DPF | 512 | 86.2 µs (CPU, batch 1) | — | — | 115 µs | — | — |
| EzPC | 1024 | 142 ns | 64.9 ns | — | 31.3 µs | 219 ns | 72.2 ns |
| EzPC (large batch) | 262144 | 11.4 ns | 7.94 ns | — | — | 14.2 ns | 8.22 ns |
| fss 0.7.0 (bytes) | 1048576 | 54.6 ns | 19.1 ns | — | — | 140 ns | 45.5 ns |
| fss 0.7.0 (uint) | 1048576 | 54.4 ns | 19 ns | — | — | NA[^legacy-dcf] | NA[^legacy-dcf] |
| fss (bytes) | 1048576 | 1.86 ns | 1.34 ns | — | — | 1.96 ns | 1.35 ns |
| fss (uint) | 1048576 | 1.89 ns | 1.34 ns | — | — | 2.23 ns | 1.33 ns |

GPU-DPF's ChaCha12 Python generation takes 86.2 µs per key on
host CPU 16. Its reduction reads a nonzero 2^20 × 1 int32 table, padded to 16
columns by upstream. EzPC's `EvalAll` returns a packed reduction bit per key.
Their reduction throughput counts keys, while materialized `EvalAll`
throughput can count domain outputs. These columns describe different work.
GPU-DPF has 128-bit internal arithmetic and int32 table results. EzPC has
1-bit outputs. fss 0.7.0 and fss retain their respective groups and PRGs from
Settings.

The fss software-AES DPF evaluation is reported separately in
[the software AES comparison](bench_aes128_soft.md). Raw PRG calls are also
reported there.

The source reports are under `build/third_party/results/`: `comparison-cpu-performance/summary.csv`, `comparison-cpu-rust-google-performance/summary.csv`, `comparison-gpu-performance/summary.csv`, `comparison-fss070-cpu-correct-provider/summary.csv`, `comparison-fss070-gpu-correct-provider/summary.csv`.
The fss 0.7.0 rows use the corrected provider adapter. Earlier CPU and GPU
DPF measurements are superseded because the adapter supplied PRG buffer sizes
that could overrun their allocations. The corrected adapter uses each
provider's required block count and initialization length. The pinned upstream
algorithms are unchanged.

[^legacy-dcf]: fss 0.7.0 uint GPU DCF Gen and Eval both fail with a CUDA
    misaligned-address error. Its packed 33-byte correction-word stride contains
    a `__uint128_t` access at byte offset 16 that can be unaligned. These cases
    have no timings or fallback results. The run records exactly two failed
    processes. The valid DPF and bytes DCF cases use separate processes.
