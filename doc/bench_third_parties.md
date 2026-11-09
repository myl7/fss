# Benchmark runner and third-party comparisons

The native suite and third-party adapters use `in_bits=20` by default, a domain of 1,048,576 inputs.
See [benchmark methodology and comparison curves](bench_figures.md) for the
current adapters, additive uint128 DCF measurements, and domain sweeps.

`N` is the domain size, `T` is the CUDA threads per block, and `K` is the
number of keys processed per benchmark iteration or call. Keys per launch is
an additional count. In `kernel<<<grid_dim, block_dim>>>`, `block_dim` sets the
threads per block. The key count and domain size do not specify this launch
parameter.

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
| mastic       | [TrustworthyComputing/mastic](https://github.com/TrustworthyComputing/mastic)               | Rust            | CPU      | `bench.py --libraries mastic`                     |
| plasma       | [TrustworthyComputing/plasma](https://github.com/TrustworthyComputing/plasma)               | Rust            | CPU      | `bench.py --libraries plasma`                     |
| sycret       | [OpenMined/sycret](https://github.com/OpenMined/sycret)                                     | Rust            | CPU      | `bench.py --libraries sycret`                     |
| dpf-cpp      | [dkales/dpf-cpp](https://github.com/dkales/dpf-cpp)                                        | C++             | CPU      | `bench.py --libraries dpf-cpp`                    |
| dpfplusplus  | [rh3nry/dpfplusplus](https://github.com/rh3nry/dpfplusplus) (fetched, not vendored)          | C++             | CPU      | `bench.py --libraries dpfplusplus`                |
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
| mastic          | x       | x[^mastic] | x[^mastic] |         |          |             |             |
| plasma          | x       | x[^plasma] |             |         |          |             |             |
| sycret          |         |           |             |         |          |             |             |
| dpf-cpp         | x       | x         | x           |         |          |             |             |
| dpfplusplus     | x       | x         | x           |         |          |             |             |
| torchcsprng     |         |           |             |         |          |             | x           |

[^1]: See `doc/bench_aes128_soft.md` for implementation details and results.
[^mastic]: Mastic is a VIDPF: Eval chains the running proof, and EvalAll is
    `eval_tree` over the full domain for both parties including the proof digest
    comparison. Outputs are Field128 shares (prio's 128-bit prime field).
[^plasma]: PLASMA is a VIDPF: Eval chains the running proof over u64 shares
    mod 2^63-25. Upstream has no full-domain API, so no EvalAll is measured.
    sycret has no DPF rows in this table: it benchmarks equality (EQ) and
    comparison (CMP) keygen and batched point eval through its C ABI, with
    additive bit shares mod 2^32 over its fixed 32-bit domain.

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
| AES block pipelining | Yes (~8 blocks per AES-NI fill)                                                                |
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
| out_bits         | 128                                          |
| PRG              | AES-128 MMO (Matyas-Meyer-Oseas), 3 instances (left, right, value) |
| PRG acceleration | AES-NI (BoringSSL) + SIMD (Highway library), up to 64 AES blocks per fill         |
| Output group     | DPF: XOR<uint128>; DCF: additive uint128                                                       |
| Threading        | Single-thread (no rayon/OpenMP); SIMD parallelism via Highway      |
| Build            | Bazel, `-c opt`                                                    |
| Toolchain        | Bazel's configured C++ compiler                                    |
| Bench framework  | Google Benchmark 1.8.3 (Bazel module)                              |

The former Google DCF XOR configuration failed reconstruction and its timings are invalid.
The current DCF adapter uses additive uint128 outputs.

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
| Keys/call        | 512 (`BATCH_SIZE`)                                                                         |
| GPU strategy     | Hybrid (dpf_hybrid.cu, Z=128)                                                            |
| Entry size       | 16 x 32-bit values per DPF entry                                                         |
| Threading        | CUDA (128 threads/block, `DPF_HYBRID_THREADS_PER_BLOCK`)                                                                 |
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
| Keys/call        | 1024 (original); 2^18 (262144) for tuned run                                    |
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
| Keys/iteration   | 2^20 parallel gen/eval instances (1 thread each)    |
| Output group     | u128_le (wrapping add) and bytes (XOR)              |
| Threading        | CUDA (256 threads/block, 2^20 total threads)        |
| Build            | CMake, Release                                      |
| Toolchain        | nvcc + gcc                                          |
| Bench framework  | Google Benchmark 1.9.5 (UseManualTime, CUDA events) |

The GPU binary crashes at startup on SM 86 (misaligned address in the
upstream benchmark), so it only contributes curves on SM 120 hosts and its
GPU rows are dropped when the GPU reference is the A6000.

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
| PRG (DPF)       | ChaCha<2> (ChaCha20, 20 rounds)                                        |
| PRG (DCF)       | ChaCha<4> (ChaCha20, 20 rounds)                                        |
| PRG (AesSoft)   | `Aes128Soft<2>` (shared-mem Te0+sbox) — see `doc/bench_aes128_soft.md` |
| Output groups   | BytesGroup (XOR, 128-bit), UintGroup (Z\_{2^127})                      |
| Keys/iteration  | 2^20 parallel gen/eval instances (1 thread each)                       |
| Threading       | CUDA (256 threads/block, 2^20 total threads)                           |
| Build           | CMake, Release; requires OpenSSL                                       |
| Toolchain       | nvcc + g++ (C++20)                                                     |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events)                    |

### mastic (Rust)

| Setting              | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| in_bits              | 12, 16, 20                                                   |
| out_bits             | 128 (prio Field128, Montgomery form, counter appended)       |
| PRG                  | AES-128 MMO (fixed-key, `AES_0(ctr) xor ctr`)                |
| PRG acceleration     | AES-NI (`aes` crate)                                         |
| Proof                | BLAKE3 node hashes (16-byte); EvalAll compares both parties' digests |
| EvalAll method       | `eval_tree` over all leaves for both parties, proof included in timing |
| Output group         | Field128 additive shares                                     |
| Threading            | Single-thread (`RAYON_NUM_THREADS=1`)                        |
| Build                | Cargo via `mastic-bench/` wrapper: opt-level=3, LTO, codegen-units=1 |
| Toolchain            | Rust stable                                                  |
| Bench framework      | Criterion 0.5.1                                              |

### plasma (Rust)

| Setting              | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| in_bits              | 16, 20                                                       |
| out_bits             | 64 (u64 shares mod p = 2^63-25)                              |
| PRG                  | AES-128-CTR (software, RustCrypto)                           |
| Proof                | 12-byte BLAKE3 XOF running proof, chained through Eval       |
| EvalAll method       | None upstream; point eval only                               |
| Output group         | u64 additive shares mod p                                    |
| Threading            | Single-thread (`RAYON_NUM_THREADS=1`)                        |
| Build                | Cargo via `plasma-bench/` wrapper: opt-level=3, LTO, codegen-units=1 |
| Toolchain            | Rust stable                                                  |
| Bench framework      | Criterion 0.5.1                                              |

### sycret (Rust)

| Setting              | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| Domain               | Fixed 32-bit inputs                                          |
| Operations           | EQ and CMP keygen and batched point eval (C ABI entry points) |
| Key widths           | 621 B (EQ), 920 B (CMP)                                      |
| Batch sizes          | 1024, 16384, 65536 keys or points per call                   |
| PRG                  | Fixed-key AES-NI stream, 128 fixed streams                   |
| Output group         | Additive bit shares mod 2^32                                 |
| Threading            | Single-thread (`RAYON_NUM_THREADS=1`)                        |
| Maintenance          | Upstream unmaintained since 2021-10; no full-domain API      |
| Build                | Cargo via `sycret-bench/` wrapper: opt-level=3, LTO, codegen-units=1 |
| Toolchain            | Rust stable                                                  |
| Bench framework      | Criterion 0.5.1                                              |

### dpf-cpp (C++)

| Setting              | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| in_bits              | 16, 20, 24                                                   |
| out_bits             | 1 (packed-bit truth table)                                   |
| PRG                  | AES-128 MMO (AES-NI, two fixed keys for left/right children) |
| Gen randomness       | Fixed-seed test PRNG (upstream `PRNG::getTestPRNG`)          |
| EvalAll method       | `EvalFull` scalar and `EvalFull8` 8-way batched              |
| Output group         | GF(2) XOR                                                    |
| Threading            | Single-thread                                                |
| Build                | CMake, Release, `-maes -msse4.1 -mavx2 -mpclmul`, x86-64 only |
| Toolchain            | g++ (C++14)                                                  |
| Bench framework      | Google Benchmark 1.9.5                                       |

### dpfplusplus (C++)

| Setting              | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| in_bits              | 16, 20, 24                                                   |
| out_bits             | 1 (bool leaves packed 128 per `__m128i` node)                |
| PRG                  | Fixed-key AES (static zero key, counter/xor mode)            |
| Gen randomness       | `arc4random_buf`                                             |
| EvalAll method       | `evalfull` level-by-level traversal                          |
| Output group         | GF(2) XOR                                                    |
| Threading            | Single-thread                                                |
| Source               | Header-only; fetched and patched on demand, never committed (see [bench-dpfplusplus.md](bench-dpfplusplus.md)) |
| Build                | CMake, Release, `-O2 -maes -mavx2`, x86-64 only              |
| Toolchain            | g++ (C++17)                                                  |
| Bench framework      | Google Benchmark 1.9.5                                       |

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
dpfplusplus is the exception to vendoring: its headers are an anonymous
pre-release that forbids redistribution, so `prepare` clones it on demand under
`build/third_party/dpfplusplus/`, checks out the pinned commit, and applies
three documented text fixes before the wrapper compiles against it; no
upstream byte is committed. See [bench-dpfplusplus.md](bench-dpfplusplus.md)
for provenance, license status, and reproduction steps. The mastic, plasma,
and sycret entries build on the Rust stable toolchain through the same pinned
`RUST` map that libdpf uses.
Each CMake wrapper uses Google Benchmark 1.9.5, while Google DPF uses 1.8.3.
`prepare` initializes selected submodules recursively and installs the pinned
Rust toolchains and GPU-DPF environment. It does not run EzPC's setup script,
which also builds unrelated applications and downloads datasets. The EzPC
dependency wrapper uses upstream `sytorch/random.cpp`, cryptoTools, bitpack,
and LLAMA. It preserves their crypto and random-number implementations while
omitting unused SCI/SEAL and neural-network backends. GPU RNG and AES remain
the pinned EzPC CUDA implementations. Its compile commands and link map are
saved with the build.

### Run the native suite

Build the native binaries with `-DBUILD_BENCH=ON` and run them directly, or use the Makefile to select a CPU and GPU through the runner:

```bash
CPU_ID=0 make bench_cpu
GPU_ID=1 CUDA_ARCH=86 make bench_gpu
```

Select idle devices allowed by your host.
Direct binary invocations leave CPU affinity and the governor to the caller.
Both Makefile targets use the `decision` profile by default.
Set `BENCH_PROFILE=publication BENCH_WORKLOAD=legacy` to restore historical sampling and full-domain workloads.
Set `CUDA_ARCH` only when CMake cannot infer it.

### Native benchmark coverage

`bench_cpu` and `bench_gpu` cover DPF, DCF, VDPF, Half-Tree DPF, packed Half-Tree DPF, Grotto DCF, Half-Tree Grotto DCF, DMPF, and VDMPF.
The CPU suite compares OpenSSL AES, AES-NI, software AES, and ChaCha PRGs.
The GPU suite uses the device-callable software AES and ChaCha backends.
The suite also measures VDPF proof aggregation, Grotto preprocessing, and independent-key GPU batches.
Internal compact point-evaluation experiments are separate from the public `Eval` rows and are not public APIs.

Native names identify the scheme, operation, output group, PRG, and domain bits.
For example, `BM_DcfEval_Uint_AesRaw/17` evaluates DCF over a 2^17 domain with the `Uint` group and AES-NI PRG.
PRG suffixes are `Aes` for OpenSSL, `AesRaw` for AES-NI, `AesSoft` for software AES, and `ChaCha`.
Grotto rows omit the output group because they produce F2 shares.

A batch row times the whole API call, while its item counter may count keys, queries, or domain outputs.
Packing and other preparation may happen outside the timed call.
Check the timing boundary and denominator before comparing throughput or deriving latency per item.

### Run profiles

Both Python runners resolve a shared profile through `third_party/benchmark_policy.py`.
Explicit `--min-time` and `--repetitions` values (and `--warmup` on the figure runner) always override the profile timing, and explicit selection arguments always override the profile selection.

| Profile | Timing defaults | Selection defaults |
| --- | --- | --- |
| `decision` (default) | 0.3 s minimum measurement, one repetition, no extra warmup | current library only: `--libraries main` for `bench.py`; the FSS wrapper at `n = 8, 20` on the figure runner |
| `smoke` | 0.01 s, one repetition | current library, full-domain rows included |
| `publication` | `bench.py`: 1 s and five repetitions; figure runner: 0.1 s measurement, 0.1 s warmup, five repetitions | each runner's historical default selection |

`--smoke` remains an alias for `--profile smoke`; combining it with a different explicit profile fails.
The `decision` default excludes the internal compact-path and point-launcher experiment rows (names containing `EvalPoint`) from the `main` library; an explicit `--filter` and the `publication` profile keep the full selection.
The filters are plain POSIX extended alternations over the registered operation names, because the native binaries compile `--benchmark_filter` with a regex backend that rejects lookaround.
Smoke includes the full-domain rows instead of excluding them, so one correctness run covers at least one real `n = 20` full evaluation.
Explicit `--min-time` and `--repetitions` values reach the native commands unchanged in every profile, including smoke.
When the resolved repetitions exceed one, the current library's native commands (`main` and the FSS wrapper) run their repeated samples with Google Benchmark random interleaving, one `--benchmark_enable_random_interleaving=true` per command; a single repetition, third-party adapters, and the historical FSS binaries carry no such flag.
An explicit `--benchmark_enable_random_interleaving=false|true` after `--` overrides the mode for the run, is resolved before `--plan` prints, and the command carries exactly one such flag.
Interleaving shuffles the repetitions of the rows selected within one native command; it does not interleave independent processes, binaries, or cases, and is not a global A/B schedule or a precision guarantee.
The resolved mode is recorded per command in `--plan` and the run manifest.

`--plan` prints the resolved libraries, exact filters, timing budget, and work configuration without building or touching hardware.
A `--benchmark_filter=` value passed after `--` is folded into the plan before it is printed, so the plan always shows the exact filter the run would use.
The native result-row count is printed as unresolved because it is known only after the run.

#### Full-domain launch control

The main GPU batch full-domain rows process the `2^20`-key fixture through 1024-key API calls.
The fixture's alphas and evaluation points are masked to the configured input domain, and the receipt records that fixture domain.
`--workload daily` (default) runs one full API call per iteration, `--workload legacy` restores the historical 1024 calls per iteration, and `--full-launches N` selects 1 to 1024 calls explicitly.
The runner forwards the resolved value through `FSS_BENCH_FULL_LAUNCHES` and records it in the run manifest and `--plan`; the executed `calls_per_iteration` and `batch_keys` counters are checked against that plan, and batch keys stay 1024.
The four batch full-domain rows (DPF, DCF, HalfTree DPF, packed HalfTree DPF) must carry both counters; a receipt missing either fails the run.
The single-key full-domain rows (Grotto DCF, VDPF Merkle) are exempt.
`--workload legacy` conflicts with an explicit `--full-launches` other than 1024.
The daily row keeps real batch throughput semantics with a workload identity different from the legacy row, and its 1024-key batch cost divided by 1024 is an amortized per-key cost, not single-key latency.

New main and FSS receipts carry per-iteration work counters with `work_schema_version = 1`; a new `main` or `fss` collection whose receipts lack the marker fails instead of falling back to the legacy reading.
The runner normalizes them as time per iteration divided by work per iteration, keeps logical output counts, output bytes, proof counts, proof bytes, point-hash bytes, and GPU host wall time in separate summary columns, and records the process wall time per command separately.
The work unit names the primary metric: keys or PRG calls, queries, domain outputs, prepared leaves for preprocess rows, or proofs for proof rows.
Proof rows report their real zero keys and zero logical output bytes with the proof count and proof bytes in their own columns, and leave the key columns empty, because they process no keys.
`output_bytes` counts logical output storage bytes, not DRAM traffic; `proof_bytes` counts final proof storage; `point_hash_bytes` counts the raw corrected per-point hash material `Eval` stores, which is the input to `Prove`, not a proof.
The main GPU VDPF point rows carry `point_hash_bytes_per_iteration` with 64 hash bytes per key next to their 16 logical output bytes per key, and a new main collection missing it on those rows fails; the single-key full-domain Merkle row reports different fields.
Native `items_per_second` is cross-checked on CPU against `cpu_time` and on GPU against the manual event time, each raw repetition on its own, and never overwrites the wall-normalized rate.
Rows without the schema marker, including historical receipts and unchanged third-party adapters, keep the legacy name-based reading and are marked `legacy` in the summary.

Reproduce the historical workload and sampling together with `--profile publication --workload legacy`.

`make ptx_info` collects current compiler resource reports in `build/ptx_info.log`.
`make profile_gpu` records an Nsight Systems trace for `GPU_PROFILE_BENCH`.
See the README's Flamegraph section for CPU profiling.

### Run third-party comparisons

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
# The smoke profile defaults to the current library only, so name the suite explicitly.
python3 third_party/bench.py run --libraries all --platform all --cuda-arch 120 --cpu 24 --gpu 0 \
  --no-build --smoke --run-id smoke

# Full comparison runs. The publication profile restores the historical
# selection (every library) and five repetitions for Google Benchmark and Python;
# the default decision profile would run the current library only.
python3 third_party/bench.py run --profile publication --platform cpu --cpu 24 --no-build --run-id comparison-cpu
python3 third_party/bench.py run --profile publication --platform gpu --cpu 24 --gpu 0 --no-build --run-id comparison-gpu
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

- `run.json`: hardware, declared library versions, compiler and tool versions,
  governor, device selection, timing parameters, commands, and process status.
- Per-library logs and raw Google Benchmark JSON, Criterion estimates and
  samples, or Python timings and validation results.
- `summary.csv` and `summary.md`: normalized times, keys per iteration or call, throughput,
  measurement boundaries, and any failed processes.

The raw JSON and normalized CSV/JSON retain the field name `batch` for
compatibility. It records keys per timed iteration or call, or PRG calls for
raw AES rows. It does not record CUDA threads per block.

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

The [interactive comparison](https://myl7.github.io/fss/bench/) shows measured values, collection dates, and the configurations of each method.
See the [curve methodology](bench_figures.md) for the current figures, timing boundaries, and differences between collections.
Software AES has a separate [historical implementation comparison](bench_aes128_soft.md).
