# Third-party DPF/DCF Library Benchmarks

All benchmarks use in_bits=20 (domain size 2^20 = 1,048,576).

## Libraries

| Library | Source | Language | Platform | Script |
|---------|--------|----------|----------|--------|
| libdpf | [weikengchen/libdpf](https://github.com/weikengchen/libdpf) | Rust | CPU | `bench_libdpf.sh` (runs `libdpf-bench/` wrapper) |
| libfss | [nicholasgasior/libfss](https://github.com/nicholasgasior/libfss) | C++ | CPU | `bench_libfss.sh` |
| google_dpf | [google/distributed_point_functions](https://github.com/google/distributed_point_functions) | C++ | CPU | `bench_google_dpf.sh` |
| GPU-DPF | [facebookresearch/GPU-DPF](https://github.com/facebookresearch/GPU-DPF) | C++/CUDA/Python | CPU+GPU | `bench_gpu_dpf_cpu.sh`, `bench_gpu_dpf_gpu.sh` |
| EzPC | [mpc-msri/EzPC](https://github.com/mpc-msri/EzPC) | CUDA C++ | GPU | `bench_ezpc.sh` |
| fss-rs 0.6.0 | [pado-labs/fss-rs](https://github.com/pado-labs/fss-rs) | Rust | CPU | `bench_fss_v060.sh` |
| fss 0.7.0 | (internal) | C | CPU+GPU | `bench_fss_v070_cpu.sh`, `bench_fss_v070_gpu.sh` |
| fss | (this repo) | C++/CUDA | CPU+GPU | `bench_fss_cpu.sh`, `bench_fss_gpu.sh` |
| torchcsprng | [pytorch/csprng](https://github.com/pytorch/csprng) | C++/CUDA | CPU+GPU | (see Running) |

## Benchmarked Operations

| Library | DPF Gen | DPF Eval | DPF EvalAll | DCF Gen | DCF Eval | DCF EvalAll | AesSoft[^1] |
|---------|---------|----------|-------------|---------|----------|-------------|-------------|
| libdpf | x | x | x | | | | |
| libfss | x | x | x | | | | |
| google_dpf | x | x | x | x | x | | |
| GPU-DPF (CPU) | x | x | x | | | | |
| GPU-DPF (GPU) | x | x (batch) | | | | | |
| EzPC | x | x | x | x | x | | |
| fss-rs 0.6.0 | x | x | x | x | x | x | |
| fss 0.7.0 (CPU) | x | x | | x | x | | |
| fss 0.7.0 (GPU) | x | x | | x | x | | |
| fss (CPU) | x | x | x | x | x | | x |
| fss (GPU) | x | x | | x | x | | x |
| torchcsprng | | | | | | | x |

[^1]: See `doc/bench_aes128_soft.md` for implementation details and results.

## Settings

### libdpf (Rust)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 1 (packed in 128-bit blocks) |
| Tree depth | 13 (n-7; packs 128 points per block) |
| PRG | AES-128 MMO |
| PRG acceleration | AES-NI (`aes` crate 0.8, auto-detected) |
| AES batch pipelining | Yes (~8 blocks per AES-NI fill) |
| Output group | 1-bit XOR (packed 128-bit blocks) |
| Threading | Single-thread (`RAYON_NUM_THREADS=1`); rayon multi-thread available (threshold >= 512 parents) |
| Build | Cargo, release profile: opt-level=3, LTO, codegen-units=1 |
| Toolchain | Rust nightly |
| Bench framework | Criterion 0.5 (default: 5s measurement, 100 samples, 3s warm-up) |

### libfss (C++)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | ~32 (prime field mod 2^32+15, GMP mpz_class) |
| PRG | AES-128 |
| PRG acceleration | AES-NI (OpenSSL `AES_encrypt`) |
| Output group | Z_p (p = next prime > 2^32 = 4294967311) |
| Threading | Single-thread (`OMP_NUM_THREADS=1`); OpenMP available |
| Build | CMake, Release; `-maes -msse4.2` |
| Toolchain | g++ (C++11) |
| Bench framework | Google Benchmark 1.9.5 |

### google_dpf (C++)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (XorWrapper<uint128>) |
| PRG | AES-128 MMO (Matyas-Meyer-Oseas), 3 instances (left, right, value) |
| PRG acceleration | AES-NI (BoringSSL) + SIMD (Highway library), batch size 64 |
| Output group | XOR<uint128> |
| Threading | Single-thread (no rayon/OpenMP); SIMD parallelism via Highway |
| Build | Bazel, `-c opt` |
| Toolchain | Clang (Bazel default) |
| Bench framework | Google Benchmark 1.8.3 (Bazel module) |

### GPU-DPF CPU (C++)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (uint128_t) |
| PRG | AES-128 |
| PRG acceleration | Software AES (table-based, from `aes_core.h`) |
| Output group | Modular integer (uint128_t) |
| EvalAll method | Sequential loop of N point evals |
| Threading | Single-thread |
| Build | CMake, Release, `-O3` |
| Toolchain | g++ (C++17) |
| Bench framework | Google Benchmark 1.9.5 |

### GPU-DPF GPU (Python/CUDA)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (uint128_t) |
| PRG | ChaCha20 |
| PRG acceleration | GPU ChaCha20 (12 rounds) |
| Output group | Modular integer (uint128_t) |
| Batch size | 512 (BATCH_SIZE) |
| GPU strategy | Hybrid (dpf_hybrid.cu, Z=128) |
| Entry size | 16 x 32-bit values per DPF entry |
| Threading | CUDA (128 threads/block) |
| Build | PyTorch CUDA extension (`uv pip install torch && CC=g++ uv run python setup.py install`) |
| Toolchain | nvcc + PyTorch (managed via uv) |
| Bench framework | Python `time.perf_counter` (gen) + CUDA events (eval), 10 reps, median |

### EzPC (CUDA C++)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 1 (DPF), 1 (DCF) |
| PRG | AES-128 |
| PRG acceleration | Software AES (GPU shared memory S-box lookup, `gpu_aes_shm.cu`) |
| Output group | u64 (modular integer) |
| Batch size | 1024 |
| GPU memory pool | 20 GB pre-allocated (cudaMallocAsync) |
| Threading | CUDA (256 threads/block) |
| Build | CMake, Release; requires pre-built sytorch (`cd EzPC/GPU-MPC && bash setup.sh`) |
| Toolchain | nvcc + g++ |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events) |

### fss-rs 0.6.0 (Rust)

| Setting | Value |
|---------|-------|
| in_bits | 20 (FILTER_BITN=20, IN_BLEN=3 bytes) |
| out_bits | 128 (OUT_BLEN=16 bytes) |
| PRG | AES-128 MMO (Matyas-Meyer-Oseas); DPF uses 2 AES instances, DCF uses 4 |
| PRG acceleration | AES-NI (`aes` crate, auto-detected x86_64/ARM) |
| Output group | ByteGroup (XOR) and U128Group (wrapping add) |
| Threading | Single-thread (`RAYON_NUM_THREADS=1`); rayon multi-thread compiled in by default |
| Build | Cargo, release profile |
| Toolchain | Rust stable or nightly |
| Bench framework | Criterion 0.5.1 (default: 5s measurement, 100 samples) |

### fss 0.7.0 CPU (C)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (kLambda=16 bytes; 127 effective, MSB truncated) |
| PRG | AES-128 MMO (Matyas-Meyer-Oseas), AES-NI hardware (`aes_mmo_ni.c`) |
| PRG acceleration | AES-NI (`-msse2 -maes`, `_mm_aesenc_si128`) |
| BLOCK_NUM | 4 (for DCF compatibility) |
| Output group | u128_le (wrapping add) and bytes (XOR) |
| Threading | Single-thread |
| Build | CMake, Release |
| Toolchain | gcc (C11) + g++ (C++17 for bench harness) |
| Bench framework | Google Benchmark 1.9.5 |

### fss 0.7.0 GPU (C/CUDA)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (kLambda=16 bytes; 127 effective) |
| PRG | Salsa20 (12 rounds) |
| PRG acceleration | GPU Salsa20 kernel |
| BLOCK_NUM | 2 (Salsa20 compatibility) |
| GPU instances | 2^20 parallel gen/eval instances (1 thread each) |
| Output group | u128_le (wrapping add) and bytes (XOR) |
| Threading | CUDA (256 threads/block, 2^20 total threads) |
| Build | CMake, Release |
| Toolchain | nvcc + gcc |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events) |

### fss CPU (C++/CUDA)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (BytesGroup XOR, UintGroup Z_{2^127}) |
| PRG (DPF) | AES-128 MMO (Matyas-Meyer-Oseas), mul=2, AES-NI (OpenSSL EVP_CIPHER_CTX) |
| PRG (DCF) | AES-128 MMO, mul=4, AES-NI (OpenSSL EVP_CIPHER_CTX) |
| PRG (AesSoft) | `Aes128Soft<2>` — see `doc/bench_aes128_soft.md` |
| Output groups | BytesGroup (XOR, 128-bit), UintGroup (Z_{2^127}) |
| Threading | Single-thread |
| Build | CMake, Release; requires OpenSSL |
| Toolchain | nvcc + g++ (C++20) |
| Bench framework | Google Benchmark 1.9.5 |

### fss GPU (C++/CUDA)

| Setting | Value |
|---------|-------|
| in_bits | 20 |
| out_bits | 128 (BytesGroup XOR, UintGroup Z_{2^127}) |
| PRG (DPF) | ChaCha<2> (ChaCha20, 12 rounds) |
| PRG (DCF) | ChaCha<4> (ChaCha20, 12 rounds) |
| PRG (AesSoft) | `Aes128Soft<2>` (shared-mem Te0+sbox) — see `doc/bench_aes128_soft.md` |
| Output groups | BytesGroup (XOR, 128-bit), UintGroup (Z_{2^127}) |
| GPU instances | 2^20 parallel gen/eval instances (1 thread each) |
| Threading | CUDA (256 threads/block, 2^20 total threads) |
| Build | CMake, Release; requires OpenSSL |
| Toolchain | nvcc + g++ (C++20) |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events) |

### torchcsprng (C++/CUDA)

See `doc/bench_aes128_soft.md` for implementation details and results.

| Setting | Value |
|---------|-------|
| Build | CMake, Release (standalone in `third_party/torchcsprng/`) |
| Toolchain | nvcc + g++ (C++14) |
| Bench framework | Google Benchmark 1.9.5 (UseManualTime, CUDA events for GPU) |

## Running

All scripts accept extra args passed through to the benchmark binary.
CPU scripts pin to `CPU_ID` (default 0) and set the scaling governor to `performance`.
GPU scripts also accept `GPU_ID` (default 0) via `CUDA_VISIBLE_DEVICES`.

```bash
# Run all CPU benchmarks
CPU_ID=0 bash third_party/bench_libdpf.sh
CPU_ID=0 bash third_party/bench_libfss.sh
CPU_ID=0 bash third_party/bench_google_dpf.sh
CPU_ID=0 bash third_party/bench_gpu_dpf_cpu.sh
CPU_ID=0 bash third_party/bench_fss_v060.sh
CPU_ID=0 bash third_party/bench_fss_v070_cpu.sh
CPU_ID=0 bash third_party/bench_fss_cpu.sh

# Run all GPU benchmarks
GPU_ID=0 CPU_ID=0 bash third_party/bench_gpu_dpf_gpu.sh
GPU_ID=0 CPU_ID=0 bash third_party/bench_ezpc.sh
GPU_ID=0 CPU_ID=0 bash third_party/bench_fss_v070_gpu.sh
GPU_ID=0 CPU_ID=0 bash third_party/bench_fss_gpu.sh

# Run torchcsprng (no dedicated script; build and run directly)
cd third_party/torchcsprng
cmake -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build -j
CUDA_VISIBLE_DEVICES=0 taskset -c 0 ./build/bench
```

## Results

### Hardware

| Component | Value |
|-----------|-------|
| CPU | 2× AMD EPYC 7352 (96 threads total), 3194 MHz (performance governor, `taskset -c 0`) |
| GPU | NVIDIA A30 (sm_80), 24 GB HBM2e, driver 580.126.09 |
| CUDA | 12.8, nvcc V12.8.93 |

### CPU

Single-threaded, performance governor. Criterion reports 100-sample median; Google Benchmark reports mean across repetitions.

| Library | DPF Gen | DPF Eval | DPF EvalAll | DCF Gen | DCF Eval | DCF EvalAll |
|---------|---------|----------|-------------|---------|----------|-------------|
| libdpf | 1712 ns | 674 ns | 215 µs | — | — | — |
| libfss | 111 µs | 8882 ns | 9303 ms | — | — | — |
| google_dpf | 5822 ns | 1511 ns | 73.0 ms | 17260 ns | 7045 ns | — |
| GPU-DPF | 723 µs | 30954 ns | 32.4 s | — | — | — |
| fss-rs 0.6.0 (bytes) | 657 ns | 530 ns | 59.9 ms | 1104 ns | 659 ns | 75.7 ms |
| fss-rs 0.6.0 (uint) | 648 ns | 527 ns | 60.6 ms | 1101 ns | 669 ns | 81.3 ms |
| fss 0.7.0 (bytes) | 2326 ns | 1479 ns | — | 2623 ns | 1519 ns | — |
| fss 0.7.0 (uint) | 2353 ns | 1484 ns | — | 3407 ns | 1572 ns | — |
| fss (bytes) | 1753 ns | 957 ns | 62.8 ms | 3364 ns | 1671 ns | — |
| fss (uint) | 1691 ns | 910 ns | 66.6 ms | 3762 ns | 1848 ns | — |

### GPU

CUDA event timing except where noted. Batch sizes vary per library (see Settings).
fss and fss 0.7.0 run 2^20 parallel instances in a single kernel launch; times are total wall time for that launch.

| Library | Batch | DPF Gen | DPF Eval | DPF EvalAll | DCF Gen | DCF Eval |
|---------|-------|---------|----------|-------------|---------|----------|
| GPU-DPF | 512 | 264 µs[^cpug] | — | 264.7 ms | — | — |
| EzPC | 1024 | 383 µs | 165 µs | 80.6 ms | 565 µs | 173 µs |
| fss 0.7.0 (bytes) | 2^20 | 426.7 ms | 159.9 ms | — | 745.0 ms | 252.0 ms |
| fss 0.7.0 (uint) | 2^20 | 430.7 ms | 159.7 ms | — | —[^uc] | —[^uc] |
| fss (bytes) | 2^20 | 9.909 ms | 4.441 ms | — | 9.398 ms | 4.347 ms |
| fss (uint) | 2^20 | 10.001 ms | 4.454 ms | — | 11.035 ms | 4.533 ms |

[^cpug]: GPU-DPF Gen uses CPU timer (`time.perf_counter`); Eval uses CUDA events.
[^uc]: Crash: CUDA misaligned address in fss 0.7.0 uint DCF (deferred error from DPF Eval kernel); bytes group unaffected.

See `doc/bench_aes128_soft.md` for AesSoft / torchcsprng GPU results.
