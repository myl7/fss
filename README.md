# myl7/fss

Function secret sharing (FSS) primitives including:

- 2-party distributed point function (DPF), based on [Boyle et al. (CCS '16)](https://doi.org/10.1145/2976749.2978429) or [Half-Tree (EUROCRYPT '23)](https://doi.org/10.1007/978-3-031-30545-0_12).
- 2-party distributed comparison function (DCF), based on [Boyle et al. (EUROCRYPT '21)](https://doi.org/10.1007/978-3-030-77886-6_30) or [Grotto (CCS '23)](https://doi.org/10.1145/3576915.3623147).
- 2-party verifiable distributed point function (VDPF), based on [Castro & Polychroniadou (EUROCRYPT '22)](https://doi.org/10.1007/978-3-031-06944-4_6).
- 2-party verifiable distributed multi-point function (VDMPF), based on [Castro & Polychroniadou (EUROCRYPT '22)](https://doi.org/10.1007/978-3-031-06944-4_6).

[Documentation](https://myl7.github.io/fss/)

Features:

- First-class support for GPU (based on CUDA)
- Top-tier performance shown by benchmarks
- Well-commented and documented
- Header-only library, easy for integration

## Introduction

**Multi-party computation (MPC)** is a subfield of cryptography that aims to enable a group of parties (e.g., servers) to jointly compute a function over their inputs while keeping the inputs private.

**Secret sharing** is a method that distributes a secret among a group of parties, such that no individual party holds any information about the secret.
For example, a number $x$ can be secret-shared into $x_0, x_1$ via $x = x_0 + x_1$.

**FSS** is a scheme to secret-share a function into a group of function shares.
Each function share, called as a **key**, can be individually evaluated on a party.
The outputs of the keys are the shares of the original function output.
FSS consists of 2 methods: `Gen` for generating function shares as keys and `Eval` for evaluating a key to get an output share.
FSS's workflow is shown below:

[![](https://mermaid.ink/img/pako:eNpVkc1OwzAQhF_F2gNKhBPZJc2PBZVKoFzohd6QL6ax20iJXRkHGqq-O05KK2r54PF-M3PYA6xNJYGBasz3eiusQ69vXCN_5gEHRTBSFD2gxWoVv0gdqJBDeJo_Eg_05G_4_CWaYMD3_wg6EPSKoFdE6YGFsaj3jAr2Ib7_sDOfeYtGW38B5yiKZr4S3fjUc_8oUBRHqOQaMGxsXQFztpMYWmlbMUg4DDgHt5Wt5MD8s5JKdI3jwPXR23ZCvxvTnp3WdJstMCWaT6-6XSWcfKrFxor28mulrqQtTacdMJrRyZgC7AB7YGkWp3cZoSRJUn-LBEMPrMhiOkmKIiFFTvOcTo8YfsZaEufZFIOsamfs8rSOcSvHXweacxw?type=png)](https://mermaid.live/edit#pako:eNpVkc1OwzAQhF_F2gNKhBPZJc2PBZVKoFzohd6QL6ax20iJXRkHGqq-O05KK2r54PF-M3PYA6xNJYGBasz3eiusQ69vXCN_5gEHRTBSFD2gxWoVv0gdqJBDeJo_Eg_05G_4_CWaYMD3_wg6EPSKoFdE6YGFsaj3jAr2Ib7_sDOfeYtGW38B5yiKZr4S3fjUc_8oUBRHqOQaMGxsXQFztpMYWmlbMUg4DDgHt5Wt5MD8s5JKdI3jwPXR23ZCvxvTnp3WdJstMCWaT6-6XSWcfKrFxor28mulrqQtTacdMJrRyZgC7AB7YGkWp3cZoSRJUn-LBEMPrMhiOkmKIiFFTvOcTo8YfsZaEufZFIOsamfs8rSOcSvHXweacxw)

**DPF/DCF** are FSS for point/comparison functions.
They are called out because 2-party DPF/DCF can have $O(\log N)$ key size, where $N$ is the input domain size.
Meanwhile, 3-or-more-party DPF/DCF and general FSS have $O(\sqrt{N})$ key size.
More details, including the definitions and the implementation details that users must care about, can be found in the documentation of dpf.cuh and dcf.cuh files.

## Get Started

### Prerequisites

- CMake >= 3.22
- CUDA toolkit >= 12.0 (for C++20 support). Tested on the latest CUDA toolkit.
- OpenSSL 3 (only required for CPU with AES-128 MMO PRG)

### Build

Clone the repository:

```bash
git clone https://github.com/myl7/fss.git
cd fss
```

**Option A: Install via CMake and use `find_package`**

```bash
cmake -B build -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build
cmake --install build
```

Then in your project's `CMakeLists.txt`:

```cmake
find_package(fss REQUIRED)
target_link_libraries(your_target fss::fss)
```

When configuring your project, point CMake to the install prefix:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/install
```

**Option B: Use as a subdirectory (header-only)**

Without installing, you can define the target directly in your `CMakeLists.txt`, like the samples do:

```cmake
add_library(fss INTERFACE)
target_include_directories(fss INTERFACE "/path/to/fss/include")
target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)
```

Then link it in your project:

```cmake
target_link_libraries(your_target fss)
```

### CPU

This walks through using DPF and DCF on the CPU with AES-128 MMO PRG. This PRG requires OpenSSL.

1. Include the headers and set up type aliases:

   ```cpp
   #include <fss/dpf.cuh>
   #include <fss/dcf.cuh>
   #include <fss/group/bytes.cuh>
   #include <fss/prg/aes128_mmo.cuh>

   constexpr int kInBits = 8;  // Input domain: 2^8 = 256 values
   using In = uint8_t;
   using Group = fss::group::Bytes;

   // DPF uses mul=2, DCF uses mul=4
   using DpfPrg = fss::prg::Aes128Mmo<2>;
   using DcfPrg = fss::prg::Aes128Mmo<4>;
   using Dpf = fss::Dpf<kInBits, Group, DpfPrg, In>;
   using Dcf = fss::Dcf<kInBits, Group, DcfPrg, In>;
   ```

2. Create the PRG with AES keys and instantiate DPF/DCF:

   ```cpp
   // DPF PRG needs 2 AES keys
   unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
   unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
   const unsigned char *keys[2] = {key0, key1};
   auto ctxs = DpfPrg::CreateCtxs(keys);

   DpfPrg prg(ctxs);
   Dpf dpf{prg};
   ```

3. Run `Gen` to generate correction words (keys) from secret inputs:

   ```cpp
   In alpha = 42;                  // Secret point / threshold
   int4 beta = {7, 0, 0, 0};      // Secret payload (LSB of .w must be 0)

   // Random seeds for the two parties (LSB of .w must be 0)
   int4 seeds[2] = {
       {0x11111111, 0x22222222, 0x33333333, 0x44444440},
       {0x55555555, 0x66666666, 0x77777777, static_cast<int>(0x88888880u)},
   };

   Dpf::Cw cws[kInBits + 1];
   dpf.Gen(cws, seeds, alpha, beta);
   ```

4. Run `Eval` on each party and reconstruct using the group:

   ```cpp
   // Each party evaluates independently
   int4 y0 = dpf.Eval(false, seeds[0], cws, alpha);
   int4 y1 = dpf.Eval(true, seeds[1], cws, alpha);

   // Reconstruct via the group: convert to group elements, add, convert back
   // For Bytes group this is XOR; for Uint group this is arithmetic addition
   int4 sum = (Group::From(y0) + Group::From(y1)).Into();
   // sum == beta at x == alpha, 0 otherwise
   ```

5. Free the AES contexts when done:

   ```cpp
   DpfPrg::FreeCtxs(ctxs);
   ```

DCF follows the same pattern — use `DcfPrg` (mul=4, needs 4 AES keys), `Dcf`, and `Dcf::Cw`. The reconstructed output equals `beta` when `x < alpha` and `0` otherwise.

Link with OpenSSL in your `CMakeLists.txt`:

```cmake
find_package(OpenSSL REQUIRED)
target_link_libraries(your_target fss OpenSSL::Crypto)
```

See `samples/dpf_dcf_cpu.cu` for the complete working example.

### GPU

This walks through using DPF and DCF on the GPU with ChaCha PRG.

1. Include the headers and set up type aliases:

   ```cpp
   #include <fss/dpf.cuh>
   #include <fss/dcf.cuh>
   #include <fss/group/bytes.cuh>
   #include <fss/prg/chacha.cuh>

   constexpr int kInBits = 8;
   using In = uint8_t;
   using Group = fss::group::Bytes;

   // DPF uses mul=2, DCF uses mul=4
   using DpfPrg = fss::prg::ChaCha<2>;
   using DcfPrg = fss::prg::ChaCha<4>;
   using Dpf = fss::Dpf<kInBits, Group, DpfPrg, In>;
   using Dcf = fss::Dcf<kInBits, Group, DcfPrg, In>;
   ```

2. Set up a nonce in constant memory and create the PRG in a kernel:

   ```cpp
   __constant__ int kNonce[2] = {0x12345678, 0x9abcdef0};

   __global__ void GenKernel(Dpf::Cw *cws, const int4 *seeds, const In *alphas, const int4 *betas) {
       int tid = blockIdx.x * blockDim.x + threadIdx.x;

       DpfPrg prg(kNonce);
       Dpf dpf{prg};

       int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
       dpf.Gen(cws + tid * (kInBits + 1), s, alphas[tid], betas[tid]);
   }
   ```

3. Prepare host data, copy to device, and launch the `Gen` kernel:

   ```cpp
   int4 *d_seeds = /* cudaMalloc + cudaMemcpy seeds to device */;
   In *d_alphas = /* cudaMalloc + cudaMemcpy alphas to device */;
   int4 *d_betas = /* cudaMalloc + cudaMemcpy betas to device */;

   Dpf::Cw *d_cws;
   cudaMalloc(&d_cws, sizeof(Dpf::Cw) * (kInBits + 1) * N);

   GenKernel<<<blocks, threads>>>(d_cws, d_seeds, d_alphas, d_betas);
   ```

4. Write and launch an `Eval` kernel for each party, then copy results back:

   ```cpp
   __global__ void EvalKernel(int4 *ys, bool party, const int4 *seeds, const Dpf::Cw *cws, const In *xs) {
       int tid = blockIdx.x * blockDim.x + threadIdx.x;

       DpfPrg prg(kNonce);
       Dpf dpf{prg};

       ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (kInBits + 1), xs[tid]);
   }

   // Launch for party 0 and party 1, then copy d_ys back to host
   EvalKernel<<<blocks, threads>>>(d_ys, false, d_seeds0, d_cws, d_xs);
   EvalKernel<<<blocks, threads>>>(d_ys, true, d_seeds1, d_cws, d_xs);
   ```

5. Reconstruct on the host using the group, same as the CPU case:

   ```cpp
   int4 sum = (Group::From(h_y0s[i]) + Group::From(h_y1s[i])).Into();
   ```

DCF follows the same pattern — use `DcfPrg` (mul=4), `Dcf`, and `Dcf::Cw`.

See `samples/dpf_dcf_gpu.cu` for the complete working example.

### Python

The `fss_crypto` package exposes PyTorch wrappers for DPF and DCF. The first
use of each parameter set JIT-compiles a small CUDA extension with
`torch.utils.cpp_extension.load`, so the CUDA toolkit is required even when the
example below runs on CPU tensors.

```bash
uv sync --extra dev
uv run pytest
```

```python
import torch
import fss_crypto

dpf = fss_crypto.Dpf(in_bits=8, group="bytes", prg="chacha")

s0s = torch.tensor(
    [
        [0x11111111, 0x22222222, 0x33333333, 0x44444440],
        [0x55555555, 0x66666666, 0x77777777, -0x77777780],
    ],
    dtype=torch.int32,
)
beta = torch.tensor([7, 0, 0, 0], dtype=torch.int32)

cws = dpf.gen(s0s, alpha=42, beta=beta)
y0 = dpf.eval(party=0, s0=s0s[0], cws=cws, x=42)
y1 = dpf.eval(party=1, s0=s0s[1], cws=cws, x=42)

assert torch.bitwise_xor(y0, y1).equal(beta)
```

`gen` and `eval_all` are CPU-only. `eval` can run on CUDA tensors for ChaCha
PRG when CUDA is available. The JIT path sets a default `TORCH_CUDA_ARCH_LIST`
on machines with no visible GPU so CPU tests can still compile the extension.

### Compiler Warnings

You may see warnings like "integer constant is so large that it is unsigned" during compilation. These cannot be easily suppressed but are harmless and can be safely ignored.

### nvcc 12.8: `Uint` as a `__global__` kernel template argument

nvcc 12.8 fails to compile the stub file when `fss::group::Uint<__uint128_t, ...>` is used as a template argument to a `__global__` kernel — it emits a 128-bit integer literal that g++ cannot parse. `__device__` functions are not affected (no stub is generated for them).

Workaround: wrap the type in a plain aggregate struct that satisfies `Groupable` but has no `__uint128_t` non-type template parameter in its name. The struct must have no user-declared constructors to remain an aggregate. See `third_party/fss/bench.cu` for an example.

## Benchmarks

Microbenchmarks for DPF/DCF/VDPF/HalfTreeDpf `Gen`/`Eval`, GPU point eval, and full-domain GPU `EvalAll` using [Google Benchmark](https://github.com/google/benchmark), covering both CPU (AES-128 MMO PRG) and GPU (ChaCha PRG) paths.

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
./build/bench_cpu --benchmark_filter=BM_DpfEval_Uint/20
```

### CPU Results

Run on Intel Xeon Platinum 8352V @ 2.10GHz (Ice Lake), single core, performance governor, pinned with `taskset -c 0`. Per-key rows run one op per iteration, so `Avg per item` equals `Time`. `EvalAll` rows process 2^20 outputs per iteration and `Avg per item` is the reciprocal of `Items/s`.

| Benchmark | Time | Avg per item | Items/s |
| --- | --- | --- | --- |
| BM_DpfEval_Uint_Aes/20 | 1078 ns | 1078 ns | 927.6M/s |
| BM_DpfEval_Uint_Aes/14 | 751 ns | 751 ns | 1.332G/s |
| BM_DpfEval_Uint_Aes/17 | 931 ns | 931 ns | 1.074G/s |
| BM_DpfGen_Uint_Aes/20 | 2272 ns | 2272 ns | 440.1M/s |
| BM_DpfEval_Bytes_Aes/20 | 1076 ns | 1076 ns | 929.4M/s |
| BM_DpfEvalAll_Uint_Aes/20 | 78.7 ms | 74.9 ns | 13.34M/s |
| BM_DpfEval_Uint_ChaCha/20 | 3823 ns | 3823 ns | 261.6M/s |
| BM_DpfEval_Uint_AesSoft/20 | 4094 ns | 4094 ns | 244.3M/s |
| BM_DpfEval_Uint_AesRaw/20 | 333 ns | 333 ns | 3.003G/s |
| BM_DpfEval_Bytes_AesRaw/20 | 345 ns | 345 ns | 2.899G/s |
| BM_DpfGen_Uint_AesRaw/20 | 463 ns | 463 ns | 2.160G/s |
| BM_DpfGen_Bytes_AesRaw/20 | 428 ns | 428 ns | 2.336G/s |
| BM_DcfEval_Uint_AesRaw/20 | 343 ns | 343 ns | 2.915G/s |
| BM_DcfEval_Bytes_AesRaw/20 | 412 ns | 412 ns | 2.427G/s |
| BM_DcfGen_Uint_AesRaw/20 | 645 ns | 645 ns | 1.550G/s |
| BM_DcfGen_Bytes_AesRaw/20 | 698 ns | 698 ns | 1.433G/s |
| BM_DcfEval_Uint_Aes/20 | 1481 ns | 1481 ns | 675.2M/s |
| BM_DcfGen_Uint_Aes/20 | 3264 ns | 3264 ns | 306.4M/s |
| BM_DcfEval_Bytes_Aes/20 | 1772 ns | 1772 ns | 564.3M/s |
| BM_DcfEvalAll_Uint_Aes/20 | 97.8 ms | 93.2 ns | 10.73M/s |
| BM_DcfEvalAll_Bytes_Aes/20 | 98.6 ms | 93.9 ns | 10.64M/s |
| BM_VdpfEval_Uint_Aes_Sha256/20 | 2499 ns | 2499 ns | 400.2M/s |
| BM_VdpfGen_Uint_Aes_Sha256/20 | 4146 ns | 4146 ns | 241.2M/s |
| BM_VdpfEval_Uint_Aes_Blake3/20 | 1437 ns | 1437 ns | 695.9M/s |
| BM_VdpfProve_Uint_ChaCha_Blake3/20 | 181 ns | 181 ns | 5.525G/s |
| BM_VdpfEvalAll_Uint_Aes_Sha256/20 | 2138 ms | 2037 ns | 491k/s |
| BM_HalfTreeDpfEval_Uint_Aes/20 | 1017 ns | 1017 ns | 983.3M/s |
| BM_HalfTreeDpfGen_Uint_Aes/20 | 2236 ns | 2236 ns | 447.2M/s |
| BM_HalfTreeDpfEvalAll_Uint_Aes/20 | 86.9 ms | 82.8 ns | 12.08M/s |
| BM_GrottoDcfEval_Aes/20 | 17.0 ns | 17.0 ns | 58.82G/s |
| BM_GrottoDcfPreprocess_Aes/20 | 57.7 ms | — | — |
| BM_GrottoDcfPreprocessEvalAll_Aes/20 | 119.9 ms | 114.2 ns | 8.758M/s |

### GPU Results

Run on NVIDIA RTX PRO 5000 (72GB VRAM, Blackwell, sm_120), CUDA 13.2, driver 595.71.05. The GPU was warmed up before running the benchmarks. Each iteration runs 1M (2^20) keys in parallel. `Time` is the whole batch. `Avg per item` is the reciprocal of `Items/s`: per key for `Eval`/`Gen`/point-eval rows, per output for `EvalAll` rows (2^40 outputs per iteration).

| Benchmark | Time | Avg per item | Items/s |
| --- | --- | --- | --- |
| BM_DpfEval_Uint/20 | 1398.8 µs | 1.334 ns | 749.6M/s |
| BM_DpfEval_Uint/14 | 765.7 µs | 0.730 ns | 1.369G/s |
| BM_DpfEval_Uint/17 | 956.8 µs | 0.912 ns | 1.096G/s |
| BM_DpfGen_Uint/20 | 1965.4 µs | 1.874 ns | 533.5M/s |
| BM_DpfEval_Bytes/20 | 1398.9 µs | 1.334 ns | 749.6M/s |
| BM_DpfEval_Uint_AesSoft/20 | 3087.5 µs | 2.944 ns | 339.6M/s |
| BM_DcfEval_Uint/20 | 1421.2 µs | 1.355 ns | 737.8M/s |
| BM_DcfGen_Uint/20 | 1969.1 µs | 1.878 ns | 532.5M/s |
| BM_VdpfEval_Uint/20 | 1241.3 µs | 1.184 ns | 844.7M/s |
| BM_VdpfGen_Uint/20 | 2132.2 µs | 2.033 ns | 491.8M/s |
| BM_HalfTreeDpfEval_Uint/20 | 1001.7 µs | 0.955 ns | 1.047G/s |
| BM_HalfTreeDpfGen_Uint/20 | 1961.5 µs | 1.871 ns | 534.6M/s |
| BM_DpfEvalAllGpu_Uint/20 | 71.3 s | 64.8 ps | 15.43G/s |
| BM_HalfTreeDpfEvalAllGpu_Uint/20 | 90.9 s | 82.7 ps | 12.09G/s |
| BM_DpfEvalPointGpu_Uint/20 | 1024.5 µs | 0.977 ns | 1.024G/s |
| BM_DcfEvalPointGpu_Uint/20 | 1120.6 µs | 1.069 ns | 935.7M/s |
| BM_HalfTreeDpfEvalPointGpu_Uint/20 | 1000.9 µs | 0.955 ns | 1.048G/s |
| BM_VdpfEvalPointGpu_Uint/20 | 1109.2 µs | 1.058 ns | 945.3M/s |

GPU kernel register usage (compiled for sm_52, `--ptxas-options=-v`):

| Kernel          | Group      | Registers | Stack | Smem  |
| --------------- | ---------- | --------- | ----- | ----- |
| DpfEval         | Uint/Bytes | 39        |       |       |
| DpfGen          | Uint/Bytes | 48        |       |       |
| DpfEvalAes      | Uint       | 72        | 992B  | 1280B |
| DpfGenAes       | Uint       | 72        | 992B  | 1280B |
| HalfTreeDpfEval | Uint       | 41        |       |       |
| HalfTreeDpfGen  | Uint       | 47        |       |       |
| VdpfEval        | Uint       | 38        |       |       |
| VdpfGen         | Uint       | 72        |       |       |
| DcfEval         | Uint       | 38        |       |       |
| DcfGen          | Uint       | 56        |       |       |

The AES-based kernels use shared memory and spill to stack. All other kernels have zero spills.

### Flamegraph

Generate a CPU flamegraph with `perf` and [FlameGraph](https://github.com/brendangregg/FlameGraph):

```bash
perf record -g ./build/bench_cpu --benchmark_filter=BM_DpfEval_Uint/20
perf script | /path/to/FlameGraph/stackcollapse-perf.pl | /path/to/FlameGraph/flamegraph.pl > build/flamegraph.svg
```

Open `build/flamegraph.svg` in a browser. The graph is interactive: click a frame to zoom in.

## License

Apache License, Version 2.0

Copyright (C) 2026 Yulong Ming <i@myl7.org>
