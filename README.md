# myl7/fss

Function secret sharing (FSS) primitives including:

- 2-party distributed point function (DPF)
- 2-party distributed comparison function (DCF)

[Documentation](https://myl7.github.io/fss/)

Features:

- First-class support for GPU (based on CUDA)
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

### Compiler Warnings

You may see warnings like "integer constant is so large that it is unsigned" during compilation. These cannot be easily suppressed but are harmless and can be safely ignored.

## Benchmarks

Microbenchmarks for DPF/DCF `Gen`/`Eval` using [Google Benchmark](https://github.com/google/benchmark), covering both CPU (AES-128 MMO PRG) and GPU (ChaCha PRG) paths.

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

Run on Intel Xeon Platinum 8352V @ 2.10GHz (Ice Lake), single core, performance governor, pinned with `taskset -c 0`.

```
--------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations
--------------------------------------------------------------------
BM_DpfGen_Uint_Aes/14           5401 ns         5399 ns       135596
BM_DpfGen_Uint_Aes/17           6467 ns         6458 ns       111512
BM_DpfGen_Uint_Aes/20           7754 ns         7745 ns        94862
BM_DpfGen_Bytes_Aes/20          7767 ns         7755 ns        93902
BM_DpfGen_Uint_ChaCha/20      134016 ns       133877 ns         5130
BM_DpfEval_Uint_Aes/14          2883 ns         2880 ns       239868
BM_DpfEval_Uint_Aes/17          3515 ns         3511 ns       203638
BM_DpfEval_Uint_Aes/20          3170 ns         3166 ns       179563
BM_DpfEval_Bytes_Aes/20         2862 ns         2860 ns       337629
BM_DpfEval_Uint_ChaCha/20      65493 ns        65387 ns        11237
BM_DcfGen_Uint_Aes/14          12316 ns        12304 ns        58867
BM_DcfGen_Uint_Aes/17          14896 ns        14881 ns        49506
BM_DcfGen_Uint_Aes/20          17215 ns        17212 ns        41366
BM_DcfGen_Bytes_Aes/20         17804 ns        17797 ns        38546
BM_DcfGen_Uint_ChaCha/20      139071 ns       138851 ns         4857
BM_DcfEval_Uint_Aes/14          6041 ns         6034 ns       120311
BM_DcfEval_Uint_Aes/17          7385 ns         7373 ns        98980
BM_DcfEval_Uint_Aes/20          8753 ns         8750 ns        81551
BM_DcfEval_Bytes_Aes/20         8991 ns         8988 ns        79812
BM_DcfEval_Uint_ChaCha/20      69761 ns        69688 ns         9848
```

### GPU Results

Run on NVIDIA RTX A6000 (48GB VRAM), CUDA 12.6, driver 560.35.05. Each iteration runs 1M (2^20) Gen/Eval in parallel. The GPU was warmed up before running the benchmarks.

```
------------------------------------------------------------------------------------------
Benchmark                                Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------
BM_DpfGen_Bytes/14/manual_time     3742742 ns      3758600 ns          188 items_per_second=280.162M/s
BM_DpfGen_Uint/14/manual_time      3605138 ns      3620776 ns          195 items_per_second=290.856M/s
BM_DpfEval_Bytes/14/manual_time    1834773 ns      1846636 ns          385 items_per_second=571.502M/s
BM_DpfEval_Uint/14/manual_time     1829063 ns      1840391 ns          386 items_per_second=573.286M/s
BM_DcfGen_Bytes/14/manual_time     3683808 ns      3690193 ns          191 items_per_second=284.645M/s
BM_DcfGen_Uint/14/manual_time      3660811 ns      3669146 ns          192 items_per_second=286.433M/s
BM_DcfEval_Bytes/14/manual_time    1896609 ns      1904905 ns          371 items_per_second=552.869M/s
BM_DcfEval_Uint/14/manual_time     1958738 ns      1967174 ns          362 items_per_second=535.333M/s
BM_DpfGen_Bytes/17/manual_time     4581579 ns      4597150 ns          154 items_per_second=228.868M/s
BM_DpfGen_Uint/17/manual_time      4420796 ns      4432505 ns          159 items_per_second=237.192M/s
BM_DpfEval_Bytes/17/manual_time    2234414 ns      2243475 ns          316 items_per_second=469.285M/s
BM_DpfEval_Uint/17/manual_time     2232655 ns      2237366 ns          315 items_per_second=469.654M/s
BM_DcfGen_Bytes/17/manual_time     4508062 ns      4519748 ns          156 items_per_second=232.6M/s
BM_DcfGen_Uint/17/manual_time      4481349 ns      4493369 ns          156 items_per_second=233.987M/s
BM_DcfEval_Bytes/17/manual_time    2336961 ns      2352788 ns          300 items_per_second=448.692M/s
BM_DcfEval_Uint/17/manual_time     2387943 ns      2398429 ns          294 items_per_second=439.113M/s
BM_DpfGen_Bytes/20/manual_time     5405055 ns      5418053 ns          139 items_per_second=193.999M/s
BM_DpfGen_Uint/20/manual_time      5234761 ns      5240723 ns          100 items_per_second=200.31M/s
BM_DpfEval_Bytes/20/manual_time    2661169 ns      2670157 ns          261 items_per_second=394.028M/s
BM_DpfEval_Uint/20/manual_time     2643466 ns      2655424 ns          262 items_per_second=396.667M/s
BM_DcfGen_Bytes/20/manual_time     5308714 ns      5318537 ns          100 items_per_second=197.52M/s
BM_DcfGen_Uint/20/manual_time      5294843 ns      5306015 ns          100 items_per_second=198.037M/s
BM_DcfEval_Bytes/20/manual_time    2741223 ns      2749239 ns          252 items_per_second=382.521M/s
BM_DcfEval_Uint/20/manual_time     2833979 ns      2842343 ns          248 items_per_second=370.001M/s
```

### Flamegraph

Generate a CPU flamegraph with `perf` and [FlameGraph](https://github.com/brendangregg/FlameGraph):

```bash
perf record -g ./build/bench_cpu --benchmark_filter=BM_DpfEval_Uint/20
perf script | /path/to/FlameGraph/stackcollapse-perf.pl | /path/to/FlameGraph/flamegraph.pl > build/flamegraph.svg
```

Open `build/flamegraph.svg` in a browser. The graph is interactive: click a frame to zoom in.

## License

Apache License, Version 2.0

Copyright (C) 2026 Yulong Ming <i@myl.moe>
