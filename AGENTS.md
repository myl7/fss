- Follow Google C++ Style Guide
- Start error messages with a lowercase letter unless it is a proper noun or variable name
- Never reorder `#include`
- Build, save perf data, or save flamegraphs to ./build
- Try searching the FlameGraph lib in ../
- In GPU device code, registers are limited and memory access is expensive.
  Avoid use of `memcpy`, `memset`, and `reinterpret_cast`.
  Prefer plain assignments. Prefer `int4` for types larger than 8B.

## Repo Architecture

- This is a header-only C++20/CUDA20 library. The CMake target `fss` is an
  `INTERFACE` target, so public library code lives under `include/fss`.
- `fss_crypto` contains PyTorch bindings. The wrappers JIT-compile small CUDA
  extension modules from `fss_crypto/_csrc` and include headers from
  `include/fss`.
- `src` contains tests and benchmarks. It is not the library source tree.
- `samples` contains small CPU and GPU integration examples.
- `test` contains Python binding tests. These tests may trigger JIT compilation
  through PyTorch.
- `third_party` contains benchmark ports and external comparison code. Do not
  treat it as the core API.
- Main scheme headers:
  - `include/fss/dpf.cuh`: 2-party distributed point function.
  - `include/fss/dcf.cuh`: 2-party distributed comparison function.
  - `include/fss/half_tree_dpf.cuh`: Half-Tree DPF variant.
  - `include/fss/grotto_dcf.cuh`: Grotto DCF variant.
  - `include/fss/vdpf.cuh`: verifiable DPF.
  - `include/fss/vdmpf.cuh`: verifiable distributed multi-point function.
- Core extension points are concepts:
  - `Groupable` in `include/fss/group.cuh`, with built-ins in
    `include/fss/group/`.
  - `Prgable` in `include/fss/prg.cuh`, with built-ins in
    `include/fss/prg/`.
  - `Hashable` and `XorHashable` in `include/fss/hash.cuh`, with built-ins in
    `include/fss/hash/`.
  - `Permutable` in `include/fss/prp.cuh`, used by VDMPF Cuckoo hashing.
- Outputs, seeds, correction words, and hash blocks are usually 16-byte `int4`
  values. The last word's LSB is a clamped bit and should stay zero unless a
  scheme explicitly stores a control bit there.
- CPU paths commonly use AES-128 MMO and may need OpenSSL. GPU paths commonly
  use ChaCha and must keep nonce lifetime explicit.
- `EvalAll` APIs use OpenMP on host. Device code paths are marked with
  `__host__ __device__` where they are intended to run on GPU.
- Python wheels install the public C++ headers as data files. Keep
  `pyproject.toml` and `fss_crypto/_jit.py` in sync if the header layout moves.

## File Index

- `CMakeLists.txt`: library target, tests, benchmarks, install rules, and
  external dependencies fetched for test or benchmark builds.
- `Makefile`: formatting, CPU/GPU benchmark, flamegraph, and PTX helper
  commands. Keep generated outputs under `build`.
- `Doxyfile`: generated API documentation config. Public docs should focus on
  `README.md`, `include/fss`, and examples instead of benchmark internals.
- `doc/doxygen_input_filter.py`: Doxygen math filter for `$...$` comments and
  Markdown.
- `include/fss/*.cuh`: public scheme and concept headers.
- `include/fss/group/*.cuh`: built-in output groups.
- `include/fss/prg/*.cuh`: PRG backends for CPU and GPU paths.
- `include/fss/hash/*.cuh`: hash and xor-hash backends for verifiable schemes.
- `include/fss/prp/*.cuh`: PRP backends used by Cuckoo hashing in VDMPF.
- `src/*_test.cu`: C++/CUDA tests registered through CTest and GTest.
- `src/bench_cpu.cu`: Google Benchmark CPU microbenchmarks.
- `src/bench_gpu.cu`: Google Benchmark GPU microbenchmarks. Use
  `CUDA_VISIBLE_DEVICES` to pin a run to a free GPU.
- `samples/dpf_dcf_cpu.cu` and `samples/dpf_dcf_gpu.cu`: small integration
  examples for users.
- `fss_crypto/*.py`: public Python wrappers and validation.
- `fss_crypto/_csrc/*.cuh`: CUDA binding implementation compiled by PyTorch JIT.
- `test/*.py`: Python integration and validation tests.
- `third_party/`: comparison benchmark ports. Read it for benchmark context, not
  for core library architecture.

## Build And Test

- Configure regular tests with `cmake -B build`.
- Build with `cmake --build build`.
- Run tests from `build` with `ctest --output-on-failure`.
- Configure benchmarks with `cmake -B build -DBUILD_BENCH=ON`.
- Run GPU benchmarks with `GPU_ID=1 CUDA_ARCH=86 make bench_gpu` when CMake
  cannot infer the CUDA arch or when a specific GPU is free.
- Capture a GPU Nsight Systems profile with
  `GPU_ID=1 GPU_PROFILE_BENCH=BM_DpfEval_Uint/20 make profile_gpu`.
- Run Python tests with `uv run --extra dev pytest`.
- Keep all build output, perf data, and flamegraphs under `./build`.

## Updating This File

- Update this file in the same change when repo architecture changes, including:
  new public scheme headers, new concept interfaces, changed source/test/sample
  ownership, changed Python binding layout, changed build targets, or changed
  benchmark layout.
- Keep this file short and navigational. Put API tutorials in `README.md` or
  generated docs, not here.
