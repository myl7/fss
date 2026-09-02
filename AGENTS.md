- Follow Google C++ Style Guide
- Start error messages with a lowercase letter unless it is a proper noun or variable name
- Never reorder `#include`
- Build, save perf data, or save flamegraphs to ./build
- Try searching the FlameGraph lib in ../
- In GPU device code, registers are limited and memory access is expensive.
  Avoid use of `memcpy`, `memset`, and `reinterpret_cast`.
  Prefer plain assignments. Prefer `int4` for types larger than 8B.
- Write commit messages as [Scoped Commits](https://scopedcommits.com/)

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
- GPU-only entry points live outside the scheme headers:
  - `include/fss/eval_all_gpu.cuh`: batched full-domain eval for DPF and
    Half-Tree DPF.
  - `include/fss/point_eval_gpu.cuh`: level-major point eval and the relayout
    helpers that pack correction words for it.
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
- CPU benchmark names carry a PRG suffix and GPU names do not, so
  `BM_DpfEval_Uint_Aes/20` and `BM_DpfEval_Uint/20` name the same benchmark on
  the two binaries. A filter copied from one matches nothing in the other.
- Only the `EvalAll` benchmarks call `SetItemsProcessed`. Google Benchmark
  prints no throughput for every other row, so the `Items/s` column of the
  README tables is 1/Time there and counts keys, not outputs. Check the unit
  when refreshing those tables.
