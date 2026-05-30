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
- `src` contains tests and benchmarks. It is not the library source tree.
- `samples` contains small CPU and GPU integration examples.
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

## Build And Test

- Configure regular tests with `cmake -B build`.
- Build with `cmake --build build`.
- Run tests from `build` with `ctest --output-on-failure`.
- Configure benchmarks with `cmake -B build -DBUILD_BENCH=ON`.
- Keep all build output, perf data, and flamegraphs under `./build`.

## Updating This File

- Update this file in the same change when repo architecture changes, including:
  new public scheme headers, new concept interfaces, changed source/test/sample
  ownership, changed build targets, or changed benchmark layout.
- Keep this file short and navigational. Put API tutorials in `README.md` or
  generated docs, not here.
