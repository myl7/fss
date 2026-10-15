# dpf++ (rh3nry/dpfplusplus) CPU Benchmark

## Provenance

This entry benchmarks the header-only DPF++ library at https://github.com/rh3nry/dpfplusplus, pinned to commit `e3f7f0fba75d1736b0da3edb0ce3b22d1b3c73d5` ("ancient, yet functional copy of core dpf++ headers").
The library implements Boyle-Gilboa-Ishai style distributed point functions over `__m128i` nodes with a fixed-key AES-NI PRG (`dpf::prg::aes`), packing 128 one-bit leaves per node for `dpf::dpf_key<bool>`; `dpf::dpf_key<uint64_t>` supports wider point values.
It is the "dpf++" lineage: the pinned repository states that it temporarily holds an old copy of dpf++ and will later be replaced by a redirect to the latest version of libdpf++.
Upstream ships no examples or build files, so `third_party/dpfplusplus-bench/bench.cc` drives the header API (`gen`, `eval`, `evalfull`) directly.

## License status and why nothing is vendored

Every upstream header carries a notice that summarizes to: copyright (C) 2019 Anonymous; this is a pre-release version of the DPF++ library distributed anonymously for peer review; a public release under LGPL v2.1 is planned in the near future; please do not redistribute this version of the software.
Because redistribution is not permitted, this repository commits zero upstream bytes: the tree is not a Git submodule, and no vendored copy, patched fork, or `.patch` file with full hunks is stored here.
Instead, `third_party/dpfplusplus-bench/fetch.sh` clones upstream at bench time into the gitignored build directory and applies three scripted text fixes.
The fix script quotes only the tiny before/after snippets it needs to locate each edit, which keeps the committed material to interface-level fragments rather than a redistributable copy of the headers.

## How the bench acquires the code

`bench.py` runs `bash third_party/dpfplusplus-bench/fetch.sh build/third_party/dpfplusplus` during `prepare`.
The script clones the repository into `build/third_party/dpfplusplus/src`, checks out the pinned commit above, force-resets the tree, and applies the three fixes to `src/include/dpf.hpp`.
Everything lands under the repository's gitignored `build/` directory, so a fresh checkout contains no upstream bytes and a clean of `build/` removes them entirely.
The script is idempotent: re-runs re-fetch, force-reset to the pinned commit, and re-apply the fixes.
It runs under `set -euo pipefail` and each fix demands an exact match count, so upstream drift or a moved commit fails loudly instead of silently benchmarking something else.
The CMake wrapper never calls `add_subdirectory` on the fetched tree; it only adds the fetched `src/include` directory to the include path, passed as `-DDPFPP_INCLUDE_DIR=<build>/src/include`.

## The three text fixes

The authoritative implementation is the `python3` heredoc in `third_party/dpfplusplus-bench/fetch.sh`; the prose below describes what and why.

1. The pinned header includes `<bsd/stdlib.h>` unconditionally for `arc4random_buf`, which requires libbsd headers and linkage on Linux.
   The fix guards the include so Apple targets and glibc 2.36 or newer (which provide `arc4random_buf` in `<stdlib.h>`) use `<stdlib.h>`, keeping `<bsd/stdlib.h>` only for other systems.
   The bench host (Ubuntu 24.04, glibc 2.39) installs no libbsd headers, so the glibc branch of the guard is required there; the Apple branch covers macOS development hosts.
2. `dpf_key::nodes_per_leaf` is declared `constexpr` but initialized with `std::ceil` on a `double`, which is not a constant expression before C++23 and fails to compile under C++17.
   The fix replaces it with the equivalent integer ceil-div `(bits_per_leaf + bits_per_node - 1) / bits_per_node`.
3. Two `if constexpr(dpfkey.is_packed)` tests, in `dpf::eval` and `dpf::__reeval`, read the static member through a function parameter, which is not usable as a constant expression at those sites.
   The fix names the static class member `dpf_key<leaf_t, node_t, prg_t>::is_packed` instead, at both occurrences.

## Build and benchmark shape

`third_party/dpfplusplus-bench/CMakeLists.txt` builds the `bench` executable from `bench.cc` alone, links Google Benchmark 1.9.5 via FetchContent like the sibling wrappers, and compiles with C++17, `-O2`, `-maes`, and `-mavx2`.
The `-O2` flag is appended after the build-type flags, so it also pins `-O2` when the runner configures `CMAKE_BUILD_TYPE=Release`.
The upstream headers use x86 SSE/AVX/AES intrinsics unconditionally, so the wrapper refuses to configure on non-x86 hosts; the bench machine is x86 Linux.
Benchmark IDs are `dpfplusplus/CPU/<Op>/<nbits>` with `state.range(0)` carrying `nbits`: `Gen` and `Eval` at 16 and 20 bits, `EvalAll` at 16, 20, and 24 bits.
`Gen` times `dpf_key<bool>::gen` producing both parties' keys, including the internal `arc4random_buf` root draw.
`Eval` times one `key.eval` point evaluation per iteration and sweeps the input domain between iterations.
`EvalAll` times one party's `evalfull` over the full domain and reports every domain output as processed work via `SetItemsProcessed`; following the upstream API, `evalfull`'s internal `malloc`/`free` of the traversal hint array is inside the timed region.
Each benchmark first runs an untimed self-check: both parties' `evalfull` shares must XOR to exactly the alpha bit across the whole domain, and a `dpf_key<uint64_t>` variant is spot-checked by point evaluation (value at the target, zero away from it).

## Reproducing

Once `bench.py` registers the library under the name `dpfplusplus`, the rows reproduce with:

```bash
python3 third_party/bench.py prepare --libraries dpfplusplus --platform cpu
python3 third_party/bench.py build --libraries dpfplusplus --platform cpu --jobs 4
python3 third_party/bench.py run --libraries dpfplusplus --platform cpu --cpu <idle-cpu> --no-build --run-id dpfplusplus-cpu
```

A manual path without the runner:

```bash
bash third_party/dpfplusplus-bench/fetch.sh build/third_party/dpfplusplus
cmake -S third_party/dpfplusplus-bench -B build/third_party/dpfplusplus/cmake \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DDPFPP_INCLUDE_DIR="$PWD/build/third_party/dpfplusplus/src/include"
cmake --build build/third_party/dpfplusplus/cmake --parallel 16
./build/third_party/dpfplusplus/cmake/bench --benchmark_filter='^dpfplusplus/CPU/' \
  --benchmark_min_time=1s --benchmark_repetitions=5
```

Numbers from this entry must be measured on the same host as the rest of the comparison bench (cs11369a), under the same runner pinning (`taskset`, performance governor, single-threaded environment variables).
Do not compare smoke-mode timings with full measurements.
