# myl7/fss

A header-only C++20 library for two-party function secret sharing, with CPU evaluation, CUDA evaluation, and PyTorch bindings.

Generate two function shares, evaluate each share independently, and combine the outputs to recover the function value.
The library provides these constructions:

| Function | C++ schemes |
| --- | --- |
| Point function | `Dpf`, `HalfTreeDpf`, `PackedHalfTreeDpf` |
| Comparison function | `Dcf`, `GrottoDcf`, `HalfTreeGrottoDcf` |
| Verifiable point function | `Vdpf` |
| Multiple point functions | `Dmpf`, `Vdmpf` |

Choose the output group, pseudorandom generator, hash, and input width when configuring a scheme.
Packed Half-Tree DPF stores several narrow outputs in each 128-bit block.

[API documentation](https://myl7.github.io/fss/) describes each construction, its paper references, and its input and backend requirements.

## Build and install

You need a C++20 toolchain, CMake 3.22 or newer, and CUDA Toolkit 12 or newer.
The CUDA toolkit is required for CPU use too.
OpenSSL backends require OpenSSL 3 development headers and libraries.
Parallel CPU evaluation requires OpenMP compiler and linker support.

```bash
git clone https://github.com/myl7/fss.git
cd fss
cmake -S . -B build -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build
cmake --install build
```

Link the installed target from your CMake project:

```cmake
find_package(fss REQUIRED)
target_link_libraries(your_target PRIVATE fss::fss)
```

Pass `-DCMAKE_PREFIX_PATH=/path/to/install` when configuring that project.
The [samples](samples/) also show how to use the headers directly.

## Generate and evaluate a point function

This example shares a function that returns 7 at input 42 and zero elsewhere.
The caller owns the seeds and correction storage.
The constant seeds make the example reproducible; applications must sample independent seeds using secure randomness.

```cpp
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <fss/prg/chacha.cuh>

constexpr int kInBits = 8;
using In = uint8_t;
using Group = fss::group::Bytes;
using Prg = fss::prg::ChaCha<2>;
using Dpf = fss::Dpf<kInBits, Group, Prg, In>;

Dpf dpf{Prg{1, 2}};
int4 s0s[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
int4 b_buf = {7, 0, 0, 0};
Dpf::Cw cws[kInBits + 1];
dpf.Gen(cws, s0s, In{42}, b_buf);

int4 y0 = dpf.Eval(false, s0s[0], cws, In{42});
int4 y1 = dpf.Eval(true, s0s[1], cws, In{42});
int4 value = (Group::From(y0) + Group::From(y1)).Into();
```

`value` equals `b_buf`.
The example keeps both parties together to show reconstruction; each deployed party receives only its own seed and the shared corrections.
Use `EvalRange` or `EvalAll` to evaluate an interval or the full domain.
Set `par_depth=0` for sequential CPU evaluation, or use a positive depth with OpenMP and a parallel-safe backend.

VDPF, DMPF, and VDMPF generation can request a retry.
Their `Gen` documentation specifies which caller-owned seeds to resample.
Callers choose the randomness source, retry limit, and failure handling.

## Evaluate on CUDA

Scheme headers provide host launch methods, and device-callable Core methods can be used inside your own kernels.
For example, after copying the corrections above to `device_cws` and allocating `device_ys`, launch DPF full-domain evaluation on your stream:

```cpp
cudaError_t status = dpf.EvalAllGpu(false, s0s[0],
    cuda::std::span<const Dpf::Cw>(device_cws, kInBits + 1),
    cuda::std::span<int4>(device_ys, size_t{1} << kInBits), stream);
```

Check the returned status and check execution errors when completing stream work.
Keep inputs, outputs, and backend resources alive until queued work completes.
The method's documentation specifies any temporary allocation and synchronization behavior.
`EvalAllGpuBatch` evaluates multiple keys; sequence methods evaluate multiple coordinates of a key.

ChaCha and software AES support host and device execution.
OpenSSL and AES-NI backends are host-only.
See [the CUDA sample](samples/dpf_dcf_gpu.cu) for direct calls from a kernel, [CUDA usage notes](doc/cuda_usage.md) for compiler limitations, and the API documentation for supported launch methods.

## Use Python

`fss_crypto` uses PyTorch tensors for seeds, corrections, outputs, and proofs.
It requires Python 3.10 or newer and PyTorch 2.6 or newer.
Its JIT compiler requires a CUDA toolkit compatible with the installed PyTorch CUDA major version, including for CPU operations.

Install the checkout and development dependencies:

```bash
uv sync --extra dev
```

```python
import secrets
import torch
from fss_crypto import Dpf, group, prg

dpf = Dpf(8, group.Bytes(), prg.ChaCha(nonce=(1, 2), mul=2))
words = [secrets.randbits(32) - 2**31 for _ in range(8)]
s0s = torch.tensor(words, dtype=torch.int32).reshape(2, 4)
b_buf = torch.tensor([7, 0, 0, 0], dtype=torch.int32)
cws = torch.empty(dpf.cws_shape, dtype=torch.uint8)
dpf.gen(cws, s0s, a=42, b_buf=b_buf)
y0 = dpf.eval(b=False, s0=s0s[0], cws=cws, x=42)
y1 = dpf.eval(b=True, s0=s0s[1], cws=cws, x=42)
assert torch.equal(torch.bitwise_xor(y0, y1), b_buf)
```

Use contiguous tensors on the same device for each operation.
For CUDA execution, allocate or transfer all tensors to the selected CUDA device first.

The [Python guide](doc/python.md) covers buffer layouts, retries, CUDA streams, JIT compilation, and reusable compilation caches.
Runnable [Python samples](samples/python/) cover point, comparison, verifiable, and multipoint functions.

## Run the samples

```bash
cmake -S samples -B build/samples
cmake --build build/samples
```

Start with [DPF and DCF on CPU](samples/dpf_dcf_cpu.cu), [Half-Tree DPF](samples/half_tree_dpf_cpu.cu), or [VDPF](samples/vdpf_cpu.cu).
The [samples directory](samples/) includes the other schemes and CUDA examples.

## Measure performance

```bash
cmake -S . -B build -DBUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_cpu bench_gpu
./build/bench_cpu --benchmark_filter=BM_DcfGen
./build/bench_gpu --benchmark_filter=BM_DpfEvalPointGpu
```

Select an idle GPU and set `CUDA_VISIBLE_DEVICES` before running CUDA benchmarks.
Point, batch, and full-domain timings measure different amounts of work.
See [benchmark and profiling commands](doc/running_benchmarks.md), [benchmark coverage](doc/benchmarks.md), [comparison methodology](doc/bench_figures.md), and [third-party comparisons](doc/bench_third_parties.md) for timing boundaries and reproduction commands.

Measured with the `decision` profile (0.3 s minimum, five interleaved repetitions, medians) on 2026-11-08 UTC (2026-11-09 HKT) at the unreleased tree after v1.3.0, over the 2^20 domain:

- CPU, cs659b (one core, `performance` governor, OpenSSL AES PRG, `Uint` group): DPF `Gen` 1.0 µs/key, point `Eval` 344 ns/query, `EvalAll` 27.7 ms per full-domain evaluation (about 37.9 M outputs/s); DCF `Gen` 1.8 µs/key, point `Eval` 519 ns/query, `EvalAll` 54.3 ms per evaluation (about 19.3 M outputs/s).
- GPU, cs11369a (A6000, SM 86, CUDA event time): in the native suite (`Uint` group, ChaCha PRG, 1,048,576 keys or queries per call), DPF point `Gen` is about 3.6 ns/key and point `Eval` about 2.6 ns/query, and one 1024-key batch full-domain `EvalAll` API call takes 136 ms (DPF) and 142 ms (DCF), about 7.9 and 7.5 G domain outputs/s per call, which is batch time and not single-key latency. The domain curves measure the byte group at 262,144 keys per call; their per-call values differ and are on the comparison site.

These are single-collection medians on shared hosts, not a strict comparison against historical settings.
The [comparison site](https://myl7.github.io/fss/bench/) carries the full domain sweeps, the third-party baselines with their original dates, and the exact per-run sampling.

## License

[Apache-2.0](LICENSE.txt).
