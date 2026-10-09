# AES-128 MMO Software PRG: T-table vs Textbook

Benchmark comparing two software AES-128 implementations used as PRGs
with Matyas-Meyer-Oseas (MMO) mode: `out = AES(key, seed) XOR seed`.
Both use mul=2 (two outputs per call, matching DPF usage).

- `fss::prg::Aes128Soft<2>` (`include/fss/prg/aes128_mmo_soft.cuh`):
  T-table optimization. Combines SubBytes + MixColumns into 4 uint32_t
  Te0 lookups per round. Tables (1024B Te0 + 256B sbox) in `__shared__`
  memory on GPU.

- `torchcsprng::Aes128Mmo<2>` (`third_party/torchcsprng/torchcsprng/aes128_mmo_soft.cuh`):
  Textbook byte-by-byte. Separate SubBytes (16 sbox lookups), ShiftRows
  (byte shuffles), and MixColumns (xtime per byte) per round. No lookup
  tables beyond the 256B sbox.
  Ported from [meta-pytorch/csprng](https://github.com/meta-pytorch/csprng).

The standalone comparison in `third_party/torchcsprng/bench.cu` benchmarks raw
PRG calls for both implementations. On CPU, construction and key expansion are
outside the timed loop. On GPU, both kernels expand keys per thread inside the
timed launch. The fss kernel also initializes its shared tables inside that
launch. Each call produces two 16-byte blocks, and throughput counts calls.

The separate `fss/{CPU,GPU}/DPF-bytes/AesSoft/Eval` cases measure DPF evaluation
using the software PRG. Their timings are not raw PRG latency.

## Settings

| Setting     | `fss::prg::Aes128Soft<2>`                                          | `torchcsprng::Aes128Mmo<2>`                                                        |
| ----------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Algorithm   | T-table: Te0[256] + sbox[256]                                      | Textbook: sbox[256] only                                                           |
| GPU tables  | `__shared__` memory (1280B/block)                                  | none                                                                               |
| Bench name  | `fss-prg/{CPU,GPU}/AesSoft`                                        | `torchcsprng/{CPU,GPU}/AesSoft`                                                    |
| Test (CPU)  | raw PRG Gen (single call)                                          | raw PRG Gen (single call)                                                          |
| Test (GPU)  | raw PRG Gen, 2^20 parallel                                         | raw PRG Gen, 2^20 parallel                                                         |
| GPU threads | 256/block                                                          | 256/block                                                                          |
| Source      | `third_party/torchcsprng/bench.cu`                                  | `third_party/torchcsprng/bench.cu`                                                 |
| Build       | `python3 third_party/bench.py build --libraries torchcsprng --platform all --cuda-arch 120` | same command |

The harness checks a known AES-MMO output and agreement between implementations
on CPU, then checks a GPU sample against those CPU outputs before timing.
To generate raw results and a report:

```bash
python3 third_party/bench.py run --libraries torchcsprng --platform all \
  --cpu 24 --gpu 0 --cuda-arch 120 --repetitions 5 --run-id aes-soft
```

Replace the CPU, GPU, and architecture for your host. See
[the reproduction guide](bench_third_parties.md#running) for prerequisites,
governor handling, and report formats.

## Hardware

Measured on cs659b on 2026-10-10 (Asia/Hong_Kong).

- CPU: 2 × AMD EPYC 9115, 16 cores/socket, 64 hardware threads total.
- CPU cases: CPU 24, one thread, performance governor.
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, sm_120, 72 GB, GPU 0,
  driver 595.71.05. GPU host process: CPU 16, performance governor.
- CUDA: 13.2, nvcc V13.2.78. Release build, CUDA architecture 120.

## GPU Results

2^20 parallel raw PRG calls, each returning two 16-byte MMO blocks. Values are
medians over five repetitions. CUDA event timing includes per-thread
construction and key expansion, PRG generation, and output stores. The fss
kernel also initializes Te0 and sbox in shared memory during the timed launch.

| Benchmark | Batch time (ms) | Time/call (ns) | Throughput (calls/s) | Speedup |
| --- | --- | --- | --- | --- |
| fss T-table | 0.3939 | 0.3757 | 2662 M | 86.9× |
| torchcsprng textbook | 34.22 | 32.63 | 30.65 M | 1× |

### GPU Resource Usage (sm_120)

The raw harness was rebuilt with CUDA 13.2 and ptxas resource reporting.
`AesSoftKernel<true>` is the T-table kernel, and `AesSoftKernel<false>` is the
textbook kernel. The compiler reports:

| Kernel | Regs/thread | Stack frame | Shared mem/block | Spill stores/loads |
| --- | --- | --- | --- | --- |
| fss T-table | 80 | 624 B | 1280 B | 0 / 0 B |
| torchcsprng textbook | 123 | 608 B | 0 B | 0 / 0 B |

The source log is `build/third_party/aes-registers.log`. Stack-frame storage
and register spills are reported separately by ptxas.

## CPU Results

Single-threaded raw PRG latency, five repetitions, median. Construction,
table initialization, and key expansion occur before the timed loop. The
wall-time and CPU-time columns retain both Google Benchmark clocks.
Throughput is its native calls-per-second counter based on CPU time.

| Benchmark | Wall time/call (ns) | CPU time/call (ns) | Throughput (calls/s) | Speedup |
| --- | --- | --- | --- | --- |
| fss T-table | 82.23 | 82.2 | 12.17 M | 3.71× |
| torchcsprng textbook | 305.1 | 304.9 | 3.28 M | 1× |

## Software-AES DPF Evaluation

These separate cases evaluate a DPF point over a 2^20-point domain using
`Aes128Soft<2>` and the bytes group. They include the DPF traversal and are
reported separately from the raw PRG calls above.

| Platform | Batch | Total time | Time/key |
| --- | --- | --- | --- |
| CPU | 1 | 1.67 µs | 1.67 µs |
| GPU | 1048576 | 3.21 ms | 3.06 ns |

## Analysis

The textbook implementation's median raw latency is 3.71× the T-table
latency on CPU and 86.9× on GPU. The CPU ratio uses wall time, and the
GPU ratio uses CUDA event time. The GPU comparison includes each
implementation's key expansion and the T-table kernel's shared-memory setup.

The T-table combines SubBytes and MixColumns through Te0 lookups. The textbook
implementation performs the byte substitutions, row shifts, and MixColumns
arithmetic separately. The current T-table kernel uses 80 registers per
thread compared with 123 for the textbook kernel. Both report zero spill
stores and loads, with stack frames of 624 and 608 bytes respectively. The
T-table kernel uses 1280 bytes of shared memory per block. These compiler
counts provide context for the timing difference. Attributing the speedup to
occupancy or a specific bottleneck would require kernel profiling.

The raw source rows are in
`build/third_party/results/comparison-cpu-performance/summary.csv` and
`build/third_party/results/comparison-gpu-performance/summary.csv`. The separate
software-AES DPF rows come from those same reports. GPU-DPF and EzPC reduction
timings are in [the third-party comparison](bench_third_parties.md#gpu).
