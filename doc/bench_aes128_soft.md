# AES-128 MMO Software PRG: T-table vs Textbook

Benchmark comparing two software AES-128 implementations used as PRGs
with Matyas-Meyer-Oseas (MMO) mode: `out = AES(key, seed) XOR seed`.

- `fss::prg::Aes128Soft` (`include/fss/prg/aes128_mmo_soft.cuh`):
  T-table optimization. Combines SubBytes + MixColumns into 4 uint32_t
  Te0 lookups per round. Tables (1024B Te0 + 256B sbox) in `__shared__`
  memory on GPU.

- `torchcsprng::Aes128Mmo` (`third_party/torchcsprng/aes128_mmo_soft.cuh`):
  Textbook byte-by-byte. Separate SubBytes (16 sbox lookups), ShiftRows
  (byte shuffles), and MixColumns (xtime per byte) per round. No lookup
  tables beyond the 256B sbox.
  Ported from [meta-pytorch/csprng](https://github.com/meta-pytorch/csprng).

Both pre-expand round keys in the constructor (key setup cost excluded).

Benchmark source: `third_party/bench_aes128_soft.cu`.
Build: `cmake -S third_party -B build/third_party -DCMAKE_BUILD_TYPE=Release`.

## Hardware

- GPU: NVIDIA A30 (sm_80), 24 GB HBM2e, driver 580.126.09
- CPU: 2x AMD EPYC 7413 (96 threads total), 2.3 GHz base
- CUDA: 12.8, nvcc V12.8.93
- Build flags: `-O3`, `CMAKE_CUDA_ARCHITECTURES=80`

## GPU Results

1M parallel AES-128-MMO operations. CUDA event timing (5 reps, median).

| Benchmark | Time (ms) | Throughput | Speedup |
|-----------|-----------|------------|---------|
| Aes128Soft mul=1 | 0.784 | 1.337 G/s | 69.9x |
| torchcsprng mul=1 | 54.82 | 19.13 M/s | 1x |
| Aes128Soft mul=2 | 2.463 | 425.7 M/s | 44.1x |
| torchcsprng mul=2 | 108.63 | 9.65 M/s | 1x |

### GPU Resource Usage

`nvcc -O3 -arch=sm_80 --ptxas-options=-v`:

| Kernel | Regs/thread | Shared mem/block | Max occupancy |
|--------|-------------|------------------|---------------|
| Aes128Soft mul=1 | 72 | 1280 B | 896 threads/SM (43%) |
| Aes128Soft mul=2 | 72 | 1280 B | 896 threads/SM (43%) |
| torchcsprng mul=1 | 128 | 0 | 512 threads/SM (25%) |
| torchcsprng mul=2 | 122 | 0 | 512 threads/SM (25%) |

## CPU Results

Single-threaded latency on the same machine (5 reps, median).

| Benchmark | Time/op | Throughput | Speedup |
|-----------|---------|------------|---------|
| Aes128Soft mul=1 | 70.7 ns | 14.15 M/s | 2.91x |
| torchcsprng mul=1 | 206 ns | 4.86 M/s | 1x |
| Aes128Soft mul=2 | 140 ns | 7.13 M/s | 3.00x |
| torchcsprng mul=2 | 421 ns | 2.38 M/s | 1x |

## Analysis

1. The T-table merges SubBytes + MixColumns into one 32-bit lookup per
   state byte (4 lookups per column). The textbook approach does 4
   separate sbox lookups + 4 xtime calls + 12 XORs per column.

2. On GPU, register pressure is the main factor. The textbook approach
   uses 122-128 registers per thread, capping occupancy at 25% (512
   threads/SM). The T-table variant uses 72 registers, allowing 43%
   occupancy and better latency hiding.

3. The 1280B shared memory cost for Te0 + sbox is negligible (A30 has
   164 KB shared memory per SM).

Aes128Soft is 3x faster on CPU and 44-70x faster on GPU.
