# Orca Shared Memory AES: Per-Bank T-table Replication to Eliminate Bank Conflicts

## Background

Standard AES intermediate rounds (SubBytes + ShiftRows + MixColumns) can be
merged into four uint32_t lookups from a 256-entry T-table plus XOR.  On a GPU,
shared memory has 32 banks.  When threads in the same warp access different
addresses in the same bank, accesses serialize (bank conflict).

Orca (IEEE S&P 2024) proposes replicating the T-table 32 times
(`T0[256][32]`), so that each thread reads from column `threadIdx.x & 31`.
Since 32 threads always land on 32 distinct banks, bank conflicts are
eliminated.

This document records the implementation of `Aes128ShmSoft`, a device-only PRG
class that applies this optimization, and its benchmark comparison against the
existing `Aes128Soft`.

## Design

### Class interface

```cpp
namespace fss::prg {
template <int mul>
class Aes128ShmSoft {
public:
    struct ShmContext {
        uint32_t t0[256][32];   // T-table replicated 32 times, ~32 KB
        uint8_t sbox[256];      // last-round S-box, 256 B
    };

    __device__ static void LoadShm(ShmContext &ctx);
    __device__ Aes128ShmSoft(const ShmContext &ctx, const uint8_t keys[][16]);
    __device__ cuda::std::array<int4, mul> Gen(int4 seed);
};
}
```

### Comparison with `Aes128Soft`

| Aspect | Aes128Soft | Aes128ShmSoft |
|--------|-----------|---------------|
| AES algorithm | T-table: 4x u32 lookup + rotation + XOR | Same, but from 32-copy table |
| Table storage | Shared memory, single copy | Shared memory, 32 copies |
| Bank conflict | Possible | Eliminated |
| Host support | `__host__ __device__` | `__device__` only |
| Construction | Needs external te0/sbox pointers | Needs external ShmContext |
| Shared memory | ~1 KB (te0[256] + sbox[256]) | ~33 KB (t0[256][32] + sbox[256]) |
| Byte rotation | Shift-based (`RotWord8/16/24`) | `__byte_perm` intrinsic |
| Byte swap | `reinterpret_cast` to `uint8_t*` | `__byte_perm(val, 0, 0x0123)` |

### Kernel usage pattern

```cuda
__global__ void ExampleKernel(...) {
    // 1. Declare and load shared memory (all threads participate)
    __shared__ fss::prg::Aes128ShmSoft<2>::ShmContext aes_shm;
    fss::prg::Aes128ShmSoft<2>::LoadShm(aes_shm);
    __syncthreads();  // must be before any early return

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kN) return;

    // 2. Per-thread PRG construction and use
    fss::prg::Aes128ShmSoft<2> prg(aes_shm, kAesSoftKeys);
    fss::Dpf<in_bits, Group, fss::prg::Aes128ShmSoft<2>, uint> dpf{prg};
    // ...
}
```

`__syncthreads()` must precede the early return guard.  If placed after it,
threads that exit early cause the remaining threads to deadlock.

### Shared memory budget

- `t0[256][32]` = 256 * 32 * 4 B = 32768 B
- `sbox[256]` = 256 B
- Total: 33024 B

This fits on all modern GPUs (V100: 96 KB, A6000/A100: 100+ KB, H100: 228 KB)
but reduces occupancy compared to the 1 KB footprint of `Aes128Soft`.

## Implementation details

### T-table lookup (middle rounds)

Each of the 9 middle rounds computes four state words via:

```cpp
int wTid = threadIdx.x & 31;
uint32_t t0 = ctx_->t0[s0 >> 24][wTid]
            ^ RotRight8(ctx_->t0[(s1 >> 16) & 0xff][wTid])
            ^ RotRight16(ctx_->t0[(s2 >> 8) & 0xff][wTid])
            ^ RotRight24(ctx_->t0[s3 & 0xff][wTid])
            ^ rk[r * 4];
// ... similarly for t1, t2, t3
```

`RotRight8/16/24` use `__byte_perm(x, x, selector)`:

| Function | Selector | Effect |
|----------|----------|--------|
| RotRight8 | `0x0321` | `(x >> 8) \| (x << 24)` |
| RotRight16 | `0x1032` | `(x >> 16) \| (x << 16)` |
| RotRight24 | `0x2103` | `(x >> 24) \| (x << 8)` |

### Byte order conversion

`int4` stores four little-endian ints.  AES state columns are big-endian u32.
The conversion in both directions is `__byte_perm(val, 0, 0x0123)` (byte
reverse).

### Key expansion

Reuses the existing `aes_detail::KeyExpansion` (byte-level), then converts the
176-byte round key array into 44 big-endian `uint32_t` values at construction
time.  This runs once per thread and is not on the hot path.

### Register usage

Compiled for sm_75 (local) and sm_52 (remote):

| Kernel | Registers | Stack | Spill |
|--------|-----------|-------|-------|
| `AesShmPrgTestKernel` (sm_75) | 72 | 624 B | 0 |
| `DpfEvalKernelAesShm` (sm_52) | 56 | 608 B | 0 |
| `DpfGenKernelAesShm` (sm_52) | 73 | 624 B | 0 |

No spills.  The 624 B stack frame comes from `KeyExpansion`'s 176-byte
temporary buffer and the `round_keys_[2][44]` member (352 B).

## Correctness

A test (`src/aes128_shm_soft_test.cu`) runs `Aes128ShmSoft<2>::Gen()` on GPU
for 1024 random seeds and compares each output byte-for-byte against
`Aes128Soft<2>::Gen()` on host with the same keys.  The test passes,
confirming identical AES encryption results.

## Benchmark results

Machine: 4x NVIDIA RTX A6000 (compute capability 8.6, Ampere), CUDA 12.6.
Block size 256, 1M DPF instances (in_bits=20, UintGroup).

```
BM_DpfEval_Uint_AesSoft/20       23.9 ms    43.8 M items/s
BM_DpfEval_Uint_AesShmSoft/20    23.7 ms    44.2 M items/s   (+0.7%)
BM_DpfGen_Uint_AesShmSoft/20     46.4 ms    22.6 M items/s
```

The 32-copy T-table shows a marginal ~0.7% improvement for DPF Eval over the
single-copy version.

### Analysis

The lack of significant speedup likely comes from several factors:

1. The single-copy `Aes128Soft` already uses shared memory for its T-table.
   With 256 threads per block, the 32 threads within each warp access T-table
   entries determined by AES state bytes, which are pseudo-random.  Random
   accesses across 256 entries in 32 banks have a low collision probability by
   chance (~2-3 conflicts per round on average), so the baseline already has
   limited bank conflict overhead.

2. The 33 KB shared memory footprint of `Aes128ShmSoft` reduces occupancy.  On
   A6000 (100 KB shared memory per SM), this allows at most 3 blocks per SM
   vs. potentially 4+ for the 1 KB `Aes128Soft`.  Lower occupancy reduces the
   GPU's ability to hide memory latency through warp switching.

3. The DPF computation is not purely AES-bound.  Each DPF level calls
   `prg.Gen()` once but also does control-bit logic, XOR corrections, and
   memory loads for correction words.  AES throughput improvements are diluted
   by these surrounding operations.

4. Orca targets a different workload (large-batch AES encryption with many
   rounds per thread) where bank conflicts dominate.  In DPF evaluation, each
   thread does 20 AES calls (one per tree level) with other work interleaved,
   which changes the performance profile.

## Files

- `include/fss/prg/aes128_shm_soft.cuh` -- implementation
- `src/aes128_shm_soft_test.cu` -- correctness test
- `src/bench_gpu.cu` -- benchmark (AesShmSoft kernels and registration)
- `doc/plans/2026-03-10-aes128-shm-soft.md` -- original implementation plan

## References

- Orca: N. Jawalkar, K. Gupta, A. Bhatia, N. Chandran, D. Gupta, R. Sharma.
  "Orca: FSS-based Secure Training and Inference with GPUs." IEEE S&P 2024.
- EzPC reference implementation:
  https://github.com/mpc-msri/EzPC/blob/master/GPU-MPC/fss/gpu_aes_shm.cu
