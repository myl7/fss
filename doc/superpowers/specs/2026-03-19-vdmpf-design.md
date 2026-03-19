# VDMPF (Verifiable Distributed Multi-Point Function) Design

Based on Section 4 of "Lightweight, Maliciously Secure Verifiable Function Secret Sharing" (Castro & Polychroniadou, EUROCRYPT 2022).

Reference implementation: [myl7/vdmpf](https://github.com/myl7/vdmpf) (Rust).

## Overview

VDMPF is a verifiable FSS scheme for multi-point functions. A multi-point function $f: [N] \to \mathbb{G}$ is the sum of $t$ point functions $f_{\alpha_i, \beta_i}$. The construction packs point functions into Cuckoo-hash buckets, each containing a VDPF operating on a reduced domain. Evaluation looks up $\kappa=3$ buckets per input, evaluates the VDPF in each, and sums the results.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PRP interface | Generic `Permutable` concept | Follows library extensibility pattern (Prgable, Hashable) |
| Inner VDPF domain | Compile-time `bucket_bits` template param | Consistent with library; assert to prevent out-of-range eval |
| Eval API | `BatchEval` only (batch verifiable) | Verification is the point of VDMPF; single-point eval is limited |
| Normal CDF | `std::erfc` from `<cmath>` | Standard C++, no external dependency |
| Key structure | Flat struct, pre-allocated arrays | Zero heap allocation; `max_points` as template param |
| Host/device | All host-only (Gen, BatchEval, Verify) | Routing/dedup logic requires dynamic data structures |
| Random seeds | Caller-provided (sigma + 2*m per-bucket s0s) | Matches existing VDPF pattern |
| Architecture | Layered: CuckooHash utility + Vdmpf | CuckooHash reusable; clean separation |

## New Concepts

### Permutable (include/fss/prp.cuh)

```cpp
template <typename Prp>
concept Permutable = requires(Prp prp, const int4 seed, int4 x) {
    { prp.Permu(seed, x) } -> std::same_as<int4>;
};
```

16-byte seed, 16-byte input -> 16-byte output. Maps to AES-128 block cipher.

### Aes128Prp (include/fss/prp/aes128.cuh)

Host-only AES-128 encrypt using OpenSSL. Same dependency pattern as existing Aes128Mmo.

## CuckooHash Utility (include/fss/cuckoo_hash.cuh)

Namespace `fss::cuckoo_hash`.

### ChBucket

```cpp
constexpr int ChBucket(int t, int kappa, int lambda);
```

Computes bucket count $m$ from Lemma 5 of the paper:
- $\lambda = a_t \cdot e - b_t - \log_2(t)$
- $a_t = 123.5 \cdot \text{CDF}_\text{Normal}(x=t, \mu=6.3, \sigma=2.3)$
- $b_t = 130 \cdot \text{CDF}_\text{Normal}(x=t, \mu=6.45, \sigma=2.18)$
- $m = \lceil t \cdot \frac{\lambda + b_t + \log_2(t)}{a_t} \rceil$ solved from above

Uses `std::erfc` for Normal CDF: $\Phi(x) = 0.5 \cdot \text{erfc}(-x / \sqrt{2})$.

### PrpHash

```cpp
template <typename Prp, typename In>
    requires Permutable<Prp>
struct PrpHash {
    Prp prp;
    // Returns (bucket_index, within_bucket_index) for hash function k
    std::pair<int, int> Locate(int4 sigma, In x, int k, In n, int b_size);
};
```

Implements Equations (1) and (2) from the paper:
- $h_i(x) = \lfloor \text{PRP}(\sigma, x + n \cdot (i-1)) / B \rfloor$
- $\text{index}_i(x) = \text{PRP}(\sigma, x + n \cdot (i-1)) \mod B$

### Compact

```cpp
template <typename Prp, typename In>
    requires Permutable<Prp>
struct Compact {
    Prp prp;
    // Inserts t elements into m buckets. Algorithm 4 from paper.
    // table[i] = (alpha_j, k) if occupied, or empty.
    // Returns 0 on success, 1 on failure.
    int Run(std::span<const In> as, int m, int4 sigma, In n, int b_size,
            std::span<std::pair<In, int>> table);
};
```

## Vdmpf Class (include/fss/vdmpf.cuh)

```cpp
template <int in_bits, int max_points, int bucket_bits, typename Group,
    typename Prg, typename XorHash, typename Hash, typename Prp,
    typename In = uint>
    requires((std::is_unsigned_v<In> || std::is_same_v<In, __uint128_t>) &&
        in_bits <= sizeof(In) * 8 && Groupable<Group> && Prgable<Prg, 2> &&
        XorHashable<XorHash> && Hashable<Hash> && Permutable<Prp>)
class Vdmpf {
public:
    static constexpr int kappa = 3;
    static constexpr int m = cuckoo_hash::ChBucket(max_points, kappa, 127);
    static constexpr In n = In(1) << in_bits;
    static constexpr int b_size = (int)((uint64_t)n * kappa + m - 1) / m;

    using InnerVdpf = Vdpf<bucket_bits, Group, Prg, XorHash, Hash, int>;

    Prg prg;
    XorHash xor_hash;
    Hash hash;
    Prp prp;

    struct BucketKey {
        typename InnerVdpf::Cw cws[bucket_bits];
        cuda::std::array<int4, 4> cs;
        int4 ocw;
        int4 s0;
    };

    struct Key {
        int4 sigma;
        BucketKey bks[m];
    };

    int Gen(Key &k0, Key &k1, int4 sigma,
        cuda::std::span<const cuda::std::array<int4, 2>, m> s0s,
        std::span<const In> as, std::span<const int4> b_bufs, int t);

    void BatchEval(bool b, const Key &key, std::span<const In> xs,
        std::span<int4> ys, cuda::std::array<int4, 4> &pi);

    static bool Verify(cuda::std::span<const int4, 4> pi0,
        cuda::std::span<const int4, 4> pi1);
};
```

### Gen Algorithm (Figure 2, VerDMPF.Gen)

1. Set `k0.sigma = k1.sigma = sigma`.
2. Compute runtime `m_` from `t` via `ChBucket(t, 3, 127)`. Assert `m_ <= m`.
3. Compute runtime `B` = `ceil(n * kappa / m_)`. Assert `B <= (1 << bucket_bits)`.
4. Run `Compact` to insert `as[0..t]` into `m_` buckets. Return 1 on failure.
5. For each bucket `i` in `[0, m)`:
   - If `i < m_` and `table[i]` is occupied: `a' = index_k(alpha_j)`, `b_buf' = b_bufs[j]`.
   - Else: `a' = 0`, `b_buf' = 0` (zero function).
   - Call `inner_vdpf.Gen(bks[i].cws, bks[i].cs, bks[i].ocw, s0s[i], a', b_buf')`.
   - Retry (return 1) if inner Gen returns 1.
   - Set `k0.bks[i].s0 = s0s[i][0]`, `k1.bks[i].s0 = s0s[i][1]`.
6. Return 0.

### BatchEval Algorithm (Figure 2, VerDMPF.BVEval)

1. Parse `key`: `sigma`, `bks[0..m]`.
2. Build per-bucket input lists (heap-allocated temporaries):
   - For each input `xs[omega]`, for each `k` in `[0, kappa)`:
     - `(i_k, j_k) = PrpHash.Locate(sigma, xs[omega], k, n, b_size)`
     - Assert `j_k < (1 << bucket_bits)`.
     - Append `(j_k, omega)` to `inputs[i_k]`, deduplicating per bucket.
3. Initialize `ys[0..eta]` to zero. Initialize `pi = {0, 0, 0, 0}`.
4. For each bucket `i` in `[0, m)`:
   - For each `(j_ell, omega_ell)` in `inputs[i]`:
     - `pi_tilde = inner_vdpf.Eval(b, bks[i].s0, bks[i].cws, bks[i].cs, bks[i].ocw, j_ell, y)`
     - `ys[omega_ell] += y` (group addition).
     - Accumulate per-bucket proof (same as `Vdpf::Prove` logic).
   - Cross-bucket proof: `pi = pi XOR H'(pi XOR pi_bucket)`.
5. Output `ys`, `pi`.

### Verify

Delegates to `Vdpf::Verify`: byte-wise comparison of `pi0` and `pi1`.

## File Layout

| File | Contents |
|---|---|
| `include/fss/prp.cuh` | `Permutable` concept |
| `include/fss/prp/aes128.cuh` | `Aes128Prp` (host-only, OpenSSL) |
| `include/fss/cuckoo_hash.cuh` | `ChBucket`, `PrpHash`, `Compact` |
| `include/fss/vdmpf.cuh` | `Vdmpf` class |
| `src/vdmpf_test.cu` | Tests |

## Tests

1. **EvalAtAlpha**: Gen with t points, BatchEval on all alpha values, verify `ys0[i] + ys1[i] == beta_i`.
2. **EvalAtNonAlpha**: BatchEval on random non-alpha points, verify `ys0[i] + ys1[i] == 0`.
3. **VerifyBatchEval**: BatchEval on mixed points, verify `Verify(pi0, pi1) == true`.
4. **CuckooHashCompact**: Unit test Cuckoo hashing with various t values.

Test parameterization:
- Bytes group + ChaCha + Blake3 + Aes128Prp
- Uint127 group + ChaCha + Blake3 + Aes128Prp
- Small domain: `in_bits=16`, `max_points=8`, `bucket_bits=12`
