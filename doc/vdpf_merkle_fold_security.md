# VDPF and VDMPF Merkle-fold proof accumulation: security analysis

This document analyzes the modification to the 2-party VDPF of de Castro and Polychroniadou
([dCP22], EUROCRYPT 2022, eprint 2021/580) implemented in this library: `Gen`, `Eval`, and
`Verify` are kept exactly as in the paper/implementation, and only the `Prove` accumulation is
replaced by an order-independent binary Merkle fold. The new algorithm ships as `ProveMerkle`, `EvalAllMerkle`, and `VdpfEvalAllMerkleGpu`, with the VDMPF extension shipping as `BatchEvalMerkle` and `VdmpfBatchEvalGpu` (Section 8); the paper-faithful chains `Prove` and VDMPF `BatchEval` remain implemented as the references. Both parties must switch jointly — proofs from the two accumulators are not
compatible, and `Verify` compares within one version only.

The document is self-contained for a reader with [dCP22] at hand. It states what is proven here,
what is inherited from the paper verbatim, and what is assumed of the hashes. Where the paper
asserts a step in one sentence, we say so; where our own reduction is complete, we write it out.

## 1. Overview

The chain accumulation of Figure 1 is inherently sequential: each step reads the running
accumulator, so the n domain points must be folded one after another, each step one Blake3
compression deep. On the GPU pipeline this dominates everything else. At `in_bits = 20`
(n = 2^20), the shipped `VdpfEvalAllProveGpu` path runs 185.4 ms per party
(`BM_VdpfEvalAllGpu_Uint_ChaCha_Blake3/20`), of which roughly 178 ms is the sequential
host-side chain; the GPU tree walk, the per-leaf hash kernel, and the device-to-host copy of the
64B per-point values account for the remaining ~7 ms. A single GPU thread running the chain is
about 10x slower than CPU SIMD Blake3 (see the comment in `src/bench_gpu.cu`), so the chain has
to stay on the host, serialized.

The Merkle fold removes the serial dependency. The per-point proof values are hashed into leaves
independently, and adjacent nodes are combined level by level: O(n) hash work at depth
log2(n), level-parallel on the GPU and under OpenMP on the CPU. The device-to-host copy of the
per-point values disappears as well, since the fold can run entirely on device. The whole
pipeline becomes kernel-bound. The cost is roughly 2n-1 `HashMerkle` calls against the chain's
n `Hash` calls — about twice the hash count, bought back many times over by parallelism.

This is a performance-motivated deviation from Figure 1's `Prove`. The security question is
whether the deviation preserves the soundness statement that [dCP22]'s Theorem 3 needs. Sections
4-6 answer yes, under hash assumptions no stronger than the paper's, and with the binding step of
the reduction written out in full — fuller, in fact, than the paper's own one-sentence treatment
of the chain accumulator (Section 5, Remark 2).

## 2. The original construction

Notation follows the library. Parties are b ∈ {0,1}; the input domain is
{0,...,n-1} with n = 2^in_bits; the output domain is a group of 16B elements whose last word's
least significant bit is clamped to zero, i.e. λ = 127 nominal. Concrete widths: seeds and group
elements are 16B (`int4`); the paper's H is `XorHash`, 32B in (x packed into 16B, seed 16B) to
64B out, matching the paper's H: (x, s) ↦ 4λ bits; the paper's H' is `Hash`, 64B in to 32B out,
matching H': 4λ ↦ 2λ bits. Both are single fixed-key Blake3 compressions
(`include/fss/hash/blake3.cuh`). `⊕` is XOR and `||` concatenation.

**Gen** (per level i = 0..in_bits-1, `Cw` = {s_cw with tl_cw in its LSB, tr_cw}): expand both
parties' seeds with the PRG, compute s_cw from the off-path child pair and the two control bits,
advance along the bit of a. After the loop, with s_b the two leaf seeds at a:

```
pi_tilde_b = XorHash(a || s_b)                // 64B
cs = pi_tilde_0 XOR pi_tilde_1                // 64B correction seed, part of both keys
if t_0 == t_1: retry with fresh seeds          // library returns 1, caller resamples
ocw = +-(beta - Group(s_0) + Group(s_1))       // sign from t_1
```

The key for party b is (s_b^(0), cws, cs, ocw); the honest case has the two keys identical
except for the initial seed.

**Eval** walks the correction words for in_bits levels, tracking the seed s (LSB clamped) and
the control bit t separately, then computes

```
y_b(x)   = (-1)^b (Group(s) + t * ocw)         // output share
pi_tilde = XorHash(x || s)                     // s is the clamped leaf seed
C_b(x)   = pi_tilde XOR (t ? cs : 0)           // Eval's return: corrected per-point hash
```

C_b(x) is what the paper calls correct(π̃, cs, t); it is the input to both accumulators.

**Prove** (the chain, Figure 1 as implemented; `Vdpf::Prove`):

```
pi = cs                                        // 64B = 4 x int4
for x = 0 .. n-1:
    h = Hash(pi XOR C(x))                      // 64B -> 32B
    pi[0] ^= h[0]; pi[1] ^= h[1]               // only the first 32B ever change
```

The final proof is `pi = (pi[0:2], cs[2:3])`: a 2λ-bit (32B) accumulated half, followed by the
last 32B of cs verbatim. The paper exchanges a 2λ-bit proof; the library's 64B shape means
`Verify` — plain 64B equality between the parties — also compares cs[2:3] across keys. This is an
implementation convention, not a security mechanism; see Assumption 1.

**Verify** accepts iff the two 64B proofs are byte-equal.

Two implementation conventions differ from Figure 1's letter and matter for the analysis:

1. The library clamps the seed's LSB and tracks the control bit t through the correction-word
   walk; Figure 1 defines the leaf control bit as LSB(s) (`t_b^(n+1) = LSB(s_b^(n+1))` in Gen,
   and BVEval sets t ← LSB(s) before computing π̃ and y). For honestly generated keys the two
   conventions coincide at every leaf. For adversarial keys they differ, which is why soundness
   needs Assumption 2 in Section 5.
2. The proof carries cs[2:3] verbatim, as above.

## 3. The Merkle-fold Prove

`ProveMerkle` replaces only the accumulation. `HashMerkle` is the existing Blake3 `Hash`
(64B in, 32B out, `array<int4,2>`); no hash code changes.

```
leaf:      L(x) = HashMerkle(C(x))                  // x = 0..n-1, leaves indexed by domain point
internal:  N = HashMerkle(N_left || N_right)        // two 32B children, 64B input
root:      the single node at depth in_bits
proof:     pi = { root[0], root[1], cs[2], cs[3] }  // 64B, same shape as the chain's
```

The leaves are indexed by the domain point x in {0,...,n-1} in domain order, forming a complete
binary tree of depth in_bits. `Verify` is unchanged. Tree for n = 8:

```
x:        0      1      2      3      4      5      6      7
          |      |      |      |      |      |      |      |
leaf:     L(0)   L(1)   L(2)   L(3)   L(4)   L(5)   L(6)   L(7)
           \    /        \    /        \    /        \    /
level 1:   H(L0||L1)     H(L2||L3)     H(L4||L5)     H(L6||L7)
              \           /                \           /
level 2:      H(N00||N01)                  H(N10||N11)
                    \                        /
level 3:            H(M0||M1) = root

pi = { root[0], root[1], cs[2], cs[3] }
```

The fold makes 2n-1 `HashMerkle` calls total: n leaves and n-1 internal nodes. Each level is an
embarrassingly parallel batch — one kernel per level on GPU (`VdpfEvalAllMerkleGpu`), one OpenMP
parallel-for per level on CPU (`EvalAllMerkle`). `ProveMerkle` is the standalone accumulation
over a supplied sequence of per-point values, mirroring `Prove`.

## 4. Correctness

**Theorem 1 (completeness).** Let (k_0, k_1) be honestly generated keys for f_{a,b} with
t_0 ≠ t_1 at the leaf (Gen's retry condition). Then both parties, evaluating the same full
domain with the same `HashMerkle` IV, produce identical 64B proofs, and `Verify` accepts.

*Proof.* For every x ≠ a, the DPF invariant of the correction-word machinery gives equal leaf
seeds and equal control bits for the two parties ([dCP22], Lemma 1 / the standard BGI invariant;
we do not reprove it): s_0(x) = s_1(x) and t_0(x) = t_1(x). Then π̃_0(x) = π̃_1(x) (same `XorHash`
input) and the same correction is applied or not applied, so C_0(x) = C_1(x). At x = a the seeds
differ, but Gen sets cs = π̃_0(a) ⊕ π̃_1(a) and the retry condition gives t_0(a) ≠ t_1(a), so the
party with control bit 1 has C_b(a) = π̃_b(a) ⊕ cs equal to the other party's π̃_{1-b}(a).
So the two parties' sequences (C_b(x))_{x=0..n-1} are identical, their leaf sequences and trees
are identical, the roots coincide, cs[2:3] is shared key material, and the 64B proofs are equal.
`Verify` accepts. ∎

The proof uses nothing about the accumulator beyond determinism as a function of the C-sequence;
it is the same argument as for the chain.

## 5. Soundness

Threat model: a static malicious client (PPT adversary) outputs a key pair (k_0, k_1); the two
servers honestly run the prescribed evaluation on the full domain and compare proofs. The target
property is [dCP22] Lemma 2's share integrity, which is the integrity half of Theorem 3
(share integrity + function privacy): except with negligible probability, `Verify` accepting
implies the parties' output shares form a 2-party sharing of some point function f_{α,β}.

### Assumptions

**Assumption 1 (common key material).** The two keys carry identical cws, cs, and ocw; only the
initial seeds differ, as Gen produces. This is a protocol-level assumption about key distribution
(send the common arrays once, or have the servers compare them). [dCP22]'s Lemma 2 proof uses
exactly this ("The servers also have the same correction seed cs") without deriving it from
`Verify`, and we inherit it unchanged. We note for honesty that `Verify` compares cs[2:3]
verbatim in both accumulators, which incidentally rejects key pairs disagreeing on that half of
cs, but this is not load-bearing and does not extend to cws or ocw; soundness rests on the
protocol assumption in both the paper and this library.

**Assumption 2 (nonzero correction seed).** cs ≠ 0. This assumption is specific to this
implementation's bookkeeping and deserves an explicit flag. Figure 1 defines the leaf control
bit as LSB(s), which makes "control bits differ ⇒ seeds differ" definitional. This library
clamps the seed, hashes XorHash(x || s_clamped) — the control bit is *not* in the `XorHash`
input — and applies both the output correction and the cs correction on the separately walked
control bit. With that bookkeeping, equal seeds with differing control bits are constructible
*if cs = 0*: take equal initial seeds, correction words whose seed parts are all zero and whose
control parts are all one. Both parties then walk identical seeds at every point while their
control bits differ at every leaf; every C_b(x) = π̃(x) matches (cs = 0, so the correction is a
no-op), both accumulators output identical proofs, `Verify` accepts, and the output shares sum
to ±ocw at every point of the domain — not a point-function sharing. With cs ≠ 0 the implication
is restored wherever the per-point values match: C_0(x) = C_1(x) with equal π̃ forces t_0(x) =
t_1(x), since differing control bits would XOR cs into exactly one side and make the values
differ by cs ≠ 0. Honest Gen outputs cs = 0 with probability 2^{-512} (cs is the XOR of two
512-bit `XorHash` outputs), and servers can reject cs = 0 keys syntactically at zero cost.
This is a property of `Eval`, not of the accumulator: the reference chain carries the same
exposure, and the paper's own construction is immune to it because of the LSB convention.

**Hash assumptions.** H = `XorHash` is collision-resistant and XOR-collision-resistant
([dCP22], Definition 11); `HashMerkle` is collision-resistant. Section 6 discusses the
instantiation. Under these assumptions the failure probability of everything below is bounded by
the sum of the adversary's collision-finding advantages against the two hashes.

### The reduction

**Lemma 1 (fold binding).** Let (L_b(x))_{x=0..n-1} be the two parties' leaf sequences. If the
roots are equal and the leaf sequences differ in at least one position, then a literal collision
of `HashMerkle` exists: two distinct 64B inputs with the same 32B output.

*Proof.* Pick a position x* with L_0(x*) ≠ L_1(x*) and consider the path from leaf x* to the
root. Let j be the largest level on this path at which the two parties' node values differ;
level 0 qualifies, so j is well defined. Since the roots are equal, j < in_bits. At level j the
path nodes differ (32B values), and at level j+1 they are equal. The level j+1 node of party b
is HashMerkle(c_b^left || c_b^right), where the pair (c_b^left, c_b^right) are its level-j
children; the path child occupies the same 32B slot in both parties' inputs and differs, so the
two 64B concatenations are distinct. They have the same output. That is a collision. ∎

The induction extracts the collision along the topmost differing edge of the sibling path; it
never refers to the order in which the tree is computed, only to the positional structure.

**Lemma 2 (leaf preimages).** If L_0(x) = L_1(x) for every x, and C_0(x) ≠ C_1(x) for some x,
then that x exhibits a literal `HashMerkle` collision (distinct 64B preimages C_0(x), C_1(x)
with equal 32B images). ∎

**Theorem 2 (soundness of the Merkle fold).** Under Assumptions 1 and 2 and the hash assumptions
above, for any PPT adversary outputting keys (k_0, k_1), except with negligible probability
(bounded by the collision-finding advantages against `HashMerkle` and H), if the two parties
run the full-domain evaluation with the Merkle-fold accumulator and `Verify` accepts, then there
exist α ∈ {0,...,n-1} and β in the output group such that y_0(x) + y_1(x) = β if x = α and 0
otherwise.

*Proof.*

**Step 1 (acceptance ⇒ root equality).** `Verify` accepts, so the 64B proofs are equal; in
particular the first 32B are equal, i.e. root_0 = root_1.

**Step 2 (root equality ⇒ per-point value agreement).** By Lemma 1, if any leaf differed we
would have a `HashMerkle` collision; ruling that out, L_0(x) = L_1(x) for all x. By Lemma 2,
if any preimage pair C_0(x) ≠ C_1(x) hid behind equal leaves we would again have a collision;
ruling that out, C_0(x) = C_1(x) for all x. These two steps are the fully written-out
replacement for the step the paper asserts for the chain (Remark 2 below).

**Step 3 (per-point agreement ⇒ at most one differing seed; paper's Lemma 3 machinery).**
Fix x and write π̃_b(x) = H(x || s_b(x)). Consider three cases, assuming C_0(x) = C_1(x).

- If s_0(x) = s_1(x): π̃_0(x) = π̃_1(x), and by Assumption 2 (as shown there) t_0(x) = t_1(x).
  The output shares sum to zero: y_0(x) + y_1(x) = (g + t_0 ocw) - (g + t_1 ocw) = 0.
- If s_0(x) ≠ s_1(x) and t_0(x) = t_1(x): C-equality gives H(x || s_0(x)) = H(x || s_1(x)) on
  distinct inputs — a plain collision of H.
- If s_0(x) ≠ s_1(x) and t_0(x) ≠ t_1(x): C-equality gives π̃_0(x) ⊕ π̃_1(x) = cs (with cs ≠ 0,
  since cs = 0 reduces to the previous case).

Suppose two points u ≠ v both had differing seeds. Each is in the third case (the second is a
collision), so π̃_0(u) ⊕ π̃_1(u) = π̃_0(v) ⊕ π̃_1(v) = cs ≠ 0. By collision resistance of H the
four values π̃_0(u), π̃_1(u), π̃_0(v), π̃_1(v) are pairwise distinct: a same-point pair differs in
the seed and a cross-point pair differs in x, and x is inside the `XorHash` input, so any
equality among them is a plain collision. Four pairwise distinct inputs with
f(x_0) ⊕ f(x_1) = f(x_2) ⊕ f(x_3) ≠ 0, (x_0,x_1) ≠ (x_2,x_3), (x_0,x_1) ≠ (x_3,x_2) is an
XOR-collision of H in the sense of [dCP22], Definition 11. Contradiction. So at most one point
α has s_0(α) ≠ s_1(α).

**Step 4 (outputs are a point-function sharing).** Every x ≠ α falls in the first case, where
the shares sum to zero (this is [dCP22]'s computation, verbatim: same seed, same control bit,
shared ocw, party 1 negates). If α exists, set β := y_0(α) + y_1(α); if no point differs, all
outputs are zero and any α with β = 0 works. ∎

What is inherited from the paper and what is new:

- Inherited verbatim: the Lemma 3 collision case analysis (restated for this implementation's
  (s, t) bookkeeping and with Assumption 2 in place of the paper's LSB convention), the
  sum-to-zero computation at agreeing points, and the privacy half of Theorem 3 (Lemma 4). The
  fold does not touch privacy: π remains a deterministic function of each party's own view, so
  the simulator of Lemma 4 computes the matching proof locally exactly as before.
- New here: Lemma 1, Lemma 2, the observation that x-tagging inside H makes every cross-point
  π̃-equality a plain collision (this is what lets the accumulator-independent part of the
  argument go through unchanged), and the Assumption 2 handling.

**Remark 1 (evaluation subsets).** The theorem is stated for the full domain, which is what
`EvalAllMerkle` computes. The identical argument works for any common evaluation set X folded
in domain order (the point-evaluation API): "at most one differing seed" then holds within X,
and the outputs form a point-function sharing on X. This matches the paper's Lemma 2, which is
likewise relative to the evaluated points.

**Remark 2 (honesty about the chain).** For the reference chain, the step "matching accumulated
proofs ⇒ per-point C-agreement at every point" is asserted by [dCP22]'s Lemma 2 proof in one
sentence ("From the collision resistance of H', if the proofs match then π^(x)_0 = π^(x)_1 for
all x"), and the paper's accumulation semantics is recovered from a one-line formula; a full
induction like Lemma 1's is not written there. We rely on that asserted sentence for the chain
and do not improve it in this document. The Merkle fold's corresponding step — Steps 1-2 above —
is the reason the fold was chosen: its reduction to pairwise collision resistance is complete
and short enough to write out, so the library's alternative algorithm does not inherit an
asserted aggregation argument.

## 6. Required hash properties

- **H = `XorHash`** (32B in, 64B out, used for π̃): must be collision-resistant and
  XOR-collision-resistant, exactly as the paper requires (Definition 11). The 4λ = 512-bit
  output width is the paper's own parameter choice for making XOR-collisions hard.
- **`HashMerkle`** (64B in, 32B out, used for leaves and internal nodes): must be
  collision-resistant. This is a strictly weaker flavor of assumption than XOR-collision
  resistance — the reduction (Lemma 1, Lemma 2) only ever extracts a pairwise collision.

Both are instantiated as single fixed-key Blake3 compressions with a caller-supplied IV.
Modeling a fixed-key Blake3 compression as collision-resistant (or as a random oracle) is a
standard but unproven heuristic; no reduction to Blake3's underlying primitives is claimed.

Concrete level. The binding strength of the fold is the collision resistance of a 256-bit
output: a single grinding attempt to make two chosen distinct 64B preimages collide succeeds
with probability ~2^{-256}, and the usual birthday accounting gives advantage ~q²/2^256 for q
compression-function evaluations, i.e. 128-bit collision security — the standard ceiling for any
256-bit hash. The chain's accumulator updates a state of 2λ = 254 bits under the library's
λ = 127 accounting (concretely the full 32B, since the whole first half is XORed with the 32B
`Hash` output), produced by the same Blake3 compression with the same 256-bit output width. The
fold's root is likewise the full 32B output. Both accumulators therefore reduce to collisions
of the same primitive at the same width: the concrete security level is not reduced by the
switch.

Two remarks on domain separation:

- No leaf/internal separation is needed. Classic Merkle-tree guidance adds level tags to block
  second-preimage attacks that mix leaves with internal nodes; those attacks matter when the
  tree serves as an authenticated dictionary against a prover who must extend an *honest* tree.
  Here the tree is only an equality compressor: both leaf sequences are derived from the
  adversary's own keys, and any level-mixing collision is simply one more collision excluded by
  plain collision resistance. The reduction never distinguishes leaf from internal queries.
- `Hash` and `HashMerkle` share one interface and one Blake3 implementation. A deployment
  instantiating both accumulators in one binary should hand them different IVs. This is hygiene
  rather than a requirement of the analysis — the fold's reduction never mixes the two hash
  roles — but it costs nothing.

## 7. Discussion

**(a) A deliberate deviation.** The Merkle fold is not Figure 1's `Prove`. The library carries
both: the chain remains the paper-faithful reference implementation, and the fold is an
alternative algorithm for the same task, selected for parallelism. Both parties must switch
together; there is no cross-version compatibility, and nothing in this document should be read
as a security claim about mixing them.

**(b) Why a Merkle fold and not an XOR fold.** An XOR-fold accumulator (π = ⊕_x Hash(C(x)) and
variants) would reduce "matching accumulators with differing per-point values" to an aggregation
coincidence: a subset-XOR condition over many hash outputs. That is exactly the kind of
asserted-in-one-sentence step this modification set out to avoid, and its concrete hardness is
subtler than pairwise collision resistance — generalized-birthday (Wagner-style) techniques
trade time and memory against the number of combined terms, which is why the paper itself is
conservative about XOR-collision resistance and sizes H's output at 4λ. The Merkle fold's
reduction needs only pairwise collision resistance: root equality with a differing leaf yields
a literal collision by the sibling-path induction of Lemma 1, with no condition on any number
of terms.

**(c) Order independence.** The soundness reduction never uses the order in which the leaves are
folded — Lemma 1 only uses that both parties fold the *same* sequence into the *same* tree
positions. Each leaf's preimage carries x inside the `XorHash` input, so a leaf cannot be
reinterpreted as another point's leaf without an H collision; the accumulation order is not
load-bearing for security. The implementation nevertheless fixes domain order on both sides:
this keeps the two trees aligned by construction and the proofs byte-reproducible, which is what
`Verify`'s equality test and the completeness argument use.

**(d) VDMPF chains (future work).** The paper's VDMPF (Figure 2) accumulates per-bucket proofs
into per-bucket values and then chains across buckets. The same substitution applies twice —
fold within each bucket, fold over the bucket values — and the same two-step reduction
(binding lemma, then the paper's per-point machinery) is expected to transfer, since the buckets
are likewise x-tagged through the hash inputs. This is not implemented and not analyzed here;
`vdmpf.cuh` keeps the chain.

**(e) Incremental and point-wise accumulation.** The fold composes with the point-evaluation
API: leaves are independent hashes of per-point values, so a server evaluating a subset folds
exactly the queried leaves, and a late point costs one leaf hash plus an O(log n) path
recomputation (or a rebuild). The chain instead must absorb points strictly in order and only
by appending at the end. The same property supports streaming the full-domain evaluation in
chunks.

## 8. The VDMPF Merkle-fold accumulation

The VDMPF extension described as future work in Section 7(d) now ships in
`include/fss/vdmpf.cuh` (`BatchEvalMerkle`, with private helper `FoldNodes`) and
`include/fss/point_eval_gpu.cuh` (`VdmpfBatchEvalGpu`). It replaces the proof accumulation
within and across Cuckoo buckets. `Gen`, `Verify`, and the output shares are unchanged.
The sequential-chain `BatchEval` remains the paper-faithful reference.

### Construction

Let i = 0,...,m-1 index buckets and p ∈ {0,1} index parties. Here m is the class's
compile-time bucket count: the final fold includes all m buckets, including those beyond
`key.m_rt`. Both parties evaluate the same ordered input batch with common Cuckoo parameters
and PRP seed. For each input position ω, in batch order, they enumerate hash indices
k = 0,...,κ-1 and append the resulting record to its bucket. This fixes each bucket's
canonical sequence by (input position, hash index), not by its within-bucket point index.

Write c_i for the number of records in bucket i and C_{p,i}(r) for record r's corrected
64B per-point hash. These are the inner VDPF's C-values from Section 4: `XorHash` outputs
with the bucket's cs correction applied according to the control bit. `HashMerkle` denotes
the same Blake3 64B-to-32B hash as in Section 3, called through `hash.Hash` in this code.

```
K = kMerkleChunkSize = 256
q_i = ceil(c_i / K)

Fold(N[0..s-1]):                              // s >= 1, each node is 32B
    while s > 1:
        next[j] = HashMerkle(N[2*j] || N[2*j+1])
                                               // j = 0 .. floor(s/2)-1
        if s is odd:
            next[floor(s/2)] = N[s-1]         // carry unchanged
        N = next
        s = ceil(s/2)
    return N[0]

leaf:       L_{p,i}(r) = HashMerkle(C_{p,i}(r))
                                               // 64B -> 32B, r = 0..c_i-1

chunk:      R_{p,i}(j) = Fold(L_{p,i}[j*K : min((j+1)*K, c_i)])
                                               // half-open slice, j = 0..q_i-1

anchor:     A_{p,i}(0) = HashMerkle(cs_i)       // 64B -> 32B
bucket:     A_{p,i}(j+1) = HashMerkle(A_{p,i}(j) || R_{p,i}(j))
                                               // left to right, j = 0..q_i-1
            B_{p,i} = A_{p,i}(q_i)

root:       root_p = Fold(B_{p,0}, ..., B_{p,m-1})
proof:      pi_p = { root_p[0], root_p[1], 0, 0 }
                                               // 64B, two trailing zero int4s
```

An empty bucket has no chunks and contributes `HashMerkle(cs_i)`. A one-node fold returns
that node unchanged. At larger sizes, `Fold` uses Section 3's adjacent-pair level rule,
carrying an odd trailing node to the next level without hashing or duplicating it.
The cs anchor plays the role of the chain's cs-initialized running value: each bucket
starts from its correction seed, including when it has no evaluated records.

Bucket leaf counts are arbitrary, unlike the power-of-two full domain of the VDPF fold.
Fixed chunks let the GPU fold leaves and chunks in parallel, leaving only q_i = ceil(c_i/K)
dependent combining calls per bucket after its anchor hash. K is part of the proof
definition, like a hash function's block size. Both parties must use the same chunk
boundaries.

The GPU reproduces this construction bit-exactly. `VdmpfLocateKernel` writes records in
(input position, hash index) order, and the stable radix sort groups them by bucket while
preserving that order. `VdmpfBatchEvalKernel` computes the output shares and corrected
C-values. `VdmpfFoldSetupKernel` computes leaf and chunk offsets,
`VdmpfChunkFoldKernel` folds the same K-leaf chunks as the host, and
`VdmpfBucketFoldKernel` performs the anchored bucket chains and the final fold over all
m bucket nodes.

### Correctness

For honestly generated keys, Section 4's argument applies to each inner VDPF: both parties
share every C-value and every cs_i. Common Cuckoo parameters and the same input batch give
the same canonical record sequences and chunk boundaries. Thus their leaves, chunk roots,
anchored bucket nodes, and final roots agree. Their 64B proofs match and the shared
`Verify` accepts.

### Soundness

We inherit Section 5's common-key-material assumption and its nonzero-cs caveat for each
inner VDPF, together with the same hash assumptions. The common batch, Cuckoo mapping,
and K fix identical positions and fold shapes for the two parties. The accumulator binding
argument extends Lemma 1 through each tier.

**Lemma 3 (VDMPF fold binding).** For these common positions and shapes, equal final roots
with any differing C-value, chunk root, or bucket node yield a literal `HashMerkle`
collision: distinct 64B inputs with the same 32B output.

*Proof.* Apply Lemma 1's sibling-path argument to the top fold. If a bucket node differs
but the final roots agree, follow its path to the last differing node before an equal
parent. An odd carry preserves the node value, so it cannot turn unequal nodes into equal
ones. The transition must occur at a combining hash. Its inputs differ in the same
32B child slot and its outputs agree, giving a collision. Excluding such collisions,
B_{0,i} = B_{1,i} for every bucket i.

For a bucket with q_i > 0, its final node is

```
B_{p,i} = HashMerkle(A_{p,i}(q_i-1) || R_{p,i}(q_i-1)).
```

If the outputs agree but either child differs, these two calls exhibit a collision.
Otherwise both children agree. Repeat backwards through the bucket's left chain.
Without a collision, equality of the final bucket nodes forces equality of every
corresponding chunk root and intermediate accumulator. This is the same sibling-path
induction on a composition of pairwise combining steps. Each step concatenates two
fixed-width children, so a difference in either child gives distinct 64B inputs.

Now apply Lemma 1 within each chunk. Equal chunk roots with a differing leaf yield a
collision at the last differing node before an equal parent. Again, an odd carry cannot
erase a difference. For a one-leaf chunk, the leaf is already its root. Finally, equal
leaves with differing C-preimages give the collision of Lemma 2 directly:

```
C_{0,i}(r) != C_{1,i}(r)
HashMerkle(C_{0,i}(r)) = HashMerkle(C_{1,i}(r)).
```

Thus any disagreement at the stated tiers beneath equal final roots produces a literal
collision. ∎

`Verify` accepting implies root equality because the trailing 32B are fixed zeros.
Lemma 3 then gives agreement of all evaluated inner C-values unless a `HashMerkle`
collision occurs. Section 5's remaining per-point argument applies within each bucket:
under the inherited assumptions, agreement implies at most one differing inner seed
among that bucket's evaluated indices, except for a collision or XOR-collision of H.
The output-share computation and the Cuckoo construction remain unchanged. This supplies
the replacement accumulation step for the VDMPF share-integrity argument.

As in Section 5, a union bound accounts for failure across the tiers. Let
ε_leaf, ε_chunk, ε_bucket, and ε_top denote the collision-finding advantages of the
reductions extracting a `HashMerkle` collision at leaves, within chunks, in bucket chains,
and in the top fold, respectively. Let ε_H include the inherited H collision and
XOR-collision advantages, summed over the inner buckets. Then the failure probability
under the inherited assumptions is bounded by

```
Pr[failure] <= epsilon_leaf + epsilon_chunk + epsilon_bucket + epsilon_top
               + epsilon_H.
```

These are collision advantages against the same hash, not assumptions of independent
hashes at different tiers. The reduction requires neither independence between tiers
nor an assumption about XOR aggregation. Common cs_i makes the anchors identical without
a further failure term.

**A deliberate deviation.** As in Section 7(a), this is an alternative accumulation
algorithm chosen for parallelism. The sequential-chain `BatchEval` remains the reference.
Both parties must switch together and use the same K and canonical order. Host and GPU
implementations compute the same fold proof, but chain and fold proofs have no
cross-accumulator compatibility.

## 9. References

- Leo de Castro, Antigoni Polychroniadou. *Lightweight, Maliciously Secure Verifiable Function
  Secret Sharing.* EUROCRYPT 2022, LNCS 13276, pp. 150-179.
  <https://eprint.iacr.org/2021/580>, <https://doi.org/10.1007/978-3-031-06944-4_6>.
- Ralph C. Merkle. *Secrecy, Authentication, and Public Key Systems.* Technical Report
  ESL-R-994, Stanford University, 1979. (Merkle trees.)
- David Wagner. *A Generalized Birthday Problem.* CRYPTO 2002. (XOR-fold aggregation costs.)
- BLAKE3: <https://github.com/BLAKE3-team/BLAKE3>. (Hash instantiation.)

Citation correction note: the headers `include/fss/vdpf.cuh`, `include/fss/cuckoo_hash.cuh`,
and `include/fss/vdmpf.cuh` currently link the scheme to eprint 2024/677. That number is an
unrelated paper — *Asynchronous Consensus without Trusted Setup or Public-Key Cryptography*,
Das, Duan, Liu, Momose, Ren, Shoup, ACM CCS 2024. The correct citation for the VDPF/VDMPF
scheme is eprint 2021/580 above (the headers' own reference lists already carry the correct
DOI; only the eprint links are wrong). The headers are being corrected in the same change that
adds the Merkle-fold accumulator.
