# VDPF and VDMPF Merkle-fold proof accumulation: security analysis

This document analyzes the Merkle accumulation alternative for this library's implementation of the 2-party VDPF of de Castro and Polychroniadou ([dCP22], EUROCRYPT 2022, eprint 2021/580).
For the implemented `Gen`, `Eval`, and `Verify`, it replaces the sequential `Prove` accumulation with a parallel binary Merkle fold over a fixed ordered sequence.
Section 2 states the library's seed/control-bit bookkeeping and concrete H input encoding, which must be distinguished from the paper's notation.
The alternative ships as `ProveMerkle`, `EvalAllMerkle`, and `EvalAllMerkleGpu`, with the VDMPF extension shipping as `BatchEvalMerkle` and `BatchEvalMerkleGpu` (Section 8).
The implemented reference chains `Prove` and VDMPF `BatchEval` remain available.
Both parties must select the same accumulator and input order.
Chain and Merkle proofs differ, and `Verify` compares proofs within one accumulator only.

The document is self-contained for a reader with [dCP22] at hand. It states what is proven here,
what is inherited from the paper verbatim, and what is assumed of the hashes. Where the paper
asserts a step in one sentence, we say so; where our own reduction is complete, we write it out.

## 1. Overview

The timing measurements below describe the implementation before the public API refactor.
They are historical motivation, not measurements of the current helpers or a performance guarantee.

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
4-6 give the conditional Merkle argument under the common-key, nonzero-cs and hash assumptions stated below.
Its binding reduction is written out explicitly.

## 2. The original construction

Notation follows the library. Parties are b ∈ {0,1}; the input domain is
{0,...,n-1} with n = 2^in_bits; the output domain is a group of 16B elements whose last word's
least significant bit is clamped to zero, i.e. λ = 127 nominal. Concrete widths: seeds and group
elements are 16B (`int4`); the paper's H is `XorHash`, 32B in (x packed into 16B, seed 16B) to
64B out, matching the paper's H: (x, s) ↦ 4λ bits; the paper's H' is `Hash`, 64B in to 32B out,
matching H': 4λ ↦ 2λ bits.
With Blake3, H concatenates two 32B digests from two fixed-key compressions.
H' uses one fixed-key Blake3 compression (`include/fss/hash/blake3.cuh`).
`⊕` is XOR and `||` concatenation.

**Logical H input and encoding.** Write P(x) for the complete 128-bit coordinate block produced by `Pack(x)` and let S be the seed blocks whose fourth 32-bit word has bit 0 cleared.
The scheme calls `XorHash` with the tuple `(P(x), s)` for s in S.
VDMPF uses the within-bucket index as x in each inner VDPF.
`Pack` preserves all coordinate bits; in particular, coordinate bit 96 occupies bit 0 of P(x)'s fourth word and remains message data.
The second block has 127 effective message bits, with its fourth word's bit 0 reserved.
Thus the logical tuple domain is 128 + 127 bits, restricted to the scheme's declared coordinate range.

Let T_d(s) set that reserved seed bit to d in a copy of s, for d in {0,1}.
The built-in tuple backends concatenate the digest of `P(x) || T_0(s)` with the digest of `P(x) || T_1(s)`.
For Blake3, each message is 32B, with counter 0 and flags `CHUNK_START | CHUNK_END | ROOT | KEYED_HASH` under the configured IV.
For Sha256, each digest hashes the configured 16B key prefix followed by that 32B message, for 48B total; this is the existing prefixed-hash construction, not HMAC.
The supplied value of the second block's reserved bit is ignored by these backends, and the caller's input blocks are not modified.
Gen and every Eval path use a clamped leaf seed, so they supply zero in that position.

The notation H(x || s) below refers to this logical `(coordinate, seed)` input and its explicit encoding.
Different coordinates and different clamped seeds remain different logical messages; the reserved bit is not additional message data or the separately tracked control bit t.
Collision resistance and XOR-collision resistance on this domain are assumptions, not consequences of the encoding or of the `XorHashable` concept.
Custom backends must satisfy this domain contract and the required assumptions.

**Gen** (per level i = 0..in_bits-1, `Cw` = {s_cw with tl_cw in its LSB, tr_cw}): expand both
parties' seeds with the PRG, compute s_cw from the off-path child pair and the two control bits,
advance along the bit of a. After the loop, with s_b the two leaf seeds at a:

```
pi_tilde_b = XorHash(a || s_b)                // 64B
cs = pi_tilde_0 XOR pi_tilde_1                // 64B correction seed, part of both keys
if cs == 0: retry with fresh seeds             // library returns 1, caller resamples
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

`ProveMerkle` replaces only the accumulation.
`HashMerkle` uses the Blake3 four-block `Hash` overload (64B in, 32B out, `array<int4,2>`).
The tuple H encoding is specified separately in Section 2.

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
embarrassingly parallel batch — one kernel per level on GPU (`EvalAllMerkleGpu`), one OpenMP
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

**Assumption 1 (common key material).** The two keys carry identical cws, cs, and ocw; only the initial seeds may differ.
The malicious-key model does not require the initial seeds to be unequal or the keys to have been produced by honest Gen.
This is a protocol-level assumption about key distribution
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
differ by cs ≠ 0.
The implemented `Gen` returns failure when cs = 0, so every successful generation satisfies this condition.
This generation check does not enforce the condition on maliciously supplied keys.
Servers must reject zero cs or otherwise enforce Assumption 2 before relying on the soundness statement.
The zero-cs exposure follows from the implemented `Eval` bookkeeping and applies to both accumulators.
The paper instead ties the leaf control bit to the seed's LSB.

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
ruling that out, C_0(x) = C_1(x) for all x.

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

**Remark 1 (evaluation sequences).** The theorem is stated for the full domain, which is what `EvalAllMerkle` computes.
For `BatchEvalMerkle` and `BatchEvalMerkleGpu`, let `xs = (x_0, ..., x_{M-1})` be a common nonempty sequence of inputs in D = {0,...,2^in_bits-1}, and let X = {x_i : 0 <= i < M} be its set of distinct values.
Both parties must use the same sequence, including its order and length, the Merkle accumulator, and the same configured hash functions.
Assumptions 1 and 2 and all hash assumptions of Theorem 2 remain required.

The leaf at position i is L_b(i) = HashMerkle(C_b(x_i)).
At each level, hash adjacent pairs in order and carry an odd trailing node to the next level unchanged.
A one-leaf tree has that leaf as its root.
Thus the common length fixes the same positional tree shape even when M is not a power of two.
If `Verify` accepts, the roots agree.
If corresponding leaves differed, follow a differing leaf toward the common root until a differing node has an equal parent.
An unchanged carry cannot remove a difference, so this transition must occur at a hash of two children and exposes a `HashMerkle` collision, as in Lemma 1.
Equal leaves with different C-preimages expose Lemma 2's collision.
Therefore acceptance implies C_0(x_i) = C_1(x_i) at every sequence position, except with the collision-finding probability covered by the stated assumptions.

Apply Steps 3 and 4 to the distinct values in X.
Except with the stated hash failure probability, there exist alpha in D and beta in the output group such that y_0(x_i) + y_1(x_i) equals beta when x_i = alpha and zero otherwise, for every i.
Each occurrence of a repeated input contributes its own leaf and repeats the same output share for each party.
Consequently, several sequence positions may reconstruct to a nonzero value, but at most one distinct input value in X may do so.
This statement does not constrain outputs outside X or establish soundness for parties using different sequences.
Both servers must obtain the same `xs` by a path the client cannot split.
This is a deployment assumption, not a sequence-agreement guarantee of `Verify`.

## 6. Required hash properties

- **H = `XorHash`** (32B in, 64B out, used for π̃): must be collision-resistant and
  XOR-collision-resistant on the logical input domain defined in Section 2, as required by the paper's Definition 11.
  The byte-aligned implementation uses 512 bits for this role and 256 bits for H', while λ = 127 describes the clamped seed width.
- **`HashMerkle`** (64B in, 32B out, used for leaves and internal nodes): must be
  collision-resistant. This is a strictly weaker flavor of assumption than XOR-collision
  resistance — the reduction (Lemma 1, Lemma 2) only ever extracts a pairwise collision.

The concrete H uses two fixed-key Blake3 compressions and concatenates their first 32B outputs.
`HashMerkle` uses one fixed-key Blake3 compression and returns its first 32B output.
The IV is caller-supplied.
The required collision and XOR-collision properties of these instantiated functions remain assumptions.
No reduction to Blake3's underlying primitives is claimed.

Concrete level. In an ideal 256-bit-output hash model, two fixed distinct messages collide with probability 2^{-256}, and the birthday collision probability after q distinct queries is of order q²/2^256 while that ratio is small.
This gives the usual approximately 128-bit collision-work target, not a proof that the configured Blake3 function attains it.
The chain's accumulator updates a state of 2λ = 254 bits under the library's
λ = 127 accounting (concretely the full 32B, since the whole first half is XORed with the 32B
`Hash` output), produced by the same Blake3 compression with the same 256-bit output width. The
fold's root is likewise the full 32B output.
The fold's conditional binding reduction uses this 256-bit `HashMerkle` output.
A concrete security claim also requires the instantiated hashes to satisfy the stated assumptions.

Two remarks on domain separation:

- No leaf/internal separation is required for the binding of two evaluations on the common fixed tree shape considered here.
  The reduction compares corresponding positions in that shape and extracts a literal collision if differing values occur beneath equal roots.
  It does not establish binding across different tree shapes or sequence lengths.
  Agreement on the evaluation sequence remains a separate assumption.
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

**(c) Common sequence and tree shape.** Lemma 1 compares corresponding positions in the same tree shape.
Its argument does not depend on the execution order of independent hash computations.
Full-domain evaluation uses domain order.
Batch evaluation uses the caller's ordered sequence, including repeated inputs, with the adjacent-pair and odd-carry rule in Remark 1.
Both parties must agree on that sequence and its length.
This agreement aligns corresponding leaves and makes honestly generated proofs reproducible.
The argument does not establish agreement on the sequence or binding across different sequences or tree shapes.

**(d) VDMPF accumulation.** The paper's VDMPF (Figure 2) uses chains within buckets and across buckets.
Section 8 defines the implemented Merkle alternative, with a canonical tree within each bucket and another over the bucket nodes.
The configured bucket correction seeds bind the per-bucket roots.
The chain-based `BatchEval` remains available as a separate accumulation method.

**(e) Incremental and point-wise accumulation.** The fold composes with the point-evaluation
API: leaves are independent hashes of per-point values, so a server evaluating a subset folds
exactly the queried leaves, and a late point costs one leaf hash plus an O(log n) path
recomputation (or a rebuild). The chain instead must absorb points strictly in order and only
by appending at the end. The same property supports streaming the full-domain evaluation in
chunks.

## 8. The VDMPF Merkle-fold accumulation

The VDMPF alternative is implemented by `BatchEvalMerkle` in `include/fss/vdmpf.cuh` and an internal GPU launcher in the same header.
The launcher is in `fss::gpu::detail` and is not a supported public API.
It replaces proof accumulation within and across Cuckoo buckets.
`Gen`, `Verify`, and the output shares are unchanged.
The sequential-chain `BatchEval` remains the paper-faithful reference.
This section describes this library's alternative accumulation, not an additional construction claimed by the paper.

### Construction

Let i = 0,...,m-1 index buckets and p ∈ {0,1} index parties.
Here m is the class's compile-time `kBucketCount`, including buckets beyond the runtime `m_rt`.
Both parties evaluate the same ordered input batch with common Cuckoo parameters and PRP seed.
For each input position ω, in batch order, enumerate the three hash indices k = 0,1,2 and append the corresponding record to its bucket.
Each bucket's canonical sequence is therefore ordered by (input position, hash index), not by its within-bucket point index.

Write c_i for the number of records in bucket i and C_{p,i}(r) for record r's corrected 64B per-point hash.
These are the inner VDPF's C-values from Section 4: `XorHash` outputs with the bucket's cs correction applied according to the control bit.
`HashMerkle` denotes the configured 64B-to-32B collision-resistant hash called through `hash.Hash`.
BLAKE3 is the concrete instantiation analyzed in Section 3.

```
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
anchor:     A_{p,i} = HashMerkle(cs_i)         // 64B -> 32B
bucket:     B_{p,i} = A_{p,i}                 // if c_i = 0
            R_{p,i} = Fold(L_{p,i}[0..c_i-1]) // otherwise
            B_{p,i} = HashMerkle(A_{p,i} || R_{p,i})

root:       root_p = Fold(B_{p,0}, ..., B_{p,m-1})
proof:      pi_p = { root_p[0], root_p[1], 0, 0 }
                                               // 64B, two trailing zero int4s
```

An empty bucket contributes only its hashed correction seed.
A one-node fold returns that node unchanged.
At larger sizes, `Fold` uses Section 3's adjacent-pair rule and carries an odd trailing node without hashing or duplicating it.
The anchor binds a nonempty bucket's root to its correction seed in one combining hash.

Bucket leaf counts can be arbitrary, unlike the power-of-two full domain of the VDPF fold.
The tree definition depends only on the canonical record order and count.
GPU block size, work partitioning, and intermediate execution chunks are not inputs to this construction.
They may change which threads compute subtrees, but must preserve the same adjacent-pair tree and odd-carry rule.

The GPU Locate pass writes records in (input position, hash index) order.
A stable radix sort groups them by bucket while preserving that order.
The evaluation pass computes output shares and corrected C-values.
The folding passes compute each bucket's canonical root, combine that root with its anchor once, and fold all m bucket nodes.
A GPU implementation can first fold aligned power-of-two chunks, because each full chunk is a subtree of the canonical tree.
A trailing partial chunk follows the same odd-carry rule, and its root carries unchanged through any remaining levels within that subtree.
The chunk roots must then be folded into the bucket root with the canonical adjacent-pair rule, rather than accumulated as a chain.
Consequently, changing a valid execution chunk size does not change the proof.

### Correctness

For honestly generated keys, Section 4's argument applies to each inner VDPF: both parties share every C-value and every cs_i.
Common Cuckoo parameters and the same ordered input batch give the same per-bucket record sequences and tree shapes.
Their leaves, bucket roots, anchored bucket nodes, and final roots therefore agree.
Their 64B proofs match and `Verify` accepts.
Host and GPU evaluation must implement this same construction, including empty buckets and odd carries.

### Soundness

We inherit Section 5's common-key-material assumption and its nonzero-cs caveat for each inner VDPF, together with the same hash assumptions.
The common input batch and Cuckoo mapping fix identical positions, counts, and tree shapes for the two parties.
The binding argument concerns these fixed shapes.
It does not claim binding across different record sequences or tree shapes.

**Lemma 3 (VDMPF fold binding).** For these common positions and shapes, equal final roots with any differing C-value, bucket root, or bucket node yield a literal `HashMerkle` collision: distinct 64B inputs with the same 32B output.

*Proof.* Apply Lemma 1's sibling-path argument to the top fold.
If a bucket node differs but the final roots agree, follow its path to the last differing node before an equal parent.
An odd carry preserves the node value, so the transition must occur at a combining hash.
Its inputs differ in the same 32B child slot and its outputs agree, giving a collision.
Excluding such collisions, B_{0,i} = B_{1,i} for every bucket i.

For a nonempty bucket, its node is `HashMerkle(A_{p,i} || R_{p,i})`.
If these outputs agree while either corresponding child differs, the two calls exhibit a collision.
Otherwise the corresponding roots agree.
Apply Lemma 1 within that bucket's canonical tree.
Equal roots with a differing leaf give a collision at the last differing node before an equal parent.
An odd carry cannot erase a difference, and for a one-leaf bucket the leaf is already its root.
Finally, equal leaves with differing C-preimages give Lemma 2's direct collision:

```
C_{0,i}(r) != C_{1,i}(r)
HashMerkle(C_{0,i}(r)) = HashMerkle(C_{1,i}(r)).
```

An empty bucket has no C-values to bind and contributes its common anchor.
Thus disagreement at any stated tier beneath equal final roots produces a literal collision. ∎

`Verify` accepting implies root equality because the trailing 32B are fixed zeros.
Lemma 3 gives agreement of all evaluated inner C-values unless a `HashMerkle` collision occurs.
Section 5's remaining per-point argument applies within each bucket under its inherited assumptions and scope.
The output-share computation and Cuckoo construction are unchanged.
This supplies the replacement accumulation step in the VDMPF share-integrity argument without introducing a chunk-size assumption.

Repeated outer inputs are allowed in the common batch.
Every occurrence contributes its own candidate records at its own input position and one leaf per record.
For bucket i, apply the per-point argument to the set of distinct within-bucket indices in its records.
Its conclusion is at most one distinct evaluated within-bucket index with a nonzero reconstructed inner output, even when that index occurs in several records.
This is a per-bucket statement, not a one-point bound for the VDMPF's outer input batch.
The common Cuckoo mapping and the remaining VDMPF assumptions are still required.
Both servers must obtain the same ordered batch, including its length, by a path the client cannot split.
The fold does not establish this deployment assumption.

A union bound accounts for failures across the tiers.
Let ε_leaf, ε_tree, ε_bucket, and ε_top denote the collision-finding advantages for the leaf hashes, bucket trees, anchored bucket hashes, and top fold.
Let ε_H include the inherited H collision and XOR-collision advantages, summed over the inner buckets.
Then, under the inherited assumptions,

```
Pr[failure] <= epsilon_leaf + epsilon_tree + epsilon_bucket + epsilon_top
               + epsilon_H.
```

These are collision advantages against the same hash, not assumptions of independent hashes at different tiers.
Common cs_i makes the anchors identical without an additional failure term.

**A deliberate deviation.** As in Section 7(a), this is an alternative accumulation algorithm chosen for parallelism.
Both parties must use the same accumulator, canonical input order, and common key configuration.
Chain and Merkle proofs have no cross-accumulator compatibility.
The Merkle proof definition is independent of GPU execution parameters.

## 9. References

- Leo de Castro, Antigoni Polychroniadou. *Lightweight, Maliciously Secure Verifiable Function
  Secret Sharing.* EUROCRYPT 2022, LNCS 13276, pp. 150-179.
  <https://eprint.iacr.org/2021/580>, <https://doi.org/10.1007/978-3-031-06944-4_6>.
- Ralph C. Merkle. *Secrecy, Authentication, and Public Key Systems.* Technical Report
  ESL-R-994, Stanford University, 1979. (Merkle trees.)
- David Wagner. *A Generalized Birthday Problem.* CRYPTO 2002. (XOR-fold aggregation costs.)
- BLAKE3: <https://github.com/BLAKE3-team/BLAKE3>. (Hash instantiation.)
