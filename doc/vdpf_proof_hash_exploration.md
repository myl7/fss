# The VDPF and VDMPF proof-hash replacement attempt: a complete record

## Overview

This document records the complete attempt to replace the Blake3 `Hash` role of the VDPF and VDMPF proof machinery — the 64B-to-32B compression called H' in [dCP22] — with an AES- or ChaCha-based construction.
The attempt ran from an initial proposal, through three rounds of external security review, to measured primitive costs on both platforms, and closed without shipping: no library code from the attempt survives and Blake3 remains H'.
The motivation was measured, not aesthetic: on the reference CPU host the chain `EvalAll` spends its roughly 246 ns per-point excess over a same-shape DPF walk on three Blake3 calls, and on the GPU the Merkle path measured 0.486 ms against the DPF walk's 0.156 ms at n = 2^20 (Section 10).
The security analysis ended well — two constructions with reviewed bounds — and the measurements ended the attempt: in the two-step Hirose shape Blake3 runs at 0.057 ns per hash on the A6000 (17.6 GH/s) and 118.8-131.5 ns on the pinned CPU, and both approved candidates measure behind it on both platforms (Section 10.4).
Everything from the decision preamble through the References is preserved verbatim from the analysis as it stood after each review; this reframing only added the Overview, the Timeline, and the Part groupings.

## Timeline

| When | Phase | Outcome |
| --- | --- | --- |
| 2026-10-16 | The direction opens | A research verdict recommends an AES-MMO replacement for H' on measured cost grounds; exploratory only. |
| 2026-10-17 | First proposal, implementation starts | The split-lane AES-MMO construction is specified and its implementation begins in a branch worktree. |
| 2026-10-17 | External review, round 1 | The construction is broken at exactly 64-bit collision security (Part II); the implementation is stopped mid-flight and fully reverted, and the analysis becomes the rejection record. |
| 2026-10-17 | External review, round 2 | Feasibility restated: requirements, construction-family survey, and the two-step Hirose AES-256 selection, approved for implementation and measurement only (Part III). |
| 2026-10-17 | External review, round 3 | The truncated RFC 8439 ChaCha20 instantiation joins as a conditional co-equal candidate with a direct random-function bound (Section 7.4). |
| 2026-10-17 | Measurements on cs11369a | KAT-verified primitive costs in the exact shape: Blake3 0.057 ns/hash GPU and 118.8-131.5 ns CPU; ChaCha20 0.32 and 594.4-719.4; AES-256 T-table 1.7-12.0 GPU and 188.9-198.6 AES-NI CPU (Section 10.4). |
| 2026-10-17 | Closure | The performance gate fails on both platforms; the direction closes with Blake3 as H'. |

Measurement artifacts live outside the repository: the microbenchmark program, manifest, and raw logs are preserved at `cs11369a:~/tmp/fss-hirose-ab/scratch/`.
No code from the attempt was merged; the reverted implementation left the working tree clean, verified at revert time.

---


**Decision: reject the split-lane AES-MMO construction and approve the exact two-step AES-256 Hirose construction and the ChaCha20 variant in Section 7 for implementation and benchmarking under their respective explicit assumptions.**
The third review classifies ChaCha20 as **(a) a conditional candidate alongside AES-256**, without a GPU performance preference.
Its random-function proof retains the approximately 2^128 constant-success collision work scale, with a separate bound rather than an application of the ideal-cipher theorem.
Direction 3 is cryptographically feasible.
A CPU speedup is unestablished and looks unlikely under the supplied key-schedule estimate.
A GPU-focused deployment is valid if its measured performance justifies adoption and its host implementation computes the same function.
The existing Blake3 implementation remains the production baseline pending implementation review and measurements.
This document changes no code.

AES-based compression functions can serve as cryptographic hashes when their construction and security bound match the required use.
One-wayness alone does not imply collision resistance.
Here H' must resist finding two distinct arbitrary 64B inputs with the same 32B output at approximately 2^128 classical work for constant success probability.
It does not require the XOR-collision resistance separately assumed for H, which remains Blake3 `XorHash`.
Astra independently reviewed the requirements, construction, reduction, and decision in this second review.
Astra independently checked the third review's random-function proof and concrete ChaCha caveats.

## Part I — The requirement

## 1. Required property and code scope
The replacement must provide pairwise collision resistance for arbitrary distinct 64B inputs, with a target of about 2^128 classical work for constant success probability.
The sibling document's Section 6 obtains this target by assuming birthday resistance for the current 256-bit output.
Output width alone supplies no lower bound on collision-finding cost.

The sibling document's Lemmas 1 and 2 extract literal `HashMerkle` collisions from VDPF fold disagreements.
Its Section 8, Lemma 3, does the same for the VDMPF fold, including chunk folds, anchored bucket chains, and the top fold.
These reductions require no XOR-collision property of H'.
The reference chain has a different status: Section 5, Remark 2, inherits the paper's asserted accumulation step without supplying a complete induction.
This review does not upgrade that assertion to a proved chain-binding theorem.

The current interfaces in `include/fss/hash/blake3.cuh` are:

| Role | Input | Output | Implementation |
| --- | --- | --- | --- |
| `Hash` | `span<const int4, 4>`, 64B | `array<int4, 2>`, 32B | One Blake3 compression |
| `XorHash` | `tuple<int4, const int4>`, 32B | `array<int4, 4>`, 64B | Two Blake3 compressions with the first block's LSB cleared and set |

`Hash` uses the caller-supplied 32B IV, counter 0, block length 64, and flags `CHUNK_START | CHUNK_END | ROOT | KEYED_HASH` (`0x1B`).
It returns the first 32B of the compression output.
The two `XorHash` calls instead use block length 32 and zero padding.
The sibling document's description of both roles as a single compression is inaccurate for `XorHash`.

The relevant H' calls are:

| Code | H' input and purpose |
| --- | --- |
| `Vdpf::Prove`, `Vdpf::EvalAll` | Running 64B proof XOR corrected per-point C-value, followed by XOR of the hash output into the first 32B of the proof |
| `Vdpf::ProveMerkle`, reached by `EvalAllMerkle` | A 64B C-value for each leaf, then concatenated 32B children |
| `Vdmpf::BatchEval` | Per-bucket chain updates and cross-bucket chain updates |
| `Vdmpf::BatchEvalMerkle` | A bucket's 64B `cs` anchor, each corrected C-value, and each accumulator/chunk-root concatenation |
| `Vdmpf::FoldNodes` | Concatenated 32B children within chunks and across bucket nodes |

VDPF fold proofs contain `{root[0], root[1], cs[2], cs[3]}`.
VDMPF fold proofs contain `{root[0], root[1], 0, 0}`.
Both use the same 64B equality check in `Verify`.
The sibling document's Section 8 describes the implemented VDMPF fold and supersedes its Section 7(d) future-work note.

## 2. What one-way MMO does and does not establish

For an n-bit block cipher, distinguish the cipher's block width from its key width and from the compression function's output width.
AES-128 and AES-256 both have n = 128-bit blocks.
AES-256 has a 256-bit key and 14 rounds, but its ciphertext still has 128 bits.

Write `F(k, x) = E_k(x) XOR x`.
With two independent n-bit inputs, F is a 2n-to-n compression function.
With a fixed key, it is an n-to-n function instead.
Calling either function "one-way" does not specify the attacker's knowledge, oracle access, or collision game.
Some texts use "one-way compression function" as a category that also requires collision resistance.
Even in that usage, the security level depends on n.

| Property | Attacker's task | Relevance here |
| --- | --- | --- |
| Preimage resistance | Given a target digest, find an input mapping to it | Insufficient for fold binding |
| Second-preimage resistance | Given a particular input, find another with the same digest | Insufficient when the malicious client can choose both candidates |
| Pairwise collision resistance | Choose any distinct inputs with equal digests | The sufficient H' assumption extracted by the sibling proofs |
| Secret-key PRF security | Distinguish an oracle with a hidden random key from a random function | Different key visibility and oracle experiment |
| Pseudorandomness on a restricted input distribution | Distinguish sampled outputs under stated conditioning | Does not by itself cover adversarial collision search |

An ideal n-bit-output function typically needs about 2^n work for preimage search and 2^(n/2) work for collision search.
Thus an AES-128-based 128-bit digest can have an appropriate one-way bound and still offer only 64-bit collision resistance.
Section 5 proves the 64-bit collision bound for the exact F used in the rejected proposal.
A 256-bit output made by concatenating two such digests need not restore 128-bit collision resistance.

The PGV family includes standard MMO, `h_i = E_{h_(i-1)}(m_i) XOR m_i`, and Davies-Meyer, `h_i = E_{m_i}(h_(i-1)) XOR h_(i-1)`.
The latter uses the message as the cipher key, so a "message-as-key MMO chain" needs this naming clarification.
BRS analyze the specified PGV constructions in an ideal-cipher model with all-key encryption and decryption access [BRS02].
Their positive results justify these constructions as hashes at the security level of their n-bit output.
They do not give a 128-bit collision level for an AES-based single-block-length digest.
A common suffix preserves an intermediate chaining collision.
Neither longer iteration nor digest duplication repairs that limit.

Public fixed-key MMO also appears in PRGs and correlation-robust hashing on restricted inputs.
Such uses require their own input-distribution or correlation-robustness assumption.
A hidden AES key can support a different PRF-based argument, with a birthday loss at its output width.
Neither situation supplies an arbitrary-input public collision theorem for this H'.
Introducing a new secret shared by the evaluators would change the protocol and key-management model.
The existing Blake3 IV is public despite the implementation's `KEYED_HASH` flag.

## Part II — First proposal and rejection

## 3. Earlier candidates: valid attacks and corrected claims

The earlier draft mixed single-lane attacks with claims about unspecified two-lane arrangements.
The following statements specify which arrangements they cover.
No shared-message two-lane construction receives approval from the absence of an attack in this section.

### 3.1 Fixed-key XOR-injection chains

For a public fixed key K and 128-bit state, consider four updates:

```
h_{i+1} = E_K(h_i XOR m_i) XOR h_i,  i = 0,...,3.
```

Write delta_i = h_i XOR h'_i and DeltaE_i for the two cipher outputs' difference at step i.
Then `delta_{i+1} = DeltaE_i XOR delta_i`.
From a common initial state, the final difference is the XOR of the four DeltaE values.
However, choosing `m_i XOR m'_i = delta_i` makes the cipher inputs equal and leaves `delta_{i+1} = delta_i`.
It does not merge unequal states.
The previous draft's diagonal-merging argument was false.

A direct constant-query attack on one lane uses the public inverse cipher.
After choosing any distinct three-block prefixes with final prefix states h_3 and h'_3, choose a target t and set:

```
m_3  = h_3  XOR E_K^{-1}(t XOR h_3)
m'_3 = h'_3 XOR E_K^{-1}(t XOR h'_3).
```

Both final states equal t, and the distinct prefixes ensure distinct four-block messages.
This attack needs no generalized birthday algorithm.

Even forward queries suffice for a single lane.
For any a, the two blocks `(h_0 XOR a, h_0 XOR E_K(a) XOR a)` return the state to h_0.
Choosing two different a values gives different prefixes with the same state, after which a common two-block suffix gives a four-block collision.

For reference, Wagner's balanced four-list algorithm on independent n-bit lists costs about 2^(n/3) time and storage, up to list-count and implementation factors [W02].
At n = 128 this is about 2^42.67, not 2^32.
The latter exponent belongs to the balanced eight-list problem over 128-bit values.
Four lists of size 2^32 can have a solution with constant probability, but this fact does not supply a 2^32-time algorithm to find it.
An attack must also show that its generated list entries correspond to jointly realizable message blocks.

If two lanes have separate message blocks, one can attack one lane and hold the other fixed.
If they share all four message blocks under different fixed keys, the steering equations must hold in both lanes for the same message.
Neither the one-lane attacks above nor the previous draft's eight-term assertion proves that simultaneous condition.
We withdraw the claimed universal 2^32 bound for that unspecified two-lane design.

### 3.2 CBC-style compression

For one lane with a fixed public key and initial state, let:

```
c_{i+1} = E_K(c_i XOR m_i),  i = 0,...,3,
output = c_4.
```

Choose distinct three-block prefixes and compute c_3 and c'_3.
For any m_3, set `m'_3 = m_3 XOR c_3 XOR c'_3`.
The final cipher inputs coincide, so the two distinct four-block messages collide.
This is a constant-query final-block compensation attack at a fixed message length.
It does not require decryption access or a variable-length extension.

XORing a common fixed IV into the final output preserves this collision.
Other feedforward choices change the collision equation and require separate analysis.
The previous assertion that no final-round feedforward could help was unsupported.
For two lanes sharing message blocks, the same final-block difference works only when the required state differences agree in both lanes.
The one-lane calculation alone does not establish a two-lane collision.

### 3.3 Independent block hashing with symmetric combination

Suppose the same block function g processes each block and S is invariant under permuting its arguments:

```
H'(m) = S(g(m_0), g(m_1), g(m_2), g(m_3)).
```

Swapping two unequal input blocks gives a distinct message with the same output.
XOR and multiset combination are examples, and the attack needs constant work.
The condition that the same g applies at the swapped positions matters.
Position-dependent functions or an ordered concatenation do not inherit this attack without further analysis.
Concatenating four 128-bit block hashes also gives 64B, rather than the required 32B output.
Rejecting symmetric combination does not prove that any particular ordered or keyed alternative is secure.

## 4. The proposed two-lane construction is broken at the target level

Let K_0 and K_1 be arbitrary fixed public 128-bit constants, and let E be full 10-round AES-128.
Split the 64B input into four 128-bit blocks and define:

```
F(k, x) = E_k(x) XOR x

z_0 = F(K_0 XOR m_0, m_1)
z_1 = F(K_1 XOR m_2, m_3)
H'_split(m_0 || m_1 || m_2 || m_3) = z_0 || z_1.
```

Each lane compresses 256 input bits to 128 output bits.
The complete function is a concatenation of two functions on disjoint input coordinates.
An H'_split collision requires equal lane outputs, but it does not require distinct inputs in both lanes.

### Attack 1: keep one lane unchanged

Fix arbitrary blocks a and b for the second lane.
For q distinct keys k_i, query `v_i = E_{k_i}(0)` and construct:

```
M_i = (K_0 XOR k_i) || 0 || a || b
H'_split(M_i) = v_i || F(K_1 XOR a, b).
```

In the ideal-cipher model the v_i are independent uniform 128-bit values, because each query uses a different key.
A repeated v_i gives a literal H'_split collision on distinct 64B messages.
For N = 2^128, its success probability is exactly:

```
1 - product_{i=0}^{q-1}(1 - i/N).
```

For q much smaller than sqrt(N), this is approximately q(q-1)/(2N).
For q of order sqrt(N), it has constant success probability.
The attack therefore costs about 2^64 AES evaluations and that many stored entries with a direct birthday table.
Every AES key is permitted by the construction, so the key/plaintext split imposes no obstacle to this attack.
It works even if the lanes use genuinely independent cipher families.
Any deterministic postprocessing of z_0 || z_1 preserves these collisions, so mixing only the two final lane outputs cannot repair the construction.

### Attack 2: make both lane inputs vary together

Public XOR constants do not separate the cipher's key spaces.
For any k and x, set:

```
M(k, x) = (K_0 XOR k) || x || (K_1 XOR k) || x
H'_split(M(k, x)) = F(k, x) || F(k, x).
```

Both lanes now make the identical cipher call.
Any collision of F becomes an H'_split collision even when both lanes' input pairs differ between the two messages.
The first attack's birthday search with x = 0 supplies such a collision at the same 2^64 cost.
The maps `m_0 -> K_0 XOR m_0` and `m_2 -> K_1 XOR m_2` are bijections onto the same full key space.
Choosing unrelated constants, changing their derivation, or hiding their names in the implementation cannot change this fact.

This diagonal subset has only 2^128 possible outputs, all of the form z || z.
It also gives a deterministic distinguisher from a random 256-bit-output function after one chosen-input evaluation, apart from the random function's 2^-128 chance of equal halves.
Collision resistance does not require random-oracle behavior, but the distinction invalidates the previous independent-uniform-lanes argument.

## 5. A precise ideal-cipher bound

This section proves a bound for F and H'_split, including arbitrary key choices, adaptive queries, inverse queries, and final outputs whose cipher calls were not queried beforehand.
It establishes 64-bit collision security in this model, not the required 128-bit level.

For each 128-bit key k, sample an independent uniform permutation E_k on N = 2^128 blocks.
The adversary may query both E_k and E_k^{-1} for any k.
Within a key, answers are sampled without replacement, rather than independently.
Repeated forward or inverse queries to an already known permutation edge reveal no fresh randomness.

Represent a revealed edge by `(k, x, y)` with `y = E_k(x)`, and label it `z = x XOR y`.
The label equals F(k, x).
In a fresh forward query, the adversary fixes k and x and receives y uniformly from that key's unused ciphertexts.
In a fresh inverse query, it fixes k and y and receives x uniformly from that key's unused plaintexts.
If that key already has t_k revealed edges, there are N-t_k possible answers in either case.

Condition on the entire transcript before fresh edge number i, including the adversary's chosen query.
For each earlier label, at most one value of the unknown endpoint makes the new label equal it.
Thus the conditional probability of a collision with any earlier distinct edge is at most `(i-1)/(N-t_k)`.
This argument covers both oracle directions and does not require the query choices to be independent.

After at most q oracle queries, evaluate the adversary's two claimed collision inputs, adding any missing edges to the transcript.
For F this adds at most two edges, and for H'_split it adds at most four.
Use Q = q+4 for a bound covering both cases, and assume Q < N.
Because t_k <= Q-1, a union bound gives:

```
Pr[some distinct revealed edges have equal labels]
    <= sum_{i=1}^{Q} (i-1)/(N-Q+1)
     = Q(Q-1) / (2(N-Q+1)).
```

One may cap this expression at 1.
A collision of F is a label collision between distinct edges.
For H'_split, distinct messages differ in at least one lane's input pair.
The bijective key translation makes the corresponding `(k, x)` pairs distinct, and equality of that lane's outputs gives a label collision.
Therefore the same upper bound applies to any H'_split collision finder.

Together with Section 4's attack, this establishes Theta(2^64) classical ideal-cipher query complexity for constant success probability.
No adaptive inverse-oracle strategy, cross-lane system of equations, or many-message search beats this asymptotic bound within the stated model.
The bound counts all oracle queries, including any preprocessing or shared-cipher use available to the adversary.
It is not a bound for an adversary that receives uncharged precomputed advice about the sampled cipher.

The model follows the all-key forward/inverse setting used in block-cipher hash analyses such as [BRS02].
The proof above concerns this exact function and does not infer security from a construction's MMO name.
Replacing the ideal family by concrete AES is an additional unproven assumption.
AES's usual secret-random-key PRP security does not establish the required behavior under adversarially chosen and related keys.

### 5.1 What an invertible mixing layer cannot repair

Suppose an efficiently invertible public map T sends each 512-bit message to `(u, v)`, with each coordinate 256 bits, and define `H'(m) = F(u) || G(v)` for 128-bit-output F and G.
Fix v and enumerate distinct u values.
Using T's inverse produces distinct messages while the second digest stays fixed.
A birthday search in the first digest costs at most O(2^64) evaluations, with the usual ideal-output behavior giving that scale.
This argument covers a split construction preceded by an invertible linear mixing layer, even if every mixed coordinate depends on every original message block.
Postprocessing the final digest cannot remove an existing collision.

The universal assertion about every expression `(F(u(m)), G(v(m)))` is too broad.
It must exhibit efficient sampling of a fiber that fixes one lane while allowing enough variation in the other.
A large fiber's existence does not provide an efficient way to find its members.
For example, two independent ideal 512-to-128-bit functions on the full message concatenate to an ideal 256-bit-output function.
There is no cheap fixed-coordinate attack in that example.
Thus shared expensive computation is a useful design feature, not a general necessary-condition theorem.
Hirose supplies a concrete shared-key construction with a proof, rather than a claim based on visual mixing.

## Part III — Survey and the selected constructions

## 6. Survey of suitable and unsuitable AES construction families

A double-block-length (DBL) output has 2n bits.
A usual DBL compression step accepts a 2n-bit chaining value and an additional message block.
It therefore maps 3n to 2n bits when that message block has n bits.
"2n-to-2n" alone has no compression and does not describe this interface.
A complete 4n-to-2n mapping requires a composition argument or a separately proved construction.

| Family | Primitive and cost per usual step | Proven object and level in the ideal model | Decision for this interface |
| --- | --- | --- | --- |
| PGV MMO / Davies-Meyer | One AES-128 call and normally one variable key schedule per 128-bit message block | Single-block-length output, birthday level 2^64 for AES | Reject as a 128-bit-CR H' |
| Hirose | Two AES-256 calls sharing one 256-bit key schedule, 384-to-256 step | Arbitrary-state compression CR at constant-success work Omega(2^128), with a linear term in the advantage bound | Selected, with the exact two-step mapping below |
| Abreast-DM | Two AES-256 calls with two generally different schedules, 384-to-256 step | Arbitrary-state compression CR at Omega(2^128) [LK11] | Valid family, less attractive schedule cost, no alternate implementation approved here |
| MDC-2 | Two n-bit-key cipher calls per n-bit message block | The iterated hash has a nontrivial proof, historically about 2^(3n/5), while the unrestricted compression has only 2^(n/2) collision security [M17] | Not approved for the variable-state shortcut or this target |
| MJH | Two AES-128 calls sharing one schedule per 128-bit message block | Later full-version iteration proof reaches 2^(n(1-epsilon)) for any fixed epsilon > 0 [LS15] | Candidate for a fixed-IV iteration review, not an arbitrary-state compression substitute |
| Mennink's three-call DBL | Three n-bit-key cipher calls for a 384-to-256 step | Optimal asymptotic compression collision security [M17] | AES-128 research alternative, requiring exact instantiation and concrete-bound review |
| DMYZ | No unambiguous primary-source construction identified under this name in this review | No theorem can be assigned from the acronym alone | Unassessed, not approved |

Hirose requires a key longer than its block.
The selected instantiation uses AES-256, not two AES-128 calls as a surrogate for an AES-256 call [H06].
Abreast-DM also uses a 2n-bit key.
The shorthand "Abreast-DM gives about 2^126 security" is not a complete claim without a theorem, success probability, and query convention.
This review uses its established Omega(2^n) level and does not certify that numerical shorthand.
For a two-step 512-to-256 adaptation it would need four AES-256 encryptions and four schedules, versus Hirose's four encryptions and two schedules.

MJH illustrates why the proof's object matters.
Its stronger iteration result relies on reachability from the specified initial state.
It does not certify using the first 256 message bits as an arbitrary initial state.
A conventional four-message-block fixed-IV AES-128 MJH evaluation would cost eight encryptions and four schedules before any required padding or finalization.
The asymptotic statement for every fixed epsilon is not a concrete 128-bit guarantee at n = 128 without evaluating the finite bound [LS15].
No blanket rejection of AES-128 DBL designs follows from these exclusions.
Mennink's result supplies a further family to review if AES-256 implementation costs prove unacceptable.

## 7. Selected construction: two Hirose compression steps

### 7.1 Exact function and encoding

Treat each block below as an ordered 16-byte string.
Let E be full 14-round AES-256, using the standard AES byte-string interpretation of its 32-byte key and 16-byte input.
Set `c = 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00` in byte order.
Let `V = V0 || V1` be the deployment's fixed public 32-byte IV, split in input byte order.
V is not secret and is not an AES key schedule.
For 16-byte a, b, and t, define the Hirose compression step [H06]:

```
K = b || t
C(a, b, t) = (E_K(a) XOR a,
              E_K(a XOR c) XOR a XOR c).
```

Split the 64-byte message into consecutive blocks `m0 || m1 || m2 || m3` and define:

```
a0 = m0 XOR V0
b0 = m1 XOR V1
(a1, b1) = C(a0, b0, m2)
(a2, b2) = C(a1, b1, m3)
H'_Hirose,V(m0 || m1 || m2 || m3) = a2 || b2.
```

This definition absorbs the first 32 bytes as variable state and the remaining 32 bytes in two steps.
It uses exactly four AES-256 encryptions and two AES-256 key expansions per H' call.
Both encryptions within a step use the same expanded key and can run in parallel.
The second step depends on the first.
There is no padding because this function's domain is exactly 64 bytes.
A future variable-length API would require a separate encoding and proof.
The IV translation is bijective for every fixed V, including all-zero V.
The collision claim is for one agreed V, not cross-IV collisions.

### 7.2 Ideal-cipher theorem and local composition proof

Use one ideal cipher family with a 256-bit key and a 128-bit block.
The adversary may query encryption and decryption under every key, adaptively.
Let N = 2^128.
Fleischmann, Gorski, and Lucks give the following bound for the arbitrary-state Hirose compression in Appendix B, Definition 5 and Theorem 3 [FGL09]:

```
Adv_C^coll(q) <= 2*q^2/(N - 2*q)^2 + 2*q/(N - 2*q),  q < N/2.
```

Use the compression theorem here.
The original fixed-IV iterated-hash claim alone would not cover the variable initial state in this mapping.
The paper's matching-pair discussion has a variable typo, so use the equations of Definition 5: the paired plaintexts are a and `a XOR c` under the same key.

**Local reduction.** Given distinct messages M and M' with equal H' outputs, evaluate their two-step computations.
If `(a1, b1, m3)` differs from `(a1', b1', m3')`, equality of final outputs gives a C collision at step 2.
Otherwise the triples agree, including m3 and the first-step output.
The messages are distinct, so `(m0, m1, m2)` differs from `(m0', m1', m2')`.
The fixed IV translation preserves this difference.
Thus step 1 gives a C collision.
This extraction loses no success-probability factor and does not assume independent cipher instances across steps.

If the adversary makes q primitive queries and then outputs two candidate messages, evaluating them adds at most eight primitive queries.
With Q = q+8, the resulting explicit bound is:

```
Adv_H'^coll(q) <= min(1, 2*Q^2/(N - 2*Q)^2 + 2*Q/(N - 2*Q)),
                 Q < N/2.
```

For Q <= N/8 the right side is at most 7/18, which is below 1/2.
Thus this bound rules out success 1/2 for q <= 2^125-8 and gives an Omega(2^128) asymptotic constant-success work scale.
The powers and constants must be read together.
This is the requested approximately 128-bit classical collision level, not a literal lower bound of 2^128 queries at success 1/2.
Section 5's rejected function instead has Theta(2^64) constant-success complexity.

The linear term has a structural explanation.
If C(a,b,t) has equal output halves, then C(a XOR c,b,t) is the same ordered output.
Searching for that event under distinct keys takes order N trials in the ideal model.
Set the first-step inputs to such a pair and keep m3 common to lift it into an H' collision.
Together with the lower bound, this gives Theta(2^128) constant-success primitive-query complexity for this exact mapping.
All oracle work, including preprocessing and shared use of the primitive, must count toward q.

For q much smaller than N, the proved advantage has order `q^2/2^256 + q/2^128`.
At q = 2^64 its linear term is about 2^-63, while a random 256-bit-output function's birthday advantage is about 2^-129.
Therefore this construction does not preserve the baseline's random-output advantage curve.
The sibling reductions can use this explicit collision advantage in their failure terms because they extract literal collisions.
If a deployment instead requires a specific small failure probability at a fixed query budget, it must evaluate this bound and the protocol reduction's costs.
The present approval uses the user's constant-success work criterion.

### 7.3 Concrete AES and implementation approval

The theorem concerns an ideal cipher, not a proof about concrete AES-256.
Instantiating it assumes that full-round AES-256 resists the chosen-key collision strategies relevant to this construction.
Ordinary secret-random-key PRP security does not imply that assumption.
Published reduced-round Hirose attacks reinforce the need to preserve the full rounds [CHKM15].
This review does not claim that an ideal-cipher proof removes concrete cipher cryptanalysis risk.
The Blake3 baseline also rests on an unproved concrete collision-resistance assumption.

The exact function above is approved for implementation and measurement under these explicit assumptions and the stated target.
It is more than an unspecified future candidate.
This approval does not certify unwritten code, a CPU/GPU speedup, or an immediate default switch.
Abreast-DM, MJH, the three-call family, and Hirose variants other than the Section 7.4 construction remain candidates for a future instantiation review.

### 7.4 Third review: truncated ChaCha20 instantiation

#### Exact primitive and scope

Define a separate algorithm, `H'_Hirose-ChaCha20,V`, using the two-step equations, c, V, and output order in Section 7.1, with this replacement for E:

```
E_k(x) = ChaCha20_Block(k, LE32(x[12:16]), x[0:12])[0:16].
```

Slices denote bytes, with an exclusive end index.
Use the RFC 8439 Section 2.3 block function: a 32-byte key, a 32-bit counter, a 12-byte nonce, the standard constants, 20 rounds, wordwise addition of the initial state modulo 2^32, and little-endian output [RFC8439].
The initial words 12 through 15 are `LE32(x[12:16]), LE32(x[0:4]), LE32(x[4:8]), LE32(x[8:12])`.
This maps all 128 input bits bijectively into the counter/nonce words.
The key is all 32 bytes of `b || t`, without repetition or reduction.
With the specified c, the second lane flips bit 0 of the first nonce byte.
It does not increment the counter.
The function evaluates one block at the specified counter, including 0 and `0xffffffff`, without stream state or counter advancement.
Nonce uniqueness for secret-key stream encryption is not a requirement of this public hash experiment.

Each H' evaluation uses four ChaCha20 blocks in two dependent stages.
The two blocks within one stage share a key but have distinct inputs.
They can execute in parallel.
There is no expanded key schedule, but all four calls still perform their round computations.
This definition specifies the standard block function, not the bare ChaCha permutation, HChaCha20, or the repository's current PRG wrapper.

#### Where the AES proof uses a permutation

The local collision extraction in Section 7.2 uses determinism and the injective initial-state encoding.
It uses neither inversion nor distinct ciphertexts for distinct plaintexts.
The primitive theorem it invokes does use the ideal-cipher model [FGL09, Appendix B]:

| Proof step | Use of the permutation property | Status for ChaCha |
| --- | --- | --- |
| All-key forward/inverse transcript | An inverse query fixes a ciphertext and samples its unique unused plaintext | No inverse oracle exists for the truncated function |
| Lazy sampling and companion queries | Fresh endpoints exclude previously used endpoints under that key | Random-function answers sample with replacement |
| Non-matching pair collision bound | The two fresh-answer denominators are `N-(2i-2)` and `N-(2i-1)` | Replace with an independently proved `N^2` denominator |
| Matching pair bound | Bound equality of the two feedforward labels using unused endpoints | Prove the event directly with probability `1/N` |
| Two-step extraction and final verification | Equal final states expose a compression collision at one of the two steps | Retained, with at most eight extra primitive calls |

Distinct raw cipher outputs do not imply distinct feedforward labels.
The original matching-pair event already allowed equal labels and accounts for its linear term.
The quoted theorem is inapplicable to a non-permutation as stated.
Applying a PRP/PRF switching lemma would add a birthday-sized loss around 2^64 queries and would fail to establish the target.
That loss would reflect an unsuitable reduction, not a collision attack on the new construction.

#### A direct random-function bound

This subsection proves a new forward-oracle statement for this exact variant.
Let `R(k,x)` be an ideal random function from a 256-bit key and a 128-bit input to 128 bits, with independent uniform answers at distinct `(k,x)` pairs.
Repeated queries return the same answer.
Keys and inputs are public, arbitrary, related, and adaptively chosen.
Set `N = 2^128` and `G(k,x) = R(k,x) XOR x`.
All distinct G entries are also independent uniform n-bit values, where n = 128.

Partition the query domain into two-element sets

```
O(k,x) = {(k,x), (k,x XOR c)}.
```

These sets are disjoint or identical because c is fixed and nonzero.
Choose a canonical representative u for each set, for example the lexicographically smaller input byte string.
Its label pair is `(U,W) = (G(k,u), G(k,u XOR c))`.
The two possible compression inputs in this set return `(U,W)` and `(W,U)`.
The bijection `k = b || t` ensures that there are no further compression inputs with the same oriented primitive queries.

Strengthen the oracle by revealing both entries whenever the adversary first touches a set.
This grants information and cannot reduce the adversary's success.
After Q ordinary queries, including final verification, at most Q such sets have been touched.
For each fresh set, condition on the entire augmented transcript and the next chosen set.
Its pair consists of two fresh independent uniform n-bit values, even if the key occurred before.
A later query to either entry of this set reveals no fresh pair.
This accounts for adaptive strategies that inspect one lane before choosing which pairs to complete.

A compression collision must have one of two forms:

1. Within one set, the two orientations collide exactly when `U = W`, with probability `1/N` per fresh set.
2. Between distinct sets, the fresh pair equals a previous pair or its reversal, with probability at most `2/N^2` per previous set, conditional on the transcript.

A union bound therefore gives

```
Pr[compression collision among Q touched sets]
    <= Q/N + sum_{i=1}^{Q} 2*(i-1)/N^2
     = Q/N + Q*(Q-1)/N^2.

Adv_H'_Hirose-ChaCha20,ideal^coll(q)
    <= min(1, Q/N + Q*(Q-1)/N^2),  Q = q+8.
```

The last line applies Section 7.2's unchanged two-step extraction.
All preprocessing, direct primitive queries, and shared-use oracle queries count toward q.
It covers claimed messages whose evaluations were absent from the original transcript.
For `Q <= N/4`, the bound is at most `5/16`, below 1/2.
Thus constant-success collision finding requires Omega(2^128) classical primitive queries in this model.

The matching attack still costs O(N): fix a, try distinct `k=b||t`, and test `G(k,a) = G(k,a XOR c)` using two queries per trial.
Each trial succeeds with probability `1/N`.
On success, the distinct messages

```
(a XOR V0)         || (b XOR V1) || t || m3
((a XOR c) XOR V0) || (b XOR V1) || t || m3
```

have equal first-step states and therefore equal final hashes for any common m3.
This establishes Theta(2^128) constant-success query complexity for the exact two-step random-function variant.
Its advantage curve still has the order `q/2^128 + q^2/2^256`, rather than the pure birthday curve of an ideal 256-bit-output hash.
The AES theorem's numerical expression also upper-bounds this new expression on its stated domain, but the justification is this separate proof.
Neither result proves the same bound for concrete ChaCha20.

A single-key birthday collision `R(k,x) = R(k,y)` with `x != y` does not even equate the corresponding G labels: their difference is `x XOR y`.
A birthday collision of G equates only one coordinate of C.
The second coordinate and the paired-input structure remain required, as the proof above accounts for.
There is no additional `q^2/2^128` term from allowing repeated raw outputs.

#### Truncation and concrete assumptions

Projecting an ideal 512-bit-output random function onto its first 128 bits gives exactly the R model above.
There is no extra reduction factor or truncation error.
Even if the adversary sees all 512 bits for every query, the other 384 bits can be sampled independently and consistently per entry in that ideal model.
They do not change the fresh-pair argument.
The proof uses n = 128 throughout. It does not claim the collision strength of a 512-bit output.

Concrete ChaCha20 is a fixed public algorithm, not a sampled random function.
The necessary heuristic is that its specified 128-bit projection admits no structure-based collision strategy that materially beats this ideal-model work scale under adversarially chosen keys and inputs.
Describing distinct `(key,input)` evaluations as random-function-like is shorthand for that heuristic.
An ordinary secret-key PRF claim does not establish it.
An oracle distinguishing game that also lets the adversary evaluate the known concrete algorithm locally would be trivial, so this review does not append an unspecified PRF advantage to turn the ideal bound into a concrete theorem.

| Claim or assumption | Evidence and unproved part |
| --- | --- |
| Ideal random-function construction | The proof above covers adaptive all-key forward queries and arbitrary compression states |
| Concrete full-round ChaCha20 projection | Assume resistance to chosen-key, related-key, and related-input collision strategies for this exact mapping. No concrete 128-bit lower bound or numerical failure bound has been proved |
| Access beyond the projection | The public code exposes the full block and the invertible internal permutation. The ideal 512-bit-function extension covers the other output words, but not arbitrary raw-permutation/inverse-permutation access. Exploiting that structure is part of the unproved concrete assumption |
| Reuse with repository PRGs/PRPs | Count equivalent same-state block evaluations as shared queries. Other wrappers expose related internal states and outputs. Independent behavior across wrappers does not follow from a class name, different nonce, or public XOR tweak |
| Comparison with AES-256 | Both concrete instantiations require assumptions beyond secret-key security. AES has a published Hirose ideal-cipher theorem. The ChaCha variant has this local random-function proof and less construction-specific cryptanalytic evidence. Equal ideal work exponents do not establish equal concrete confidence |
| Protocol and implementation | Retain H's CR/XOR-CR assumptions, common-material and nonzero-cs conditions, and the unresolved reference-chain induction. Byte correctness, side channels, and deployed code remain separate checks |

Keep all 20 rounds and the standard constants and additive feedforward.
Barbero, Bellini, and Makarim analyze rotational distinguishers through 17 rounds of the internal permutation, with attacker-controlled states beyond the standard fixed-constant slice [BBM23].
They explicitly distinguish this result from an attack on the stream cipher.
A later analysis studies rotational-XOR propagation with injected constants [BBB22].
These results do not exhibit a collision attack on this full-round, truncated, two-step function.
They remain relevant to the assumption: this hash adversary controls the key and input, and can seek relations that secret-key encryption experiments do not allow.
An applicable attack would need to respect the fixed constants, input encoding, full rounds, projection, and both collision equations, or exploit them through another algorithm.
A statistical distinguisher alone supplies neither that collision nor its cost.
The retained four output words add back public constants, so an attacker can subtract those constants to recover four internal output words.
This feedforward is part of the exact specification, not an extra secrecy claim.
The outer XOR with x acts on nonce/counter bytes placed elsewhere in the initial state and has no direct wordwise cancellation with those constants.
The cited papers do not constitute an exhaustive cryptanalysis of this variant, and full rounds do not prove its chosen-key security.
Reduced-round ChaCha variants receive no approval here.

**Third-review verdict: (a), conditional candidate alongside AES-256.**
The loss of permutation structure does not lower the ideal-model constant-success target.
The new proof supports implementation and measurement under the stated concrete heuristic.
GPU preference requires performance evidence and implementation review. The measurements completed in Section 10.4 and resolved against a GPU preference: both variants measure slower than Blake3.

## Part IV — Roads not taken

## 8. Protocol-specific weakening of H'

A restricted-input proof is a serious research route, but this review does not approve a weakened H'.
The prior collisions are arbitrary hash-input collisions, not malicious VDPF or VDMPF keys realizing them.
Failing a sufficient hash assumption does not by itself give a protocol forgery.
Conversely, failing to lift the attack does not establish protocol security.

The malicious client chooses both keys subject to the sibling analysis's common-material constraints.
It can compute the public H and H', choose seeds and correction words, grind candidate keys, and correlate cs with hash outputs.
Honest-generation pseudorandomness does not characterize this experiment.
An H' leaf input is a corrected H output, internal inputs concatenate H' outputs, chain inputs mix the accumulator with corrected values, and VDMPF also hashes cs anchors.
Consequently, "the inputs are XorHash outputs" does not cover every call site.

A replacement proof would need:

1. An explicit malicious-key generation game and the same common-material and nonzero-cs conditions as the sibling analysis.
2. A characterization of jointly reachable pairs at every collision-extraction site, including adaptivity, correlations, and reuse across buckets and levels.
3. A bound on restricted collision finding for the proposed weak H', or a direct accumulator soundness argument that avoids a general H' collision assumption.
4. A justified model for H under those queries, potentially stronger than the current CR and XOR-CR assumptions.
5. Concrete query and reduction losses across full-domain evaluation, batch evaluation, and repeated instances.

This is new protocol analysis, not a consequence of AES being a PRP or of honest H outputs appearing random.
It would also retain the sibling document's unresolved full chain-binding induction unless that step were proved separately.
Recommend the general-CR Hirose route before investing in this weaker-assumption route.
The research route becomes attractive only if a measured performance advantage justifies the additional proof work and narrower deployment contract.
Its current status is **FEASIBLE via protocol-specific analysis only as an open research direction, with no security approval**.

## 9. What the cited reference implementation actually computes

The source inspected for this review is `sachaservan/vdpf/src/mmo.c`, pinned to its latest file-changing commit `7c966eadd932ffcd06c996b165dd000965ece88b` [REF].
`initMMOHash` initializes an OpenSSL `EVP_aes_128_ctr` context.
`mmoHash4to4` encrypts four input blocks with that context and then XORs those same input blocks into the output.
With non-overlapping input and output buffers and a given starting context state sigma, CTR encryption has the form `Enc_sigma(X) = X XOR S_sigma`.
Therefore the function returns `S_sigma`, independently of X, and advances the context.
The two-stage `mmoHash2to4` similarly cancels its plaintexts, returning four successive keystream blocks.
These are local algebraic conclusions about the inspected functions under normal CTR semantics.
They do not constitute an audit of every caller or a full protocol exploit.

Different readings do not make this an endorsement of the rejected H'.
As a stateful pseudorandom stream generator, its behavior can be intentional and useful.
Reinitializing it with a seed derived from the input would define another function whose security depends on that derivation and its callers.
Resetting it to the same key and counter before each call makes the output independent of the message at that position.
Leaving it stateful makes the result depend on call history rather than only on the message.
None of these observations yields the required stateless 64B-to-32B collision-resistant hash.
The function names and MMO comment cannot override the selected CTR mode and the feedforward arithmetic.

## Part V — Performance: estimates and measurements

## 10. Performance budget and measurement limits

The repository's `doc/site/data.json` records the following `in_bits = 20`, one-key, full-domain measurements.
These are operation timings per key, not timings per domain output.

| Platform and variant | Time |
| --- | ---: |
| CPU, `VDPF-bytes-AES-NI-chains` | 291.628 ms |
| CPU, `DPF-bytes-AES-NI` | 33.091 ms |
| CPU, `VDPF-bytes-AES-NI-merkle` | 334.777 ms |
| GPU, `VDPF-bytes-ChaCha20-x2-merkle` | 0.486162 ms |
| GPU, `DPF-bytes-ChaCha20-x2` | 0.156164 ms |

The CPU records use `wall_time_operation`, and the GPU records use `cuda_event_operation_kernel_launches`.
This review checks their scale and arithmetic without rerunning the benchmarks.

The CPU chain's excess over DPF is about 246.56 ns per domain point.
Two `XorHash` compressions and one H' compression per point give an estimated 82.19 ns per compression.
The fold adds n leaf hashes and n-1 internal hashes to the 2n `XorHash` compressions, giving 4n-1 total compressions for n = 2^20.
Its excess over DPF gives about 71.93 ns per compression under the same attribution.
Thus the original 72-85 ns estimate has the right scale.
These differences also contain correction, allocation, and memory costs, so they are estimates rather than isolated hash timings.
The GPU difference likewise includes the proof pipeline and does not isolate the scalar Blake3 kernel.
Changing H' alone leaves both `XorHash` compressions per point intact.

### 10.1 CPU costs

Use the supplied 2.0 GHz reference frequency only for the following estimates.
Let S128 and S256 be the cost in cycles of a fresh AES-128 and AES-256 key schedule.
Let P256 be the cycles for two AES-256 encryptions under one ready key, with their rounds interleaved where useful.
The supplied 4.3-cycle AES round latency gives about 43 cycles for one AES-128 encryption and about 60.2 cycles for one AES-256 encryption.
It does not give the throughput of a batch or the key-schedule cost.

A four-block Davies-Meyer chain has a rough non-overlapped cost `4*(S128+43)` cycles.
At S128 = 100 cycles this is 286 ns, and it still has only a 128-bit digest with a 64-bit collision level.
For standard MMO the first fixed-IV schedule can be precomputed, giving `3*S128 + 4*43`, or 236 ns at that same assumption.
These are cost scenarios, not measured lower bounds.

The selected H' has the work budget `2*S256 + 4*AES256`, with two paired cipher calls and two dependent stages.
A staged latency estimate is `2*S256 + 2*P256 + overhead` cycles.
Using P256 around 60-120 cycles and an illustrative S256 = 100 cycles gives about 160-220 ns before overhead.
The user's 100+ cycle estimate was for AES-128 expansion, so carrying 100 over to AES-256 is only a sensitivity example.
AESKEYGENASSIST latency, alternative SubWord implementations, schedule/encryption overlap, compiler choices, and the actual CPU all matter.

At 2.0 GHz the Blake3 target of 72-85 ns permits only 144-170 cycles per H' call.
In the staged model, beating it requires `S256 + P256 + overhead/2 < 72-85` cycles.
With P256 near 60 cycles, this leaves about 12-25 cycles for a schedule before overhead.
Under the supplied expensive-schedule scenario, a CPU latency win is unlikely.
This arithmetic cannot exclude a better pipelined schedule or a throughput win across independent Merkle nodes.
The CPU benchmark must report serial-chain latency and independent-node throughput separately.
Historical AES-NI DBL benchmarks explicitly account for schedule sharing and confirm that it is a material cost [BOS11].
Their old processor measurements are not measurements of this repository's exact mapping.

### 10.2 GPU costs and a conditional speed budget

No isolated GPU AES-256 encryption or variable-key schedule measurement is available here.
A defensible per-call budget is four AES-256 encryptions, two fresh expansions, feedforward, and temporary-state movement.
The existing `Aes128Soft` implements ten-round AES-128 with schedules stored by its constructor.
It is reusable implementation material, not an already implemented or timed Hirose primitive.
The new function needs fourteen-round AES-256 and fresh keys for each step of each hash.

A direct adaptation of the existing T-table scheme needs 52 full cipher rounds plus four final rounds per H' call.
That is 832 32-bit table lookups, 64 final-round S-box lookups, and 104 S-box lookups in the two AES-256 schedules, before other operations.
The total is about 1,000 indexed lookups per hash in that implementation model.
Each expanded AES-256 key has 240 bytes, shared between the step's two encryptions.
On-the-fly scheduling may reduce live storage.
Register pressure, spills, shared-memory conflicts, and occupancy can dominate the arithmetic savings.
Calling T-table AES "cheap" does not determine the cost of this variable-key workload.

For a conditional end-to-end estimate, attribute the measured GPU excess `0.486162-0.156164 = 0.329998 ms` to the 4n-1 comparable Blake3 compressions.
This deliberately optimistic simplification excludes differences in correction and memory work.
About half of those compressions belong to H' and half to unchanged H.
If r is new H' aggregate cost divided by old H' aggregate cost, the simplified model gives:

```
T_new_GPU approximately 0.156164 ms + 0.164999 ms * (1 + r).
r = 1: 0.486162 ms
r = 1/2: 0.403663 ms
r = 0: 0.321163 ms
```

These are sensitivity calculations, not predictions or new measurements.
The baseline alone supplies no honest absolute nanosecond estimate for a GPU Hirose call.
Even removing H' cost entirely would leave H and the tree walk, giving about a 1.51x ceiling under this equal-cost attribution.
For the chain, H' is only one of three proof-hash compressions per point.
Thus a claim that H' alone removes 90% of EvalAll cost would confuse H' with the whole proof-hash phase.

### 10.3 ChaCha20 cost estimate, without new measurements

The new variant replaces four AES-256 encryptions and two expansions with four ChaCha20 block evaluations.
Absence of a key schedule removes one cost, but does not imply a CPU win.
An unpruned 20-round block has 80 quarter rounds, each with four additions, four XORs, and four rotations.
Four blocks therefore require 320 quarter rounds, or 3,840 such word operations before initialization, feedforward, and data movement.
Keeping only 16 output bytes can omit unused final stores, additions, and other proven dead operations.
It does not reduce the core work to one quarter of a block.

For a public scalar reference, HACL* Table 2 reports 6.37 cycles per byte for its reference ChaCha20 implementation on an Intel Xeon E5-1630 v4 at 3.70 GHz, using GCC 6.3 and 16 KiB messages [HACL17].
That table measures stream-encryption throughput with amortized setup, not four fresh-key blocks in this hash.
Use its rate only for a scale calculation:

```
4 * 64 bytes * 6.37 cycles/byte = 1,630.72 cycles
at the Section 10.1 reference frequency of 2.0 GHz: about 815 ns
at the source machine's 3.70 GHz: about 441 ns
```

The 256 bytes here count four full blocks of primitive work, not the 64-byte H' input or 32-byte digest.
The calculation suggests hundreds of nanoseconds for scalar code, well above the 72-85 ns baseline estimate.
It is neither a measurement nor a lower bound for this repository or a modern CPU.
Fresh state setup, truncated stores, compiler scheduling, paired-call overlap, and two-stage dependencies can change the result.
The historical paper's scalar stream rate cannot establish single-hash latency or the benefit of batching independent Merkle nodes.
Use the actual same-machine baseline for adoption.

On GPU, ChaCha's integer additions, XORs, and rotations avoid the T-table accesses and expanded-key storage of Section 10.2's AES implementation model.
That is a reason to measure the variant.
Four live 16-word states, or two states at a time with staged evaluation, can still impose substantial register and instruction costs.
Occupancy and spills may reverse the expected benefit, and the unchanged Blake3 H work remains.
Report isolated H' throughput, serial-chain latency, and complete proof workloads separately.
The measurements have completed and are recorded in Section 10.4.

### 10.4 Measured primitive costs (2026-10-17)

A scratch benchmark on cs11369a (RTX A6000, sm_86, CUDA 12.6, `-O3 -arch=native`) ran the exact two-step
Hirose shape of Section 7 with three E instantiations: a T-table AES-256 following the `Aes128Soft` pattern
(constant-memory, shared-memory, and an interleaved shared-memory variant), the RFC 8439 ChaCha20 block function
truncated to 128 bits, and the repository Blake3 compression as the baseline.
Every implementation passed the FIPS-197 C.3 AES-256 and RFC 8439 Section 2.3.2 known-answer tests on host and
device and agreed bit-for-bit across variants.
The GPU timing evaluates one independent hash per thread over 2^20 threads with CUDA events, median of five
repetitions; a contended GPU 0 and a quiet GPU 2 agreed within 4 percent.
The CPU reference is one pinned Xeon Platinum 8352V core.

| E (two-step Hirose shape) | GPU ns/hash (block 128/256) | CPU ns/hash |
| --- | --- | --- |
| Blake3 baseline, one compression | 0.057 / 0.057 (17.6 GH/s) | 118.8-131.5 |
| ChaCha20, RFC 8439, truncated | 0.32 / 0.32 (3.1 GH/s) | 594.4-719.4 |
| AES-256 T-table, shared memory | 2.4 / 2.2 | 716.7-719.1 (soft) |
| AES-256 T-table, shared, interleaved | 1.9 / 1.7 | — |
| AES-256 T-table, constant memory | 12.0 / 12.1 | 188.9-198.6 (AES-NI) |

The ChaCha20 variant is 5.7x slower than Blake3 on the GPU and about 5x slower on the CPU;
the AES-256 variants are 30x (shared or interleaved) to 210x (constant memory) slower on the GPU,
and the AES-NI CPU cost of 188.9-198.6 ns confirms the Section 10.1 estimate band above the Blake3 baseline.
The baseline itself runs near the throughput ceiling in this shape, so the Section 10.2 sensitivity ceiling of
about 1.51x is unreachable through the H' primitive: with r at 5.7 or worse, the model's T_new exceeds the
baseline outright.
Limitations: an independent-hash microbenchmark is not the integrated fold kernel, and the AES-256 T-table
adaptation is untuned (byte-level schedule in local memory, no bank-conflict treatment).
Both gaps favor the Hirose variants, and they remain behind by the margins above.

The performance gate therefore fails on both platforms: no implementation of either variant is warranted on
performance grounds, and Blake3 stays as H'. The security sections stand as the record of what was analyzed
and approved for measurement.

## Part VI — Deployment and implementation requirements

## 11. Deployment and implementation requirements

Both parties must agree on the hash function, byte encoding, IV, accumulator, ordering, and chunking.
The hash choice now distinguishes Blake3, Hirose-AES-256, and Hirose-ChaCha20. AES and ChaCha instantiate different functions even with the same IV.
`Verify` compares 64 proof bytes and contains no algorithm negotiation.
It cannot enforce that agreement or guarantee rejection of every cross-configuration comparison.
For example, an empty VDPF chain initialized from common cs has the same proof regardless of its unused hash choice.
Algorithm agreement must therefore be external to this equality check.

| Deployment | Security-compatible shape | Performance requirement and present decision |
| --- | --- | --- |
| CPU-only | Both parties choose the same specified AES or ChaCha H', with H unchanged | **FEASIBLE on CPU** under that candidate's assumptions. Neither has an approved CPU speedup. Retain the Blake3 baseline |
| GPU-focused | Both parties choose the same variant, GPU executes the expensive proof workload, CPU provides a bit-identical reference or fallback | **FEASIBLE GPU-only** conditional on measured improvement. A slower CPU fallback is acceptable. ChaCha measurements are in progress |
| Mixed CPU/GPU evaluators | Both endpoints compute identical bytes for the chosen hash, accumulator, ordering, and chunking | Correctness is feasible with those agreements. A shared primitive does not make the current CPU sequential chains and CUDA Merkle folds interchangeable |
| Python/JIT callers | Generated extensions select the same hash variant and protocol path as native callers | Include the variant in specialization/cache identity and key/proof configuration. A Python device change must not silently change the algorithm |
| Different hashes on different endpoints | Blake3 versus either Hirose variant, or AES-Hirose versus ChaCha-Hirose | Not an approved configuration. Byte equality is not algorithm negotiation |

No universal slowdown allowance follows from the security proof.
For an acceleration claim, require improvement on the chosen workload including setup, memory work, and all key schedules.
A GPU-focused service can accept a slower rare CPU fallback.
A CPU-heavy deployment should retain Blake3 unless its own measurements justify the change.

Implementation must satisfy the following conditions before adoption:

1. Implement the exact Section 7 equations for the selected variant, full AES-256 or ChaCha20 rounds, nonzero c, IV translation, key concatenation, and output order.
2. Keep H = `XorHash` on Blake3 and exercise every H' call site, including VDMPF anchors, chunk folds, bucket chains, and top folds.
3. Check AES-256 known-answer vectors or RFC 8439 ChaCha20 block vectors, followed by H' vectors with zero, patterned, and arbitrary bytes, then host/device cross-tests and mixed-backend proof comparisons with the same accumulator configuration.
4. Keep the function stateless and thread-safe, with no CTR context or invocation-counter dependence.
5. Count key schedules inside each measurement and distinguish hash latency, independent-hash throughput, complete operation latency, and batch latency.
6. Record whether throughput counts keys or domain outputs, and save benchmark data under `build`.
7. Review device register use, schedule storage, and secret-dependent table accesses if the deployment includes a timing/cache leakage adversary.
8. Keep public library code under `include/fss`, obey the device-code assignment and `int4` conventions, and verify wheel/JIT header discovery if the public header layout changes.

The ideal-model arguments cover black-box functional attacks, not microarchitectural leakage.
Hash inputs can depend on evaluator seeds, so a table implementation must not claim constant-time behavior merely because its algorithm parameters are public.

The sibling document's common-key-material condition, nonzero-cs caveat, and H collision/XOR-collision assumptions remain in force.
Its Merkle-fold collision extraction applies with the new collision bound.
This review does not turn the reference chain's inherited accumulation assertion into a complete induction.

### 11.1 ChaCha implementation checks across native CPU, CUDA, and Python/JIT

At the reviewed HEAD `3bcaac5`, the shared header is `include/fss/prg/chacha.cuh`, not `include/fss/chacha.cuh`.
Its host/device annotations provide reusable implementation structure, but `ChaCha::Gen` computes another function:

- It accepts one 128-bit seed and repeats it into both key halves.
- It fixes both counter words to zero and places a 64-bit nonce in words 14 and 15.
- For `mul <= 2` it uses the 16-byte-key constants, while `mul = 4` uses the 32-byte-key constants with the repeated seed.
- It uses XOR feedforward and selects different output words depending on mul.

Do not call this wrapper as the Section 7.4 primitive or change it globally to make it fit.
A new 256-bit-key block function must preserve existing PRG/PRP behavior and pass independent RFC vectors.
Its arithmetic must implement unsigned 32-bit wraparound. The existing signed `int` additions require review before reuse because ordinary C++ signed overflow is undefined.
Keep the outer Hirose feedforward as XOR, distinct from the inner ChaCha word additions.

| Surface | Required checks for the new implementation |
| --- | --- |
| Native CPU | Independent RFC block vectors, all 32 key bytes, nonce/counter word placement, first-four-word output order, and two-step hash vectors. Include counter 0 and `0xffffffff` and high-bit-set bytes |
| CUDA device | Bit-identical vectors and proofs, defined wraparound and rotation semantics, no stateful counter, and register/spill inspection. Use plain assignments and `int4` storage conventions instead of device `memcpy`, `memset`, or `reinterpret_cast` |
| Python/JIT | Make the selected variant explicit in generated types, cache/configuration identity, and compatible proof paths. Verify wheel contents and JIT include discovery if adding or moving public headers |

Same-state reuse needs explicit accounting.
For `mul = 4`, some repeated-key PRG states coincide with valid states of the new block function.
Given its full output and initial state, an attacker can undo the PRG's XOR feedforward and apply the RFC additive feedforward.
Such evaluations count as queries to the same primitive, not as an independent random function.
Other constant choices and internal-state exposures remain part of Section 7.4's concrete shared-use assumption.
The ideal collision proof does not constitute a joint proof of every ChaCha-based protocol component.

## Part VII — Decision record

## 12. Decision record

- **NOT FEASIBLE:** the previous split-lane H'_split at the requested security level, with or without invertible coordinate mixing or output postprocessing.
  Its ideal-cipher constant-success collision complexity remains Theta(2^64).
- **FEASIBLE on CPU:** Section 7's exact two-step Hirose AES-256 function as a correct, assumption-based cryptographic replacement.
  A CPU speedup is unproved and doubtful under the supplied schedule estimate, so keep Blake3 as the performance baseline.
- **CHACHA VERDICT (a):** Section 7.4's exact two-step Hirose-ChaCha20 function is a conditional candidate alongside AES-256.
  Its direct random-function bound is `min(1, Q/2^128 + Q*(Q-1)/2^256)`, with `Q=q+8`.
  It retains the approximately 2^128 classical constant-success work target without an invertibility assumption or a truncation loss.
  Concrete chosen-key security remains an explicit heuristic. The existing PRG wrapper is not this primitive.
  No CPU win or GPU preference follows. The completed measurements (Section 10.4) put the variant 5.7x behind Blake3 on GPU and about 5x behind on CPU, closing the adoption gate on performance grounds.
- **FEASIBLE GPU-only:** either approved variant with a GPU implementation and a bit-identical CPU reference, conditional on measured GPU improvement.
  CPU speed is not a compatibility or security prerequisite.
- **FEASIBLE via protocol-specific analysis:** a legitimate research possibility requiring the new arguments in Section 8.
  No weakened-H' implementation receives approval through that route today.

**DOC-APPROVE.**
This review approves this requirements analysis, the retained rejection proof, and implementation/benchmarking of both exact Section 7 variants under their respective explicit assumptions.
The ChaCha approval includes the random-function proof, standard block-function encoding, and the implementation checks above.
It does not certify concrete chosen-key security, direct reuse of `ChaCha::Gen`, reduced rounds, a GPU win, random-256-bit advantage bounds, or a default replacement.
Blake3 remains the default. Adoption requires reviewed implementations and workload measurements.

## 13. References

- [dCP22] Leo de Castro and Antigoni Polychroniadou.
  *Lightweight, Maliciously Secure Verifiable Function Secret Sharing.*
  EUROCRYPT 2022, [eprint 2021/580](https://eprint.iacr.org/2021/580).
  The protocol assumptions used here are stated in the [sibling analysis](vdpf_merkle_fold_security.md), Sections 5, 6, and 8.
- [BRS02] John Black, Phillip Rogaway, and Thomas Shrimpton.
  *Black-Box Analysis of the Block-Cipher-Based Hash-Function Constructions from PGV.*
  CRYPTO 2002, [full version](https://www.cs.ucdavis.edu/~rogaway/papers/hash.pdf).
- [H06] Shoichi Hirose.
  *Some Plausible Constructions of Double-Block-Length Hash Functions.*
  FSE 2006, [paper](https://iacr.org/cryptodb/archive/2006/FSE/3233/3233.pdf).
- [FGL09] Ewan Fleischmann, Michael Gorski, and Stefan Lucks.
  *On the Security of Tandem-DM.*
  FSE 2009, [paper](https://www.iacr.org/archive/fse2009/56650090/56650090.pdf), Appendix B, Definition 5 and Theorem 3.
  This citation is to the Hirose compression bound in the appendix, not to a Tandem-DM instantiation.
- [LK11] Jooyoung Lee and Daesung Kwon.
  *The Security of Abreast-DM in the Ideal Cipher Model.*
  IEICE Transactions E94-A(1), 2011, [paper record](https://eprint.iacr.org/2009/225), [publisher abstract](https://globals.ieice.org/en_transactions/fundamentals/10.1587/transfun.E94.A.104/_p).
- [LS15] Jooyoung Lee and Martijn Stam.
  *MJH: A Faster Alternative to MDC-2.*
  Designs, Codes and Cryptography 76, 2015, [full-version record](https://eprint.iacr.org/2014/108).
  This later iteration bound must not be confused with the weaker 2011 bound or with arbitrary-state compression security.
- [M17] Bart Mennink.
  *Optimal Collision Security in Double Block Length Hashing.*
  Designs, Codes and Cryptography, [author manuscript](https://www.cs.ru.nl/~bmennink/pubs/17dcc.pdf), Table 1 and Sections 4-5.
- [BOS11] Joppe W. Bos, Onur Ozen, and Martijn Stam.
  *Efficient Hashing Using the AES Instruction Set.*
  CHES 2011, [paper](https://www.iacr.org/archive/ches2011/69170507/69170507.pdf).
- [CHKM15] Jiageng Chen, Shoichi Hirose, Hidenori Kuwakado, and Atsuko Miyaji.
  *A Collision Attack on a Double-Block-Length Compression Function Instantiated with Round-Reduced AES-256.*
  [Author manuscript record](https://dspace.jaist.ac.jp/dspace/handle/10119/13465).
- [W02] David Wagner.
  *A Generalized Birthday Problem.*
  CRYPTO 2002, [paper](https://www.enseignement.polytechnique.fr/informatique/profs/Francois.Morain/Master1/Crypto/projects/Wagner02.pdf), Section 2.
- [REF] Sacha Servan-Schreiber, VDPF reference implementation.
  [Pinned src/mmo.c](https://github.com/sachaservan/vdpf/blob/7c966eadd932ffcd06c996b165dd000965ece88b/src/mmo.c), inspected on 2026-10-17.
- [RFC8439] Yoav Nir and Adam Langley.
  *ChaCha20 and Poly1305 for IETF Protocols.*
  [RFC 8439](https://www.rfc-editor.org/rfc/rfc8439.html), Sections 2.1, 2.3, and 2.3.2, June 2018.
- [BBM23] Stefano Barbero, Emanuele Bellini, and Rusydi H. Makarim.
  *Rotational Analysis of ChaCha Permutation.*
  Advances in Mathematics of Communications 17(6), 2023, [published article](https://www.aimsciences.org/article/doi/10.3934/amc.2021057).
  Use the published version when quoting results. The preprint's numerical bounds differ.
- [BBB22] Stefano Barbero, Danilo Bazzanella, and Emanuele Bellini.
  *Rotational Cryptanalysis on ChaCha Stream Cipher.*
  Symmetry 14(6), 1087, 2022, [article](https://www.mdpi.com/2073-8994/14/6/1087).
- [HACL17] Jean Karim Zinzindohoue, Karthikeyan Bhargavan, Jonathan Protzenko, and Benjamin Beurdouche.
  *HACL\*: A Verified Modern Cryptographic Library.*
  CCS 2017, [author manuscript](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/tmp536.pdf), Table 2 and Section 7.
  The ChaCha20 reference row is 6.37 cycles/byte. The adjacent 6.26 row is Salsa20.
