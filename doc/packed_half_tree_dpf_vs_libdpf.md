# PackedHalfTreeDpf and LibDPF: what a LibDPF user needs to know

`PackedHalfTreeDpf` (include/fss/packed_half_tree_dpf.cuh) is our counterpart to
[LibDPF](https://github.com/weikengchen/libdpf)'s packed full-domain evaluation: when your
outputs are narrower than 128 bits, one 16-byte block carries `128 / out_bits` consecutive
outputs, so the last `log2(128 / out_bits)` levels of PRG expansion and per-input output
storage disappear. If you came from LibDPF, the performance profile and the packed output
shape are the same idea. The constructions differ where it matters for privacy, and this
document explains the difference from a user's point of view: what LibDPF's packing gives
up, why we could not reuse it verbatim, and exactly where our scheme diverges.

The scheme is **experimental**: no published paper describes this exact composition (see
the `@warning` in the header), and the API may change without notice.

## The packing trick both schemes share

A DPF full-domain evaluation over `N = 2^n` inputs normally expands a PRG tree to depth
`n` and converts every leaf into an output. When each output is a single bit, LibDPF stops
the tree early instead: at level `n - 7`, every 128-bit node already holds exactly the
material for the 128 one-bit outputs of the inputs below it, so the nodes themselves become
the packed output blocks (third_party/libdpf/libdpf-c/fsseval.c:49). A single final
correction word turns the block containing alpha into the packed beta:

```
finalCW = packed_beta XOR node_0 XOR node_1     (node_b: party b's on-path block)
```

`EvalAll` then costs about `2 * N / 128` PRG calls and writes `N / 128` blocks. We keep
this truncation and this correction-word shape; `kLanes = 128 / out_bits` and
`kDepth = in_bits - log2(kLanes)` are the generalized counterparts (out_bits in
{1, 2, 4, 8, 16, 32, 64}, not only 1).

## What LibDPF's raw-block outputs give up

LibDPF returns the tree nodes **as they are**, with no output conversion. That is where the
differences start.

1. **Shares carry deterministic structure.** In these trees the least significant bit of
   every node is reserved for the control-bit machinery: seeds are LSB-clamped to zero, and
   the control bit rides in that position with public conventions. A packed block therefore
   has a bit that is a control-bit slot rather than pseudorandom output material, and an
   individual share is distinguishable from a random 128-bit string. You can see the
   special-casing in LibDPF's own final-correction construction, which flips the block's LSB
   twice around the lane shift purely to compensate that slot
   (third_party/libdpf/libdpf-c/fsseval.c:107-135). Strictly speaking, the guarantee that
   "each party's output shares are indistinguishable from random" does not hold for raw
   blocks.

2. **The final correction word reveals the on-path node XOR.** Every key carries
   `finalCW`, and for one-bit DPF the payload beta is usually a public constant (in LibDPF's
   PIR-style usage it is 1). A key holder who knows beta can strip it off and recover
   `node_0 XOR node_1`, then combine with their own on-path node to compute the *other
   party's raw node* at the truncation level. In LibDPF's BGI-style tree the damage is
   confined to that single level's node, because the tree below the truncation level does
   not exist. It is a bounded leak, and LibDPF evidently accepts it — but it is precisely
   the kind of leak that makes raw-node reuse impossible to harden by tweaks.

3. **The same trick on a Half-Tree is not leaky but catastrophic.** This is why we did not
   simply graft LibDPF's final level onto our HalfTreeDpf. In the Half-Tree expansion the
   on-path difference `node_0 XOR node_1` is *constant across every level*: it equals the
   XOR of the two root seeds. If we returned raw nodes as packed outputs, the final
   correction word would be `finalCW = packed_beta XOR (s0 XOR s1)` — the root-seed XOR.
   Party b knows their own root seed and beta, so they would compute the other party's root
   seed as `finalCW XOR packed_beta XOR s_b` and reconstruct the *entire other key*. A
   LibDPF-style raw packing on a Half-Tree is a full key compromise, not a partial leak.

## What we add: a domain-separated leaf conversion

Our packed outputs are not raw nodes. After the truncated Half-Tree expansion, every
output block passes through one extra hash call under a separate `hash_key`:

```
Gen:    leaf_b   = PRG(hash_key XOR node_b)              (both parties' on-path nodes)
        fcw      = packed_beta XOR leaf_0 XOR leaf_1
Eval:   y_b      = PRG(hash_key XOR node_b) XOR (t_b ? fcw : 0)
```

In code: the two PRG calls and the `fcw` assignment at the end of
`PackedHalfTreeDpf::Gen` (include/fss/packed_half_tree_dpf.cuh:179-182), and
`ConvertLeaf`, which every `Eval` and `EvalAll` output goes through
(include/fss/packed_half_tree_dpf.cuh:298-302). `t_b` is party b's control bit, as in the
rest of the library.

Why this restores privacy:

- `fcw` now involves hashes of the on-path nodes, which are pseudorandom and unrelated to
  the constant root-seed XOR, so a key holder learns nothing about the other party's seeds.
- The hash inputs are the level-`kDepth` nodes, which are never expanded afterwards, so
  the expansion PRG and the conversion hash read disjoint input sets — the same
  domain-separation discipline as `HalfTreeDpf`'s last-level conversion.
- Every output block is a PRG output (optionally corrected by `fcw`), so individual shares
  are again indistinguishable from random 128-bit strings, including bit 0.

## What the fix costs

The leaf hash is one extra PRG call per output block per party, so full-domain evaluation
costs `2^(kDepth + 1)` PRG blocks — about twice LibDPF's truncated expansion. Measured on
our reference host (single thread, AES-NI PRG, `out_bits = 1`), full-domain evaluation is
1.4-1.7x slower than LibDPF for `n >= 16` (at `n = 20`: 0.228 ms vs 0.136 ms per party).
Compared with our own non-packed schemes, the same measurement is ~111x faster than
`HalfTreeDpf::EvalAll` and ~152x faster than `Dpf::EvalAll`, with 128x smaller output
arrays. Key size is `16 * (kDepth + 1)` bytes: `kDepth` correction words plus `fcw`.

## Porting notes

| | LibDPF | PackedHalfTreeDpf |
| --- | --- | --- |
| Output width | 1 bit | 1-64 bits, power of two (`out_bits`) |
| Output group | XOR | XOR only (additive groups cannot share packed lanes) |
| Expansion | BGI DPF, `mul=2` PRG | Half-Tree, `mul=1` PRG (CCR hash, 128->128 bits) |
| Extra key material | — | `hash_key` (sample it like a seed; it feeds the leaf conversion) |
| Reading one output | bit arithmetic | `Extract(block, x)` returns the `out_bits` lane for input `x` |
| Checking shares | XOR blocks, inspect bit | XOR blocks, then `Extract`, or compare against `PackBeta(alpha, beta)` |
| DCF packing | — | none: comparison outputs need per-leaf control bits, so no packed DCF exists |
| Status | — | experimental, no paper for the composition |

Two conventions worth knowing when you switch:

- `alpha`'s low `log2(kLanes)` bits select the lane inside the packed block; the high bits
  select the block. `PackBeta` mirrors this layout when you build expected values in tests.
- The output group is XOR, so the two parties' blocks XOR-reconstruct to the packed beta
  block at alpha's block and to the zero block everywhere else — there is no group
  conversion layer like `Bytes::From`.

## When to use which

Use `PackedHalfTreeDpf` when your outputs are at most 64 bits wide and XOR-shared, you
evaluate (near-)full domains, and share pseudorandomness matters — the typical LibDPF
workload. Use `HalfTreeDpf` or `Dpf` for 127-bit group outputs, point queries, or when you
need the published, proven constructions. If you only need LibDPF's speed and can live
with raw-node shares, LibDPF itself remains the reference point for that trade-off; our
scheme exists so that the packed-output workflow does not force that trade-off on you.
