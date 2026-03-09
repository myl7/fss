# Half-Tree DPF Spec Corrections

Corrections to the algorithm spec extracted from "Half-Tree: Halving the Cost of Tree Expansion in COT and DPF" (Guo et al., Eurocrypt 2023). Found during implementation and verified by invariant analysis + 27 passing tests.

## 1. CW formula (levels 1..n-1)

Spec says:
```
CW_i = H_S(node_0) ^ H_S(node_1) ^ alpha_i * Delta
```

Correct formula:
```
CW_i = H_S(node_0) ^ H_S(node_1) ^ (1 - alpha_i) * Delta
```

Reason: the CW must make the off-path (non-alpha) direction converge (shares equal). When alpha_i = 0, on-path is left and off-path is right. The right child is `left ^ parent`, so its party difference includes Delta from the parent. XORing Delta into the CW cancels it for the right child while preserving Delta for the left (on-path) child.

Derivation: for on-path parent (node0 ^ node1 = Delta, t0 ^ t1 = 1), the corrected child difference for direction x_i is:

```
diff(x_i) = x_i * Delta ^ CW_term
```

where CW_term = (1 - alpha_i) * Delta. So:
- x_i = alpha_i (on-path): diff = alpha_i * Delta ^ (1-alpha_i) * Delta = Delta
- x_i != alpha_i (off-path): diff = (1-alpha_i) * Delta ^ (1-alpha_i) * Delta = 0

## 2. HCW (level n)

Spec says:
```
HCW = high_{alpha_n}_0 ^ high_{alpha_n}_1
```

Correct formula:
```
HCW = high_{!alpha_n}_0 ^ high_{!alpha_n}_1
```

Reason: HCW corrects the non-alpha direction so both parties converge on the high bits. The alpha direction's mismatch is absorbed by the output correction word (ocw/CW_{n+1}).

## 3. LCW (level n)

Spec says:
```
LCW_0 = low_0_party0 ^ low_0_party1 ^ alpha_n
LCW_1 = low_1_party0 ^ low_1_party1 ^ alpha_n
```

Correct formula:
```
LCW_0 = low_0_party0 ^ low_0_party1 ^ !alpha_n
LCW_1 = low_1_party0 ^ low_1_party1 ^ alpha_n
```

Reason: the corrected low-bit difference for sigma direction is:
```
corrected_diff(sigma) = (low_sigma_0 ^ low_sigma_1) ^ LCW_sigma
```

We need: alpha direction (sigma = alpha_n) has diff = 1, non-alpha direction (sigma = !alpha_n) has diff = 0.

With the corrected formula:
- sigma = 0: corrected diff = !alpha_n (so diff=1 when alpha_n=0, diff=0 when alpha_n=1)
- sigma = 1: corrected diff = alpha_n (so diff=0 when alpha_n=0, diff=1 when alpha_n=1)

This ensures exactly the alpha direction has low diff = 1 (one party adds ocw) and the non-alpha direction has low diff = 0 (cancellation).
