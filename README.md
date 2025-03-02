# fss: FSS primitives including DPF and DCF

Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)

## Preliminaries

For a function $f$ whose input domain is $\mathbb{G}^{in}$ and output domain is a [(math) group](<https://en.wikipedia.org/wiki/Group_(mathematics)>) $\mathbb{G}^{out}$, FSS is a scheme to secret-share this **function** into $M$ functions $f_b$ for $b \in [M]$ with **correctness** and **privacy**:

-   **Correctness**: For any input $x \in \mathbb{G}^{in}$, $f(x) = \sum_{b = 1}^{M} f_b(x)$
-   **Privacy**: For any strict subset of parties $B \subset [M]$, $\\{f_b | b \in B\\}$ reveals no information about $f(x)$

More formal definitions can be found in the following papers:

-   [Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation](https://doi.org/10.1007/978-3-030-77886-6_30)
-   [Secure Computation with Preprocessing via Function Secret Sharing](https://doi.org/10.1007/978-3-030-36030-6_14)
-   [Function Secret Sharing: Improvements and Extensions](https://doi.org/10.1145/2976749.2978429)
-   [Function Secret Sharing](https://doi.org/10.1007/978-3-662-46803-6_12)

Assume that the cardinal (size) of the input domain $N = |\mathbb{G}^{in}|$, the trivial method for FSS is to secret-share all $N$ mappings $\\{x \rightarrow f(x) | x \in \mathbb{G}^{in}\\}$, resulting in $O(N)$ communication costs.
DPF and DCF trade higher computation costs for lower communication costs.
2-party DPF and DCF result in $O(\log N)$ communication costs, and 3-or-more-party ones (based on seed homomorphic pseudo-random functions) result in $O(\sqrt{N})$ communication costs.

## Limitations

-   We use $b \in \\{0\\} \cup [M - 1]$ other than $b \in [M]$ that is used by the papers, because computer science counts from 0
-   Currently, this library only implements 2-party DPF and DCF, fixing $M = 2$ and $b \in \\{0, 1\\}$
-   We fix input to be bits and output to be bytes.
    $\lambda$ is fixed to be a multiple of 8.
    However, users can still customize how output bytes as group elements should be computed, e.g., added.

## Licenses

Copyright (C) 2025 Yulong Ming (myl7)

Apache License, Version 2.0
