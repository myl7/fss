# myl7/fss

Function secret sharing (FSS) primitives including:

- 2-party distributed point function (DPF)
- 2-party distributed comparison function (DCF)

[Documentation](https://myl7.github.io/fss/)

Features:

- First-class support for GPU (based on CUDA)
- Well-commented and documented

## Introduction

**Multi-party computation (MPC)** is a subfield of cryptography that aims to enable a group of parties (e.g., servers) to jointly compute a function over their inputs while keeping the inputs private.

**Secret sharing** is a method that distributes a secret among a group of parties, such that no individual party holds any information about the secret.
For example, a number $x$ can be secret-shared into $x_0, x_1$ via $x = x_0 + x_1$.

**FSS** is a scheme to secret-share a function into a group of function shares.
Each function share, called as a **key**, can be individually evaluated on a party.
The outputs of the keys are the shares of the original function output.
FSS consists of 2 methods: `Gen` for generating function shares as keys and `Eval` for evaluating a key to get an output share.
FSS's workflow is shown below:

[![](https://mermaid.ink/img/pako:eNpVkc1OwzAQhF_F2gNKhBPZJc2PBZVKoFzohd6QL6ax20iJXRkHGqq-O05KK2r54PF-M3PYA6xNJYGBasz3eiusQ69vXCN_5gEHRTBSFD2gxWoVv0gdqJBDeJo_Eg_05G_4_CWaYMD3_wg6EPSKoFdE6YGFsaj3jAr2Ib7_sDOfeYtGW38B5yiKZr4S3fjUc_8oUBRHqOQaMGxsXQFztpMYWmlbMUg4DDgHt5Wt5MD8s5JKdI3jwPXR23ZCvxvTnp3WdJstMCWaT6-6XSWcfKrFxor28mulrqQtTacdMJrRyZgC7AB7YGkWp3cZoSRJUn-LBEMPrMhiOkmKIiFFTvOcTo8YfsZaEufZFIOsamfs8rSOcSvHXweacxw?type=png)](https://mermaid.live/edit#pako:eNpVkc1OwzAQhF_F2gNKhBPZJc2PBZVKoFzohd6QL6ax20iJXRkHGqq-O05KK2r54PF-M3PYA6xNJYGBasz3eiusQ69vXCN_5gEHRTBSFD2gxWoVv0gdqJBDeJo_Eg_05G_4_CWaYMD3_wg6EPSKoFdE6YGFsaj3jAr2Ib7_sDOfeYtGW38B5yiKZr4S3fjUc_8oUBRHqOQaMGxsXQFztpMYWmlbMUg4DDgHt5Wt5MD8s5JKdI3jwPXR23ZCvxvTnp3WdJstMCWaT6-6XSWcfKrFxor28mulrqQtTacdMJrRyZgC7AB7YGkWp3cZoSRJUn-LBEMPrMhiOkmKIiFFTvOcTo8YfsZaEufZFIOsamfs8rSOcSvHXweacxw)

**DPF/DCF** are FSS for point/comparison functions.
They are called out because 2-party DPF/DCF can have $O(\log N)$ key size, where $N$ is the input domain size.
Meanwhile, 3-or-more-party DPF/DCF and general FSS have $O(\sqrt{N})$ key size.
More details, including the definitions and the implementation details that users must care about, can be found in the documentation of dpf.cuh and dcf.cuh files.

## License

Apache License, Version 2.0

Copyright (C) 2026 Yulong Ming <i@myl.moe>
