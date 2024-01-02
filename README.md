# fss

| Name                | Crate name                 | crates.io                                                                                       | Docs                                                                               |
| ------------------- | -------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| DCF                 | [dcf](./dcf)               | [![Crates.io](https://img.shields.io/crates/d/dcf)](https://crates.io/crates/dcf)               | [![docs.rs](https://img.shields.io/docsrs/dcf)](https://docs.rs/dcf)               |
| DPF                 | [dpf-fss](./dpf-fss)       | [![Crates.io](https://img.shields.io/crates/d/dpf-fss)](https://crates.io/crates/dpf-fss)       | Coming soon                                                                        |
| Group (mathematics) | [group-math](./group-math) | [![Crates.io](https://img.shields.io/crates/d/group-math)](https://crates.io/crates/group-math) | [![docs.rs](https://img.shields.io/docsrs/group-math)](https://docs.rs/group-math) |
| Common types        | [fss-types](./fss-types)   | [![Crates.io](https://img.shields.io/crates/d/fss-types)](https://crates.io/crates/fss-types)   | [![docs.rs](https://img.shields.io/docsrs/fss-types)](https://docs.rs/fss-types)   |

Function secret sharing implementations including distributed comparison & point function

## References

- DCF: Elette Boyle, Nishanth Chandran, Niv Gilboa, Divya Gupta, Yuval Ishai, Nishant Kumar, and Mayank Rathee. "[Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation](https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30)." In _EUROCRYPT_. 2021.
- DPF: Elette Boyle, Niv Gilboa, and Yuval Ishai. "[Function Secret Sharing: Improvements and Extensions](https://eprint.iacr.org/2018/707)." In _CCS_. 2016.
- Fast PRG: Leo de Castro and Anitgoni Polychroniadou. "[Lightweight, Maliciously Secure Verifiable Function Secret Sharing](https://eprint.iacr.org/2021/580)." In _EUROCRYPT_. 2022.
- Fast PRG: Frank Wang, Catherine Yun, Shafi Goldwasser, Vinod Vaikuntanathan, and Matei Zaharia. "[Splinter: Practical Private Queries on Public Data](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/wang-frank)." In _NDSI_. 2017.

## Benchmark

We use [Criterion.rs] for benchmarking.
Criterion.rs reports `criterion.tar.zst` are included in releases.

We use a (my) laptop as the benchmarking machine.
It is charged to 100% with the power plugged in when benchmarking.
Its CPU is [AMD Ryzen 7 5800H with Radeon Graphics], which is 8C16T.
We use [rayon] as the data-parallelism library, which establishes 16 threads when benchmarking with multithreading.
We ensure that its memory is enough for benchmarking, which is 16GB and has more than 5GB left when benchmarking.
Notice that we do not close all other programs as many as possible to reduce scheduling, though we do avoid doing any other thing at the same time.

[Criterion.rs]: https://github.com/bheisler/criterion.rs
[AMD Ryzen 7 5800H with Radeon Graphics]: https://www.amd.com/en/products/apu/amd-ryzen-7-5800h
[rayon]: https://github.com/rayon-rs/rayon

## License

Copyright (C) myl7

SPDX-License-Identifier: Apache-2.0
