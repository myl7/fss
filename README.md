# fss

[![Crates.io](https://img.shields.io/crates/d/fss-rs)](https://crates.io/crates/fss-rs)
[![docs.rs](https://img.shields.io/docsrs/fss-rs)](https://docs.rs/fss-rs)

Function secret sharing including distributed comparison & point functions

## Get started

First add the crate as a dependency:

```bash
# Run in your project directory
cargo add fss-rs
```

By default the embedded PRG and multi-threading are included.
You can disable the default feature to select by yourself.

Then construct a PRG implementing the corresponding [`Prg`] trait, and construct an impl `DcfImpl` or `DpfImpl` to use the PRG.
Check the doc comment for the meanings of the generic parameters.

```rust
use rand::prelude::*;

use fss_rs::dcf::prg::Aes256HirosePrg;
use fss_rs::dcf::{Dcf, DcfImpl};

let keys: [[u8; 32]; 2] = thread_rng().gen();
let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
// DCF for example
let dcf = DcfImpl::<16, 16, _>::new(prg);
```

Finally, for key generation, construct the function to be shared together with 2 init keys, and call `gen`:

```rust
use fss_rs::dcf::{BoundState, CmpFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;

let s0s: [[u8; 16]; 2] = thread_rng().gen();
let f = CmpFn {
  alpha: thread_rng().gen(),
  // `ByteGroup` for example
  beta: ByteGroup(thread_rng().gen()),
  bound: BoundState::LtBeta,
};
let keys = dcf.gen(&f, [&s0s[0], &s0s[1]]);
```

See the doc comment of the returned `Share` for how to split it into 2 shares.
The 2 shares are combined like this because they share many fields.

And for evaluation, construct the evaluated points, reverse the output buffer, and call `eval`:

```rust
let x: [u8; 16] = thread_rng().gen();
let mut y = ByteGroup::zero();
// The 2 parties use `true` / `false` to evaluate independently
dcf.eval(false, &k, &[&x], &mut [&mut y]);
```

Full domain evaluation has not been implemented yet.
Use the current batch evaluation consumes near double time than the optimized full domain evaluation.
We plan to implement it in the future, but no guarantee can be made so far.

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
