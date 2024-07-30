# fss

[![Crates.io](https://img.shields.io/crates/d/fss-rs)](https://crates.io/crates/fss-rs)
[![docs.rs](https://img.shields.io/docsrs/fss-rs)](https://docs.rs/fss-rs)

Function secret sharing including distributed comparison & point functions

## Get Started

First add the crate as a dependency:

```bash
# Run in your project directory
cargo add fss-rs
```

By default the PRG implementations and multi-threading are included.
You can disable the default feature to select by yourself.
**If you are on ARM machines**, see the [_Performance_](#performance) section for hardware acceleration.

Then construct a PRG implementing the corresponding `Prg` trait, and construct an impl `DcfImpl` or `DpfImpl` to use the PRG.
Check the doc comment for the meanings of the generic parameters.
If you want to set the **bit** length of the **domain** (input domain, instead of the range that is the output domain), check `new_with_filter` method.

```rust
use rand::prelude::*;

// Matyas-Meyer-Oseas (via AES128) provides 128-bit security and should be enough.
// Hirose (via AES256) still only provides 128-bit security because the output is not chained.
// But Hirose can be helpful is you are forced to choose AES256.
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
use fss_rs::dcf::{Dcf, DcfImpl};

let keys: [[u8; 16]; 4] = thread_rng().gen();
let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(std::array::from_fn(|i| &keys[i]));
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

For full domain evaluation, use `full_eval` instead.
While similar to `eval`, `full_eval` does not accept a vector of `x`, and instead expects a vector of `y` whose length is `2 ** (IN_BLEN * 8)` to store all evaluated `y`.

More examples are available as benchmarks in the [benches dir](./benches)

## References

- DCF: Elette Boyle, Nishanth Chandran, Niv Gilboa, Divya Gupta, Yuval Ishai, Nishant Kumar, and Mayank Rathee. "[Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation](https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30)." In _EUROCRYPT_. 2021.
- DPF: Elette Boyle, Niv Gilboa, and Yuval Ishai. "[Function Secret Sharing: Improvements and Extensions](https://eprint.iacr.org/2018/707)." In _CCS_. 2016.
- Fast PRG: Leo de Castro and Anitgoni Polychroniadou. "[Lightweight, Maliciously Secure Verifiable Function Secret Sharing](https://eprint.iacr.org/2021/580)." In _EUROCRYPT_. 2022.
- Fast PRG: Frank Wang, Catherine Yun, Shafi Goldwasser, Vinod Vaikuntanathan, and Matei Zaharia. "[Splinter: Practical Private Queries on Public Data](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/wang-frank)." In _NDSI_. 2017.

## Performance

The hot path of the project is PRG and XOR operations.
We use the [aes crate of RustCrypto] for PRG.
For XOR operations, We use [Rust std SIMD] for nightly Rust, or the [wide crate] that has 9M all-time downloads for stable Rust.

[Rust std SIMD]: https://doc.rust-lang.org/std/simd/index.html
[aes crate of RustCrypto]: https://crates.io/crates/aes
[wide crate]: https://crates.io/crates/wide

For PRG, enabling archtecture-specified CPU intrinsics can largely boost the performance.
The aes crate by default performs runtime detection of CPU intrinsics and uses them if available.

For x86/x86_64 (`i686`/`x86_64` in targets), AES-NI is used if available, which works out-of-the-box.

For ARMv8 (`aarch64` in targets), while ARMv8 Cryptography Extensions is supported, to use it,
in addition to the above, you need to set some flags to enable it.
See the [doc of the aes crate] for details.
We also quote the section here:

[doc of the aes crate]: https://docs.rs/aes/latest/aes/

> From <https://docs.rs/aes/0.8.3/aes/#armv8-intrinsics-rust-161>
>
> ## ARMv8 intrinsics (Rust 1.61+)
>
> On `aarch64` targets including `aarch64-apple-darwin` (Apple M1) and Linux
> targets such as `aarch64-unknown-linux-gnu` and `aarch64-unknown-linux-musl`,
> support for using AES intrinsics provided by the ARMv8 Cryptography Extensions
> is available when using Rust 1.61 or above, and can be enabled using the
> `aes_armv8` configuration flag.
>
> On Linux and macOS, when the `aes_armv8` flag is enabled support for AES
> intrinsics is autodetected at runtime. On other platforms the `aes`
> target feature must be enabled via RUSTFLAGS.

## Benchmark

We use [Criterion.rs] for benchmarking.
Criterion.rs reports `criterion.tar.zst` are included in releases.

We use a (my) laptop as the benchmarking machine.
It is charged with the power plugged in when benchmarking.
Its CPU is [AMD Ryzen 7 5800H], which is 8C16T.
We use [rayon] as the data-parallelism library, which establishes 16 threads when benchmarking with multithreading.
We ensure its memory is enough for benchmarking.
Notice that we do not close other programs as many as possible to reduce scheduling, though all GUI applications except VSCode are closed and we do avoid doing any other things at the same time.

[Criterion.rs]: https://github.com/bheisler/criterion.rs
[AMD Ryzen 7 5800H]: https://www.amd.com/en/product/10821
[rayon]: https://github.com/rayon-rs/rayon

## Changelog

Correctness fixes:

- `fss_rs::group::int_prime` impl is corrected at v0.6.0.
- `fss_rs::dcf::prg::Aes256HirosePrg` impl is corrected at v0.4.3 of the crate fss-rs, v0.5.2 and v0.6.3 of the crate dcf, and v0.5.2 of the crate dpf-fss (since it depends on the crate dcf).

See [CHANGELOG.md](./CHANGELOG.md) for the full changelog.

## License

Copyright (C) 2023 Yulong Ming (myl7)

SPDX-License-Identifier: Apache-2.0
