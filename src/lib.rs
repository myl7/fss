// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Many variable names and the LaTeX math expressions in the doc comment are from the paper _Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation_.

#![feature(portable_simd)]
#![feature(array_chunks)]
#![feature(iter_array_chunks)]

use group::Group;

pub mod dcf;
pub mod dpf;
pub mod group;
pub mod utils;

/// Point function.
/// Despite the name, it only ships an element of the input domain and an element of the output domain.
/// The actual meaning of the 2 elements is determined by the context.
///
/// - `IN_BLEN` is the **byte** length of the size of the input domain.
/// - `OUT_BLEN` is the **byte** length of the size of the output domain. `$\lambda$` (but the byte length).
pub struct PointFn<const IN_BLEN: usize, const OUT_BLEN: usize, G>
where
    G: Group<OUT_BLEN>,
{
    /// `$\alpha$`, or say `x` in `y = f(x)`.
    pub alpha: [u8; IN_BLEN],
    /// `$\beta$`, or say `y` in `y = f(x)`.
    pub beta: G,
}

macro_rules! decl_prg_trait {
    ($ret_elem:ty) => {
        /// Pseudorandom generator.
        ///
        /// Requires `Sync` for multi-threading.
        /// We still require it for single-threading since it should be still easy to be included.
        pub trait Prg<const OUT_BLEN: usize>: Sync {
            fn gen(&self, seed: &[u8; OUT_BLEN]) -> [$ret_elem; 2];
        }
    };
}
pub(crate) use decl_prg_trait;

/// `Cw`. Correclation word.
#[derive(Clone)]
pub struct Cw<const OUT_BLEN: usize, G>
where
    G: Group<OUT_BLEN>,
{
    pub s: [u8; OUT_BLEN],
    pub v: G,
    pub tl: bool,
    pub tr: bool,
}

/// `k`.
///
/// `cws` and `cw_np1` are shared by the 2 parties.
/// Only `s0s[0]` is different.
#[derive(Clone)]
pub struct Share<const OUT_BLEN: usize, G>
where
    G: Group<OUT_BLEN>,
{
    /// For the output of `gen`, its length is 2.
    /// For the input of `eval`, the first one is used.
    pub s0s: Vec<[u8; OUT_BLEN]>,
    /// The length of `cws` must be `n = 8 * N`.
    pub cws: Vec<Cw<OUT_BLEN, G>>,
    /// `$CW^{(n + 1)}$`.
    pub cw_np1: G,
}
