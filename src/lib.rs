// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Many variable names and the LaTeX math expressions in the doc comment are from the paper _Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation_.

#![cfg_attr(not(feature = "stable"), feature(portable_simd))]

use group::Group;

pub mod dcf;
pub mod dpf;
pub mod group;
#[cfg(feature = "prg")]
pub mod prg;
pub mod utils;

/// Point function.
/// Despite the name, it only ships an element of the input domain and an element of the output domain.
/// The actual meaning of the 2 elements is determined by the context.
///
/// - `IN_BLEN` is the **byte** length of the size of the input domain.
///   `$n$` or `$\lceil \log_2 |\mathbb{G}^{in}| \rceil$` (but the byte length).
/// - `OUT_BLEN` is the **byte** length of the size of the output domain.
///   `$\lambda$` or `$\lceil \log_2 |\mathbb{G}^{out}| \rceil$` (but the byte length).
pub struct PointFn<const IN_BLEN: usize, const OUT_BLEN: usize, G>
where
    G: Group<OUT_BLEN>,
{
    /// `$\alpha$`, or say `x` in `y = f(x)`.
    pub alpha: [u8; IN_BLEN],
    /// `$\beta$`, or say `y` in `y = f(x)`.
    pub beta: G,
}

/// Pseudorandom generator (PRG).
///
/// Requires `Sync` for multi-threading.
/// We still require it for single-threading since it should be still easy to be included.
pub trait Prg<const BLEN: usize, const BLEN_N: usize>: Sync {
    fn gen(&self, seed: &[u8; BLEN]) -> [([[u8; BLEN]; BLEN_N], bool); 2];
}

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
