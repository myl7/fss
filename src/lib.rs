// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! See [`DCF`]

use std::marker::PhantomData;

use bitvec::prelude::*;

/// API of Distributed comparison function.
///
/// See [`CmpFn`] for `N` and `LAMBDA`.
pub trait DCF<const N: usize, const LAMBDA: usize, PRGImpl>
where
    PRGImpl: PRG<LAMBDA>,
{
    /// `s0s` is `$s^{(0)}_0$` and `$s^{(0)}_1$` which should be randomly sampled
    fn gen(f: &CmpFn<N, LAMBDA>, s0s: [&[u8; LAMBDA]; 2]) -> Share<LAMBDA>;

    /// `b` is the party. `false` is 0 and `true` is 1.
    fn eval(b: bool, k: &Share<LAMBDA>, x: &[u8; N]) -> [u8; LAMBDA];
}

/// Comparison function.
///
/// - `N` is the byte size of the domain.
/// - `LAMBDA` here is used as the **byte** size of the range, unlike the one in the paper.
pub struct CmpFn<const N: usize, const LAMBDA: usize> {
    /// `$\alpha$`
    pub alpha: [u8; N],
    /// `$\beta$`
    pub beta: [u8; LAMBDA],
}

/// Pseudorandom generator used in the algorithm.
///
/// `$\{0, 1\}^{\lambda} \rightarrow \{0, 1\}^{2(2\lambda + 1)}$`.
pub trait PRG<const LAMBDA: usize> {
    fn gen(seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], [u8; LAMBDA], bool); 2];
}

/// Implementation of [`DCF`]
pub struct DCFImpl<const N: usize, const LAMBDA: usize, PRGImpl>
where
    PRGImpl: PRG<LAMBDA>,
{
    _prg: PhantomData<PRGImpl>,
}

const IDX_L: usize = 0;
const IDX_R: usize = 1;

impl<const N: usize, const LAMBDA: usize, PRGImpl> DCF<N, LAMBDA, PRGImpl>
    for DCFImpl<N, LAMBDA, PRGImpl>
where
    PRGImpl: PRG<LAMBDA>,
{
    fn gen(f: &CmpFn<N, LAMBDA>, s0s: [&[u8; LAMBDA]; 2]) -> Share<LAMBDA> {
        // The bit size of `$\alpha$`
        let n = 8 * N;
        let mut v_alpha = [0; LAMBDA];
        let mut ts = Vec::<[bool; 2]>::with_capacity(n);
        // Set `$t^{(0)}_0$` and `$t^{(0)}_1$`
        ts.push([false, true]);
        let mut ss = Vec::<[[u8; LAMBDA]; 2]>::with_capacity(n);
        // Set `$s^{(1)}_0$` and `$s^{(1)}_1$`
        ss.push([s0s[0].to_owned(), s0s[1].to_owned()]);
        let mut cws = Vec::<CW<LAMBDA>>::with_capacity(n);
        for i in 1..n + 1 {
            let [(s0l, v0l, t0l), (s0r, v0r, t0r)] = PRGImpl::gen(&ss[i - 1][0]);
            let [(s1l, v1l, t1l), (s1r, v1r, t1r)] = PRGImpl::gen(&ss[i - 1][1]);
            let alpha_i = f.alpha.view_bits::<Lsb0>()[i - 1];
            let (keep, lose) = if alpha_i {
                (IDX_R, IDX_L)
            } else {
                (IDX_L, IDX_R)
            };
            let s_cw = xor(&[[&s0l, &s0r][lose], [&s1l, &s1r][lose]]);
            let mut v_cw = xor(&[[&v0l, &v0r][lose], [&v1l, &v1r][lose], &v_alpha]);
            if lose == IDX_L {
                xor_inplace(&mut v_cw, &[&f.beta]);
            }
            xor_inplace(
                &mut v_alpha,
                &[[&v0l, &v0r][keep], [&v1l, &v1r][keep], &v_cw],
            );
            let tl_cw = t0l ^ t1l ^ alpha_i ^ true;
            let tr_cw = t0r ^ t1r ^ alpha_i;
            let cw = CW {
                s: s_cw,
                v: v_cw,
                tl: tl_cw,
                tr: tr_cw,
            };
            cws.push(cw);
            ss.push([
                xor(&[
                    [&s0l, &s0r][keep],
                    if ts[i - 1][0] { &s_cw } else { &[0; LAMBDA] },
                ]),
                xor(&[
                    [&s1l, &s1r][keep],
                    if ts[i - 1][1] { &s_cw } else { &[0; LAMBDA] },
                ]),
            ]);
            ts.push([
                [t0l, t0r][keep] ^ (ts[i - 1][0] && [tl_cw, tr_cw][keep]),
                [t1l, t1r][keep] ^ (ts[i - 1][1] && [tl_cw, tr_cw][keep]),
            ]);
        }
        assert_eq!((ss.len(), ts.len(), cws.len()), (n, n, n));
        let cw_np1 = xor(&[&ss[n][0], &ss[n][1], &v_alpha]);
        Share {
            s0s: vec![s0s[0].to_owned(), s0s[1].to_owned()],
            cws,
            cw_np1,
        }
    }

    fn eval(b: bool, k: &Share<LAMBDA>, x: &[u8; N]) -> [u8; LAMBDA] {
        todo!()
    }
}

/// `CW`
pub struct CW<const LAMBDA: usize> {
    pub s: [u8; LAMBDA],
    pub v: [u8; LAMBDA],
    pub tl: bool,
    pub tr: bool,
}

/// `k`
pub struct Share<const LAMBDA: usize> {
    /// For the output of `gen`, its length is 2.
    /// For the input of `eval`, the first one is used.
    pub s0s: Vec<[u8; LAMBDA]>,
    /// The length of `cws` must be `n = 8 * N`
    pub cws: Vec<CW<LAMBDA>>,
    /// `$CW^{(n + 1)}$`
    pub cw_np1: [u8; LAMBDA],
}

fn xor<const LAMBDA: usize>(xs: &[&[u8; LAMBDA]]) -> [u8; LAMBDA] {
    let mut res = [0; LAMBDA];
    for i in 0..LAMBDA {
        for x in xs {
            res[i] ^= x[i];
        }
    }
    res
}

fn xor_inplace<const LAMBDA: usize>(lhs: &mut [u8; LAMBDA], rhss: &[&[u8; LAMBDA]]) {
    for i in 0..LAMBDA {
        for rhs in rhss {
            lhs[i] ^= rhs[i];
        }
    }
}
