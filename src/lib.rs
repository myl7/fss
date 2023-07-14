// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! See [`Dcf`]

#[cfg(feature = "prg")]
pub mod prg;

mod utils;

use bitvec::prelude::*;

use crate::utils::{xor, xor_inplace};

/// API of Distributed comparison function.
///
/// See [`CmpFn`] for `N` and `LAMBDA`.
pub trait Dcf<const N: usize, const LAMBDA: usize> {
    /// `s0s` is `$s^{(0)}_0$` and `$s^{(0)}_1$` which should be randomly sampled
    fn gen(
        &self,
        f: &CmpFn<N, LAMBDA>,
        s0s: [&[u8; LAMBDA]; 2],
        bound: BoundState,
    ) -> Share<LAMBDA>;

    /// `b` is the party. `false` is 0 and `true` is 1.
    fn eval(&self, b: bool, k: &Share<LAMBDA>, x: &[u8; N]) -> [u8; LAMBDA];
}

/// Comparison function.
///
/// - `N` is the **byte** size of the domain.
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
pub trait Prg<const LAMBDA: usize> {
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], [u8; LAMBDA], bool); 2];
}

/// Implementation of [`Dcf`].
///
/// `$\alpha$` itself is not included, which means `$f(\alpha)$ = 0`.
pub struct DcfImpl<const N: usize, const LAMBDA: usize, PrgT>
where
    PrgT: Prg<LAMBDA>,
{
    prg: PrgT,
}

impl<const N: usize, const LAMBDA: usize, PrgT> DcfImpl<N, LAMBDA, PrgT>
where
    PrgT: Prg<LAMBDA>,
{
    pub fn new(prg: PrgT) -> Self {
        Self { prg }
    }
}

const IDX_L: usize = 0;
const IDX_R: usize = 1;

impl<const N: usize, const LAMBDA: usize, PrgT> Dcf<N, LAMBDA> for DcfImpl<N, LAMBDA, PrgT>
where
    PrgT: Prg<LAMBDA>,
{
    fn gen(
        &self,
        f: &CmpFn<N, LAMBDA>,
        s0s: [&[u8; LAMBDA]; 2],
        bound: BoundState,
    ) -> Share<LAMBDA> {
        // The bit size of `$\alpha$`
        let n = 8 * N;
        let mut v_alpha = [0; LAMBDA];
        let mut ss = Vec::<[[u8; LAMBDA]; 2]>::with_capacity(n + 1);
        // Set `$s^{(1)}_0$` and `$s^{(1)}_1$`
        ss.push([s0s[0].to_owned(), s0s[1].to_owned()]);
        let mut ts = Vec::<[bool; 2]>::with_capacity(n + 1);
        // Set `$t^{(0)}_0$` and `$t^{(0)}_1$`
        ts.push([false, true]);
        let mut cws = Vec::<Cw<LAMBDA>>::with_capacity(n);
        for i in 1..n + 1 {
            let [(s0l, v0l, t0l), (s0r, v0r, t0r)] = self.prg.gen(&ss[i - 1][0]);
            let [(s1l, v1l, t1l), (s1r, v1r, t1r)] = self.prg.gen(&ss[i - 1][1]);
            // MSB is required since we index from high to low in arrays
            let alpha_i = f.alpha.view_bits::<Msb0>()[i - 1];
            let (keep, lose) = if alpha_i {
                (IDX_R, IDX_L)
            } else {
                (IDX_L, IDX_R)
            };
            let s_cw = xor(&[[&s0l, &s0r][lose], [&s1l, &s1r][lose]]);
            let mut v_cw = xor(&[[&v0l, &v0r][lose], [&v1l, &v1r][lose], &v_alpha]);
            match bound {
                BoundState::LtBeta => {
                    if lose == IDX_L {
                        xor_inplace(&mut v_cw, &[&f.beta]);
                    }
                }
                BoundState::GtBeta => {
                    if lose == IDX_R {
                        xor_inplace(&mut v_cw, &[&f.beta]);
                    }
                }
            }
            xor_inplace(
                &mut v_alpha,
                &[[&v0l, &v0r][keep], [&v1l, &v1r][keep], &v_cw],
            );
            let tl_cw = t0l ^ t1l ^ alpha_i ^ true;
            let tr_cw = t0r ^ t1r ^ alpha_i;
            let cw = Cw {
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
                [t0l, t0r][keep] ^ (ts[i - 1][0] & [tl_cw, tr_cw][keep]),
                [t1l, t1r][keep] ^ (ts[i - 1][1] & [tl_cw, tr_cw][keep]),
            ]);
        }
        assert_eq!((ss.len(), ts.len(), cws.len()), (n + 1, n + 1, n));
        let cw_np1 = xor(&[&ss[n][0], &ss[n][1], &v_alpha]);
        Share {
            s0s: vec![s0s[0].to_owned(), s0s[1].to_owned()],
            cws,
            cw_np1,
        }
    }

    fn eval(&self, b: bool, k: &Share<LAMBDA>, x: &[u8; N]) -> [u8; LAMBDA] {
        let n = k.cws.len();
        assert_eq!(n, N * 8);
        let mut ss = Vec::<[u8; LAMBDA]>::with_capacity(n + 1);
        ss.push(k.s0s[0].to_owned());
        let mut ts = Vec::<bool>::with_capacity(n + 1);
        ts.push(b);
        let mut v = [0; LAMBDA];
        for i in 1..n + 1 {
            let cw = &k.cws[i - 1];
            // `*_hat` before in-place xor
            let [(mut sl, vl_hat, mut tl), (mut sr, vr_hat, mut tr)] = self.prg.gen(&ss[i - 1]);
            xor_inplace(&mut sl, &[if ts[i - 1] { &cw.s } else { &[0; LAMBDA] }]);
            xor_inplace(&mut sr, &[if ts[i - 1] { &cw.s } else { &[0; LAMBDA] }]);
            tl ^= ts[i - 1] & cw.tl;
            tr ^= ts[i - 1] & cw.tr;
            if x.view_bits::<Msb0>()[i - 1] {
                xor_inplace(
                    &mut v,
                    &[&vr_hat, if ts[i - 1] { &cw.v } else { &[0; LAMBDA] }],
                );
                ss.push(sr);
                ts.push(tr);
            } else {
                xor_inplace(
                    &mut v,
                    &[&vl_hat, if ts[i - 1] { &cw.v } else { &[0; LAMBDA] }],
                );
                ss.push(sl);
                ts.push(tl);
            }
        }
        assert_eq!((ss.len(), ts.len()), (n + 1, n + 1));
        xor_inplace(
            &mut v,
            &[&ss[n], if ts[n] { &k.cw_np1 } else { &[0; LAMBDA] }],
        );
        v
    }
}

/// `Cw`. Correclation word.
#[derive(Clone)]
pub struct Cw<const LAMBDA: usize> {
    pub s: [u8; LAMBDA],
    pub v: [u8; LAMBDA],
    pub tl: bool,
    pub tr: bool,
}

/// `k`.
///
/// `cws` and `cw_np1` is shared by the 2 parties.
/// Only `s0s[0]` is different.
#[derive(Clone)]
pub struct Share<const LAMBDA: usize> {
    /// For the output of `gen`, its length is 2.
    /// For the input of `eval`, the first one is used.
    pub s0s: Vec<[u8; LAMBDA]>,
    /// The length of `cws` must be `n = 8 * N`
    pub cws: Vec<Cw<LAMBDA>>,
    /// `$CW^{(n + 1)}$`
    pub cw_np1: [u8; LAMBDA],
}

pub enum BoundState {
    /// `$f(x) = \beta$` iff. `$x < \alpha$`.
    ///
    /// This is the preference in the paper.
    LtBeta,
    /// `$f(x) = \beta$` iff. `$x > \alpha$`
    GtBeta,
}

#[cfg(all(test, feature = "prg"))]
mod tests {
    use super::*;

    use rand::{thread_rng, Rng};

    use crate::prg::Aes256HirosePrg;

    const KEYS: [&[u8; 32]; 2] = [
        b"j9\x1b_\xb3X\xf33\xacW\x15\x1b\x0812K\xb3I\xb9\x90r\x1cN\xb5\xee9W\xd3\xbb@\xc6d",
        b"\x9b\x15\xc8\x0f\xb7\xbc!q\x9e\x89\xb8\xf7\x0e\xa0S\x9dN\xfa\x0c;\x16\xe4\x98\x82b\xfcdy\xb5\x8c{\xc2",
    ];
    const ALPHAS: &[&[u8; 16]] = &[
        b"K\xa9W\xf5\xdd\x05\xe9\xfc?\x04\xf6\xfbUo\xa8C",
        b"\xc2GK\xda\xc6\xbb\x99\x98Fq\"f\xb7\x8csU",
        b"\xc2GK\xda\xc6\xbb\x99\x98Fq\"f\xb7\x8csV",
        b"\xc2GK\xda\xc6\xbb\x99\x98Fq\"f\xb7\x8csW",
        b"\xef\x96\x97\xd7\x8f\x8a\xa4AP\n\xb35\xb5k\xff\x97",
    ];
    const BETA: &[u8; 16] = b"\x03\x11\x97\x12C\x8a\xe9#\x81\xa8\xde\xa8\x8f \xc0\xbb";

    #[test]
    fn test_dcf_gen_then_eval_ok() {
        let prg = Aes256HirosePrg::new(KEYS);
        let dcf = DcfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = CmpFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.to_owned(),
        };
        let k = dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::LtBeta);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let y0 = dcf.eval(false, &k0, ALPHAS[0]);
        let y1 = dcf.eval(true, &k1, ALPHAS[0]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, BETA.to_owned());
        let y0 = dcf.eval(false, &k0, ALPHAS[1]);
        let y1 = dcf.eval(true, &k1, ALPHAS[1]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, BETA.to_owned());
        let y0 = dcf.eval(false, &k0, ALPHAS[2]);
        let y1 = dcf.eval(true, &k1, ALPHAS[2]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, [0; 16]);
        let y0 = dcf.eval(false, &k0, ALPHAS[3]);
        let y1 = dcf.eval(true, &k1, ALPHAS[3]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, [0; 16]);
        let y0 = dcf.eval(false, &k0, ALPHAS[4]);
        let y1 = dcf.eval(true, &k1, ALPHAS[4]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, [0; 16]);
    }

    #[test]
    fn test_dcf_gen_gt_beta_then_eval_ok() {
        let prg = Aes256HirosePrg::new(KEYS);
        let dcf = DcfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = CmpFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.to_owned(),
        };
        let k = dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::GtBeta);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let y0 = dcf.eval(false, &k0, ALPHAS[0]);
        let y1 = dcf.eval(true, &k1, ALPHAS[0]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, [0; 16]);
        let y0 = dcf.eval(false, &k0, ALPHAS[1]);
        let y1 = dcf.eval(true, &k1, ALPHAS[1]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, [0; 16]);
        let y0 = dcf.eval(false, &k0, ALPHAS[2]);
        let y1 = dcf.eval(true, &k1, ALPHAS[2]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, [0; 16]);
        let y0 = dcf.eval(false, &k0, ALPHAS[3]);
        let y1 = dcf.eval(true, &k1, ALPHAS[3]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, BETA.to_owned());
        let y0 = dcf.eval(false, &k0, ALPHAS[4]);
        let y1 = dcf.eval(true, &k1, ALPHAS[4]);
        let y = xor(&[&y0, &y1]);
        assert_eq!(y, BETA.to_owned());
    }
}
