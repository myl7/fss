// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! See [`Dcf`]

use bitvec::prelude::*;
#[cfg(feature = "multi-thread")]
use rayon::prelude::*;

use crate::group::Group;
use crate::utils::{xor, xor_inplace};
use crate::{decl_prg_trait, Cw, PointFn, Share};

#[cfg(feature = "prg")]
pub mod prg;

/// Distributed comparison function API
///
/// See [`CmpFn`] for `N` and `LAMBDA`.
pub trait Dcf<const N: usize, const LAMBDA: usize, G>
where
    G: Group<LAMBDA>,
{
    /// `s0s` is `$s^{(0)}_0$` and `$s^{(0)}_1$` which should be randomly sampled
    fn gen(&self, f: &CmpFn<N, LAMBDA, G>, s0s: [&[u8; LAMBDA]; 2]) -> Share<LAMBDA, G>;

    /// `b` is the party. `false` is 0 and `true` is 1.
    fn eval(&self, b: bool, k: &Share<LAMBDA, G>, xs: &[&[u8; N]], ys: &mut [&mut G]);
}

/// Comparison function.
///
/// See [`BoundState`] for the meaning for different `bound`.
/// See [`PointFn`] for `N`, `LAMBDA`, and fields `alpha` and `beta`.
pub struct CmpFn<const N: usize, const LAMBDA: usize, G>
where
    G: Group<LAMBDA>,
{
    /// `$\alpha$`
    pub alpha: [u8; N],
    /// `$\beta$`
    pub beta: G,
    /// See [`BoundState`]
    pub bound: BoundState,
}

impl<const N: usize, const LAMBDA: usize, G> CmpFn<N, LAMBDA, G>
where
    G: Group<LAMBDA>,
{
    pub fn from_point(point: PointFn<N, LAMBDA, G>, bound: BoundState) -> Self {
        Self {
            alpha: point.alpha,
            beta: point.beta,
            bound,
        }
    }
}

decl_prg_trait!(([u8; LAMBDA], [u8; LAMBDA], bool));

/// [`Dcf`] impl
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

impl<const N: usize, const LAMBDA: usize, PrgT, G> Dcf<N, LAMBDA, G> for DcfImpl<N, LAMBDA, PrgT>
where
    PrgT: Prg<LAMBDA>,
    G: Group<LAMBDA>,
{
    fn gen(&self, f: &CmpFn<N, LAMBDA, G>, s0s: [&[u8; LAMBDA]; 2]) -> Share<LAMBDA, G> {
        // The bit size of `$\alpha$`
        let n = 8 * N;
        let mut v_alpha = G::zero();
        let mut ss = Vec::<[[u8; LAMBDA]; 2]>::with_capacity(n + 1);
        // Set `$s^{(1)}_0$` and `$s^{(1)}_1$`
        ss.push([s0s[0].to_owned(), s0s[1].to_owned()]);
        let mut ts = Vec::<[bool; 2]>::with_capacity(n + 1);
        // Set `$t^{(0)}_0$` and `$t^{(0)}_1$`
        ts.push([false, true]);
        let mut cws = Vec::<Cw<LAMBDA, G>>::with_capacity(n);
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
            let mut v_cw = (G::from(*[&v0l, &v0r][lose])
                + G::from(*[&v1l, &v1r][lose]).add_inverse()
                + v_alpha.clone().add_inverse())
            .add_inverse_if(ts[i - 1][1]);
            match f.bound {
                BoundState::LtBeta => {
                    if lose == IDX_L {
                        v_cw += f.beta.clone()
                    }
                }
                BoundState::GtBeta => {
                    if lose == IDX_R {
                        v_cw += f.beta.clone()
                    }
                }
            }
            v_alpha += G::from(*[&v0l, &v0r][keep]).add_inverse()
                + (*[&v1l, &v1r][keep]).into()
                + v_cw.clone().add_inverse_if(ts[i - 1][1]);
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
        let cw_np1 = (G::from(ss[n][1]) + G::from(ss[n][0]).add_inverse() + v_alpha.add_inverse())
            .add_inverse_if(ts[n][1]);
        Share {
            s0s: vec![s0s[0].to_owned(), s0s[1].to_owned()],
            cws,
            cw_np1,
        }
    }

    fn eval(&self, b: bool, k: &Share<LAMBDA, G>, xs: &[&[u8; N]], ys: &mut [&mut G]) {
        let n = k.cws.len();
        assert_eq!(n, N * 8);
        let f = |x: &[u8; N], v: &mut G| {
            let mut ss = Vec::<[u8; LAMBDA]>::with_capacity(n + 1);
            ss.push(k.s0s[0].to_owned());
            let mut ts = Vec::<bool>::with_capacity(n + 1);
            ts.push(b);
            *v = G::zero();
            for i in 1..n + 1 {
                let cw = &k.cws[i - 1];
                // `*_hat` before in-place xor
                let [(mut sl, vl_hat, mut tl), (mut sr, vr_hat, mut tr)] = self.prg.gen(&ss[i - 1]);
                xor_inplace(&mut sl, &[if ts[i - 1] { &cw.s } else { &[0; LAMBDA] }]);
                xor_inplace(&mut sr, &[if ts[i - 1] { &cw.s } else { &[0; LAMBDA] }]);
                tl ^= ts[i - 1] & cw.tl;
                tr ^= ts[i - 1] & cw.tr;
                if x.view_bits::<Msb0>()[i - 1] {
                    *v += (G::from(vr_hat) + if ts[i - 1] { cw.v.clone() } else { G::zero() })
                        .add_inverse_if(b);
                    ss.push(sr);
                    ts.push(tr);
                } else {
                    *v += (G::from(vl_hat) + if ts[i - 1] { cw.v.clone() } else { G::zero() })
                        .add_inverse_if(b);
                    ss.push(sl);
                    ts.push(tl);
                }
            }
            assert_eq!((ss.len(), ts.len()), (n + 1, n + 1));
            *v += (G::from(ss[n]) + if ts[n] { k.cw_np1.clone() } else { G::zero() })
                .add_inverse_if(b);
        };
        // TODO: Seperated entries
        #[cfg(feature = "multi-thread")]
        {
            xs.par_iter()
                .zip(ys.par_iter_mut())
                .for_each(|(x, y)| f(x, y));
        }
        #[cfg(not(feature = "multi-thread"))]
        {
            xs.iter().zip(ys.iter_mut()).for_each(|(x, y)| f(x, y));
        }
    }
}

pub enum BoundState {
    /// `$f(x) = \beta$` iff. `$x < \alpha$`, otherwise `$f(x) = 0$`
    ///
    /// This is the preference in the paper.
    LtBeta,
    /// `$f(x) = \beta$` iff. `$x > \alpha$`, otherwise `$f(x) = 0$`
    GtBeta,
}

#[cfg(all(test, feature = "prg"))]
mod tests {
    use rand::prelude::*;

    use super::*;
    use crate::group::byte::ByteGroup;
    use crate::prg::Aes256HirosePrgBytes;

    const KEYS: &[&[u8; 32]] = &[
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
    fn test_dcf_gen_then_eval() {
        let prg = Aes256HirosePrgBytes::new(KEYS);
        let dcf = DcfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = CmpFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.clone().into(),
            bound: BoundState::LtBeta,
        };
        let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let mut ys0 = vec![ByteGroup::zero(); ALPHAS.len()];
        let mut ys1 = vec![ByteGroup::zero(); ALPHAS.len()];
        dcf.eval(false, &k0, ALPHAS, &mut ys0.iter_mut().collect::<Vec<_>>());
        dcf.eval(true, &k1, ALPHAS, &mut ys1.iter_mut().collect::<Vec<_>>());
        ys0.iter_mut()
            .zip(ys1.iter())
            .for_each(|(y0, y1)| *y0 += y1.clone());
        ys1 = vec![
            BETA.clone().into(),
            BETA.clone().into(),
            ByteGroup::zero(),
            ByteGroup::zero(),
            ByteGroup::zero(),
        ];
        assert_eq!(ys0, ys1);
    }

    #[test]
    fn test_dcf_gen_gt_beta_then_eval() {
        let prg = Aes256HirosePrgBytes::new(KEYS);
        let dcf = DcfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = CmpFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.clone().into(),
            bound: BoundState::GtBeta,
        };
        let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let mut ys0 = vec![ByteGroup::zero(); ALPHAS.len()];
        let mut ys1 = vec![ByteGroup::zero(); ALPHAS.len()];
        dcf.eval(false, &k0, ALPHAS, &mut ys0.iter_mut().collect::<Vec<_>>());
        dcf.eval(true, &k1, ALPHAS, &mut ys1.iter_mut().collect::<Vec<_>>());
        ys0.iter_mut()
            .zip(ys1.iter())
            .for_each(|(y0, y1)| *y0 += y1.clone());
        ys1 = vec![
            ByteGroup::zero(),
            ByteGroup::zero(),
            ByteGroup::zero(),
            BETA.clone().into(),
            BETA.clone().into(),
        ];
        assert_eq!(ys0, ys1);
    }

    #[test]
    fn test_dcf_gen_then_eval_not_zeros() {
        let prg = Aes256HirosePrgBytes::new(KEYS);
        let dcf = DcfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = CmpFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.clone().into(),
            bound: BoundState::LtBeta,
        };
        let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let mut ys0 = vec![ByteGroup::zero(); ALPHAS.len()];
        let mut ys1 = vec![ByteGroup::zero(); ALPHAS.len()];
        dcf.eval(false, &k0, ALPHAS, &mut ys0.iter_mut().collect::<Vec<_>>());
        dcf.eval(true, &k1, ALPHAS, &mut ys1.iter_mut().collect::<Vec<_>>());
        assert_ne!(ys0[2], ByteGroup::zero());
        assert_ne!(ys1[2], ByteGroup::zero());
    }
}
