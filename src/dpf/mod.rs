// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! See [`Dpf`]

use bitvec::prelude::*;
#[cfg(feature = "multi-thread")]
use rayon::prelude::*;

use crate::group::Group;
use crate::utils::{xor, xor_inplace};
pub use crate::PointFn;
use crate::{decl_prg_trait, Cw, Share};

#[cfg(feature = "prg")]
pub mod prg;

/// Distributed point function API
///
/// `PointFn` used here means `$f(x) = \beta$` iff. `$x = \alpha$`, otherwise `$f(x) = 0$`.
///
/// See [`PointFn`] for `N` and `LAMBDA`.
pub trait Dpf<const N: usize, const LAMBDA: usize, G>
where
    G: Group<LAMBDA>,
{
    /// `s0s` is `$s^{(0)}_0$` and `$s^{(0)}_1$` which should be randomly sampled
    fn gen(&self, f: &PointFn<N, LAMBDA, G>, s0s: [&[u8; LAMBDA]; 2]) -> Share<LAMBDA, G>;

    /// `b` is the party. `false` is 0 and `true` is 1.
    fn eval(&self, b: bool, k: &Share<LAMBDA, G>, xs: &[&[u8; N]], ys: &mut [&mut G]);

    /// Full domain eval.
    /// See [`Dpf::eval`] for `b`.
    /// The corresponding `xs` to `ys` is the big endian representation of `0..=u*::MAX`.
    fn full_eval(&self, b: bool, k: &Share<LAMBDA, G>, ys: &mut [&mut G]);
}

decl_prg_trait!(([u8; LAMBDA], bool));

/// [`Dpf`] impl
///
/// `$\alpha$` itself is not included, which means `$f(\alpha)$ = 0`.
pub struct DpfImpl<const N: usize, const LAMBDA: usize, P>
where
    P: Prg<LAMBDA>,
{
    prg: P,
}

impl<const N: usize, const LAMBDA: usize, P> DpfImpl<N, LAMBDA, P>
where
    P: Prg<LAMBDA>,
{
    pub fn new(prg: P) -> Self {
        Self { prg }
    }
}

const IDX_L: usize = 0;
const IDX_R: usize = 1;

impl<const N: usize, const LAMBDA: usize, P, G> Dpf<N, LAMBDA, G> for DpfImpl<N, LAMBDA, P>
where
    P: Prg<LAMBDA>,
    G: Group<LAMBDA>,
{
    fn gen(&self, f: &PointFn<N, LAMBDA, G>, s0s: [&[u8; LAMBDA]; 2]) -> Share<LAMBDA, G> {
        // The bit size of `$\alpha$`
        let n = 8 * N;
        // Set `$s^{(1)}_0$` and `$s^{(1)}_1$`
        let mut ss_prev = [s0s[0].to_owned(), s0s[1].to_owned()];
        // Set `$t^{(0)}_0$` and `$t^{(0)}_1$`
        let mut ts_prev = [false, true];
        let mut cws = Vec::<Cw<LAMBDA, G>>::with_capacity(n);
        for i in 0..n {
            // MSB is required since we index from high to low in arrays
            let alpha_i = f.alpha.view_bits::<Msb0>()[i];
            let [(s0l, t0l), (s0r, t0r)] = self.prg.gen(&ss_prev[0]);
            let [(s1l, t1l), (s1r, t1r)] = self.prg.gen(&ss_prev[1]);
            let (keep, lose) = if alpha_i {
                (IDX_R, IDX_L)
            } else {
                (IDX_L, IDX_R)
            };
            let s_cw = xor(&[[&s0l, &s0r][lose], [&s1l, &s1r][lose]]);
            let tl_cw = t0l ^ t1l ^ alpha_i ^ true;
            let tr_cw = t0r ^ t1r ^ alpha_i;
            let cw = Cw {
                s: s_cw,
                v: G::zero(),
                tl: tl_cw,
                tr: tr_cw,
            };
            cws.push(cw);
            ss_prev = [
                xor(&[
                    [&s0l, &s0r][keep],
                    if ts_prev[0] { &s_cw } else { &[0; LAMBDA] },
                ]),
                xor(&[
                    [&s1l, &s1r][keep],
                    if ts_prev[1] { &s_cw } else { &[0; LAMBDA] },
                ]),
            ];
            ts_prev = [
                [t0l, t0r][keep] ^ (ts_prev[0] & [tl_cw, tr_cw][keep]),
                [t1l, t1r][keep] ^ (ts_prev[1] & [tl_cw, tr_cw][keep]),
            ];
        }
        let cw_np1 =
            (f.beta.clone() + Into::<G>::into(ss_prev[0]).add_inverse() + ss_prev[1].into())
                .add_inverse_if(ts_prev[1]);
        Share {
            s0s: vec![s0s[0].to_owned(), s0s[1].to_owned()],
            cws,
            cw_np1,
        }
    }

    fn eval(&self, b: bool, k: &Share<LAMBDA, G>, xs: &[&[u8; N]], ys: &mut [&mut G]) {
        #[cfg(feature = "multi-thread")]
        self.eval_mt(b, k, xs, ys);
        #[cfg(not(feature = "multi-thread"))]
        self.eval_st(b, k, xs, ys);
    }

    fn full_eval(&self, b: bool, k: &Share<LAMBDA, G>, ys: &mut [&mut G]) {
        let n = k.cws.len();
        assert_eq!(n, N * 8);

        let s = k.s0s[0];
        let t = b;
        self.full_eval_layer(b, k, ys, 0, (s, t));
    }
}

impl<const N: usize, const LAMBDA: usize, P> DpfImpl<N, LAMBDA, P>
where
    P: Prg<LAMBDA>,
{
    /// Eval with single-threading.
    /// See [`Dpf::eval`].
    pub fn eval_st<G>(&self, b: bool, k: &Share<LAMBDA, G>, xs: &[&[u8; N]], ys: &mut [&mut G])
    where
        G: Group<LAMBDA>,
    {
        xs.iter()
            .zip(ys.iter_mut())
            .for_each(|(x, y)| self.eval_point(b, k, x, y));
    }

    #[cfg(feature = "multi-thread")]
    /// Eval with multi-threading.
    /// See [`Dpf::eval`].
    pub fn eval_mt<G>(&self, b: bool, k: &Share<LAMBDA, G>, xs: &[&[u8; N]], ys: &mut [&mut G])
    where
        G: Group<LAMBDA>,
    {
        xs.par_iter()
            .zip(ys.par_iter_mut())
            .for_each(|(x, y)| self.eval_point(b, k, x, y));
    }

    fn full_eval_layer<G>(
        &self,
        b: bool,
        k: &Share<LAMBDA, G>,
        ys: &mut [&mut G],
        layer_i: usize,
        (s, t): ([u8; LAMBDA], bool),
    ) where
        G: Group<LAMBDA>,
    {
        assert_eq!(ys.len(), 1 << (N * 8 - layer_i));
        if ys.len() == 1 {
            *ys[0] = (Into::<G>::into(s) + if t { k.cw_np1.clone() } else { G::zero() })
                .add_inverse_if(b);
            return;
        }

        let cw = &k.cws[layer_i];
        let [(mut sl, mut tl), (mut sr, mut tr)] = self.prg.gen(&s);
        xor_inplace(&mut sl, &[if t { &cw.s } else { &[0; LAMBDA] }]);
        xor_inplace(&mut sr, &[if t { &cw.s } else { &[0; LAMBDA] }]);
        tl ^= t & cw.tl;
        tr ^= t & cw.tr;

        let (ys_l, ys_r) = ys.split_at_mut(ys.len() / 2);
        #[cfg(feature = "multi-thread")]
        rayon::join(
            || self.full_eval_layer(b, k, ys_l, layer_i + 1, (sl, tl)),
            || self.full_eval_layer(b, k, ys_r, layer_i + 1, (sr, tr)),
        );
        #[cfg(not(feature = "multi-thread"))]
        {
            self.full_eval_layer(b, k, ys_l, layer_i + 1, (sl, tl));
            self.full_eval_layer(b, k, ys_r, layer_i + 1, (sr, tr));
        }
    }

    pub fn eval_point<G>(&self, b: bool, k: &Share<LAMBDA, G>, x: &[u8; N], y: &mut G)
    where
        G: Group<LAMBDA>,
    {
        let n = k.cws.len();
        assert_eq!(n, N * 8);
        let v = y;

        let mut s_prev = k.s0s[0];
        let mut t_prev = b;
        for i in 0..n {
            let cw = &k.cws[i];
            let [(mut sl, mut tl), (mut sr, mut tr)] = self.prg.gen(&s_prev);
            xor_inplace(&mut sl, &[if t_prev { &cw.s } else { &[0; LAMBDA] }]);
            xor_inplace(&mut sr, &[if t_prev { &cw.s } else { &[0; LAMBDA] }]);
            tl ^= t_prev & cw.tl;
            tr ^= t_prev & cw.tr;
            if x.view_bits::<Msb0>()[i] {
                s_prev = sr;
                t_prev = tr;
            } else {
                s_prev = sl;
                t_prev = tl;
            }
        }
        *v = (Into::<G>::into(s_prev) + if t_prev { k.cw_np1.clone() } else { G::zero() })
            .add_inverse_if(b);
    }
}

#[cfg(all(test, feature = "prg"))]
mod tests {
    use rand::prelude::*;

    use super::*;
    use crate::group::byte::ByteGroup;
    use crate::prg::Aes256HirosePrgBytes;

    const KEYS: &[&[u8; 32]] =
        &[b"j9\x1b_\xb3X\xf33\xacW\x15\x1b\x0812K\xb3I\xb9\x90r\x1cN\xb5\xee9W\xd3\xbb@\xc6d"];
    const ALPHAS: &[&[u8; 16]] = &[
        b"K\xa9W\xf5\xdd\x05\xe9\xfc?\x04\xf6\xfbUo\xa8C",
        b"\xc2GK\xda\xc6\xbb\x99\x98Fq\"f\xb7\x8csU",
        b"\xc2GK\xda\xc6\xbb\x99\x98Fq\"f\xb7\x8csV",
        b"\xc2GK\xda\xc6\xbb\x99\x98Fq\"f\xb7\x8csW",
        b"\xef\x96\x97\xd7\x8f\x8a\xa4AP\n\xb35\xb5k\xff\x97",
    ];
    const BETA: &[u8; 16] = b"\x03\x11\x97\x12C\x8a\xe9#\x81\xa8\xde\xa8\x8f \xc0\xbb";

    #[test]
    fn test_dpf_gen_then_eval() {
        let prg = Aes256HirosePrgBytes::new(KEYS);
        let dpf = DpfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = PointFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.clone().into(),
        };
        let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let mut ys0 = vec![ByteGroup::zero(); ALPHAS.len()];
        let mut ys1 = vec![ByteGroup::zero(); ALPHAS.len()];
        dpf.eval(false, &k0, ALPHAS, &mut ys0.iter_mut().collect::<Vec<_>>());
        dpf.eval(true, &k1, ALPHAS, &mut ys1.iter_mut().collect::<Vec<_>>());
        ys0.iter_mut()
            .zip(ys1.iter())
            .for_each(|(y0, y1)| *y0 += y1.clone());
        ys1 = vec![
            ByteGroup::zero(),
            ByteGroup::zero(),
            BETA.clone().into(),
            ByteGroup::zero(),
            ByteGroup::zero(),
        ];
        assert_eq!(ys0, ys1);
    }

    #[test]
    fn test_dpf_gen_then_eval_not_zeros() {
        let prg = Aes256HirosePrgBytes::new(KEYS);
        let dpf = DpfImpl::<16, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = PointFn {
            alpha: ALPHAS[2].to_owned(),
            beta: BETA.clone().into(),
        };
        let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let mut k1 = k.clone();
        k1.s0s = vec![k1.s0s[1]];
        let mut ys0 = vec![ByteGroup::zero(); ALPHAS.len()];
        let mut ys1 = vec![ByteGroup::zero(); ALPHAS.len()];
        dpf.eval(false, &k0, ALPHAS, &mut ys0.iter_mut().collect::<Vec<_>>());
        dpf.eval(true, &k1, ALPHAS, &mut ys1.iter_mut().collect::<Vec<_>>());
        assert_ne!(ys0[2], ByteGroup::zero());
        assert_ne!(ys1[2], ByteGroup::zero());
    }

    #[test]
    fn test_dpf_full_domain_eval() {
        let x: [u8; 2] = ALPHAS[2][..2].try_into().unwrap();
        let prg = Aes256HirosePrgBytes::new(KEYS);
        let dpf = DpfImpl::<2, 16, _>::new(prg);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = PointFn {
            alpha: x,
            beta: BETA.clone().into(),
        };
        let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let xs: Vec<_> = (0u16..=u16::MAX).map(|i| i.to_be_bytes()).collect();
        assert_eq!(xs.len(), 1 << (8 * 2));
        let xs0: Vec<_> = xs.iter().collect();
        let mut ys0 = vec![ByteGroup::zero(); 1 << (8 * 2)];
        let mut ys0_full = vec![ByteGroup::zero(); 1 << (8 * 2)];
        dpf.eval(false, &k0, &xs0, &mut ys0.iter_mut().collect::<Vec<_>>());
        dpf.full_eval(false, &k0, &mut ys0_full.iter_mut().collect::<Vec<_>>());
        for (y0, y0_full) in ys0.iter().zip(ys0_full.iter()) {
            assert_eq!(y0, y0_full);
        }
    }
}
