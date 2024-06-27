// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! See [`Dpf`].

use bitvec::prelude::*;
#[cfg(feature = "multi-thread")]
use rayon::prelude::*;

use crate::group::Group;
use crate::utils::{xor, xor_inplace};
pub use crate::PointFn;
use crate::{Cw, Prg, Share};

/// Distributed point function API.
///
/// `PointFn` used here means `$f(x) = \beta$` iff. `$x = \alpha$`, otherwise `$f(x) = 0$`.
///
/// See [`PointFn`] for `IN_BLEN` and `OUT_BLEN`.
/// See [`DpfImpl`] for the implementation.
pub trait Dpf<const IN_BLEN: usize, const OUT_BLEN: usize, G>
where
    G: Group<OUT_BLEN>,
{
    /// `s0s` is `$s^{(0)}_0$` and `$s^{(0)}_1$` which should be randomly sampled.
    fn gen(
        &self,
        f: &PointFn<IN_BLEN, OUT_BLEN, G>,
        s0s: [&[u8; OUT_BLEN]; 2],
    ) -> Share<OUT_BLEN, G>;

    /// `b` is the party. `false` is 0 and `true` is 1.
    fn eval(&self, b: bool, k: &Share<OUT_BLEN, G>, xs: &[&[u8; IN_BLEN]], ys: &mut [&mut G]);

    /// Full domain eval.
    /// See [`Dpf::eval`] for `b`.
    /// The corresponding `xs` to `ys` is the big endian representation of `0..=u*::MAX`.
    fn full_eval(&self, b: bool, k: &Share<OUT_BLEN, G>, ys: &mut [&mut G]);
}

/// [`Dpf`] impl.
pub struct DpfImpl<const IN_BLEN: usize, const OUT_BLEN: usize, P>
where
    P: Prg<OUT_BLEN, 1>,
{
    prg: P,
    filter_bitn: usize,
}

impl<const IN_BLEN: usize, const OUT_BLEN: usize, P> DpfImpl<IN_BLEN, OUT_BLEN, P>
where
    P: Prg<OUT_BLEN, 1>,
{
    pub fn new(prg: P) -> Self {
        Self {
            prg,
            filter_bitn: 8 * IN_BLEN,
        }
    }

    pub fn new_with_filter(prg: P, filter_bitn: usize) -> Self {
        assert!(filter_bitn <= 8 * IN_BLEN && filter_bitn > 1);
        Self { prg, filter_bitn }
    }
}

const IDX_L: usize = 0;
const IDX_R: usize = 1;

impl<const IN_BLEN: usize, const OUT_BLEN: usize, P, G> Dpf<IN_BLEN, OUT_BLEN, G>
    for DpfImpl<IN_BLEN, OUT_BLEN, P>
where
    P: Prg<OUT_BLEN, 1>,
    G: Group<OUT_BLEN>,
{
    fn gen(
        &self,
        f: &PointFn<IN_BLEN, OUT_BLEN, G>,
        s0s: [&[u8; OUT_BLEN]; 2],
    ) -> Share<OUT_BLEN, G> {
        // The bit size of `$\alpha$`.
        let n = self.filter_bitn;
        // Set `$s^{(1)}_0$` and `$s^{(1)}_1$`.
        let mut ss_prev = [s0s[0].to_owned(), s0s[1].to_owned()];
        // Set `$t^{(0)}_0$` and `$t^{(0)}_1$`.
        let mut ts_prev = [false, true];
        let mut cws = Vec::<Cw<OUT_BLEN, G>>::with_capacity(n);
        for i in 0..n {
            // MSB is required since we index from high to low in arrays.
            let alpha_i = f.alpha.view_bits::<Msb0>()[i];
            let [([s0l], t0l), ([s0r], t0r)] = self.prg.gen(&ss_prev[0]);
            let [([s1l], t1l), ([s1r], t1r)] = self.prg.gen(&ss_prev[1]);
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
                    if ts_prev[0] { &s_cw } else { &[0; OUT_BLEN] },
                ]),
                xor(&[
                    [&s1l, &s1r][keep],
                    if ts_prev[1] { &s_cw } else { &[0; OUT_BLEN] },
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

    fn eval(&self, b: bool, k: &Share<OUT_BLEN, G>, xs: &[&[u8; IN_BLEN]], ys: &mut [&mut G]) {
        #[cfg(feature = "multi-thread")]
        self.eval_mt(b, k, xs, ys);
        #[cfg(not(feature = "multi-thread"))]
        self.eval_st(b, k, xs, ys);
    }

    fn full_eval(&self, b: bool, k: &Share<OUT_BLEN, G>, ys: &mut [&mut G]) {
        let n = k.cws.len();
        assert_eq!(n, self.filter_bitn);

        let s = k.s0s[0];
        let t = b;
        self.full_eval_layer(b, k, ys, 0, (s, t));
    }
}

impl<const IN_BLEN: usize, const OUT_BLEN: usize, P> DpfImpl<IN_BLEN, OUT_BLEN, P>
where
    P: Prg<OUT_BLEN, 1>,
{
    /// Eval with single-threading.
    /// See [`Dpf::eval`].
    pub fn eval_st<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        xs: &[&[u8; IN_BLEN]],
        ys: &mut [&mut G],
    ) where
        G: Group<OUT_BLEN>,
    {
        xs.iter()
            .zip(ys.iter_mut())
            .for_each(|(x, y)| self.eval_point(b, k, x, y));
    }

    #[cfg(feature = "multi-thread")]
    /// Eval with multi-threading.
    /// See [`Dpf::eval`].
    pub fn eval_mt<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        xs: &[&[u8; IN_BLEN]],
        ys: &mut [&mut G],
    ) where
        G: Group<OUT_BLEN>,
    {
        xs.par_iter()
            .zip(ys.par_iter_mut())
            .for_each(|(x, y)| self.eval_point(b, k, x, y));
    }

    fn full_eval_layer<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        ys: &mut [&mut G],
        layer_i: usize,
        (s, t): ([u8; OUT_BLEN], bool),
    ) where
        G: Group<OUT_BLEN>,
    {
        assert_eq!(ys.len(), 1 << (self.filter_bitn - layer_i));
        if ys.len() == 1 {
            *ys[0] = (Into::<G>::into(s) + if t { k.cw_np1.clone() } else { G::zero() })
                .add_inverse_if(b);
            return;
        }

        let cw = &k.cws[layer_i];
        let [([mut sl], mut tl), ([mut sr], mut tr)] = self.prg.gen(&s);
        xor_inplace(&mut sl, &[if t { &cw.s } else { &[0; OUT_BLEN] }]);
        xor_inplace(&mut sr, &[if t { &cw.s } else { &[0; OUT_BLEN] }]);
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

    pub fn eval_point<G>(&self, b: bool, k: &Share<OUT_BLEN, G>, x: &[u8; IN_BLEN], y: &mut G)
    where
        G: Group<OUT_BLEN>,
    {
        let n = k.cws.len();
        assert_eq!(n, self.filter_bitn);
        let v = y;

        let mut s_prev = k.s0s[0];
        let mut t_prev = b;
        for i in 0..n {
            let cw = &k.cws[i];
            let [([mut sl], mut tl), ([mut sr], mut tr)] = self.prg.gen(&s_prev);
            xor_inplace(&mut sl, &[if t_prev { &cw.s } else { &[0; OUT_BLEN] }]);
            xor_inplace(&mut sr, &[if t_prev { &cw.s } else { &[0; OUT_BLEN] }]);
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
    use crate::prg::Aes256HirosePrg;

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
        let prg = Aes256HirosePrg::<16, 1, 1>::new(std::array::from_fn(|i| KEYS[i]));
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
    fn test_dpf_gen_then_eval_with_filter() {
        let prg = Aes256HirosePrg::<16, 1, 1>::new(std::array::from_fn(|i| KEYS[i]));
        let dpf = DpfImpl::<16, 16, _>::new_with_filter(prg, 127);
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
            BETA.clone().into(),
            ByteGroup::zero(),
        ];
        assert_eq!(ys0, ys1);
    }

    #[test]
    fn test_dpf_gen_then_eval_not_zeros() {
        let prg = Aes256HirosePrg::<16, 1, 1>::new(std::array::from_fn(|i| KEYS[i]));
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
    fn test_dpf_full_eval() {
        let x: [u8; 2] = ALPHAS[2][..2].try_into().unwrap();
        let prg = Aes256HirosePrg::<16, 1, 1>::new(std::array::from_fn(|i| KEYS[i]));
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

    #[test]
    fn test_dpf_full_eval_with_filter() {
        let x: [u8; 2] = ALPHAS[2][..2].try_into().unwrap();
        let prg = Aes256HirosePrg::<16, 1, 1>::new(std::array::from_fn(|i| KEYS[i]));
        let dpf = DpfImpl::<2, 16, _>::new_with_filter(prg, 15);
        let s0s: [[u8; 16]; 2] = thread_rng().gen();
        let f = PointFn {
            alpha: x,
            beta: BETA.clone().into(),
        };
        let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = k.clone();
        k0.s0s = vec![k0.s0s[0]];
        let xs: Vec<_> = (0u16..=u16::MAX >> 1)
            .map(|i| (i << 1).to_be_bytes())
            .collect();
        assert_eq!(xs.len(), 1 << 15);
        let xs0: Vec<_> = xs.iter().collect();
        let mut ys0 = vec![ByteGroup::zero(); 1 << 15];
        let mut ys0_full = vec![ByteGroup::zero(); 1 << 15];
        dpf.eval(false, &k0, &xs0, &mut ys0.iter_mut().collect::<Vec<_>>());
        dpf.full_eval(false, &k0, &mut ys0_full.iter_mut().collect::<Vec<_>>());
        for (y0, y0_full) in ys0.iter().zip(ys0_full.iter()) {
            assert_eq!(y0, y0_full);
        }
    }
}
