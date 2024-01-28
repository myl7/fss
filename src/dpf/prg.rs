// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! Impl [`DpfPrg`] for [`Prg`]

use bitvec::prelude::*;

use super::Prg as DpfPrg;
use crate::Prg;

impl<const LAMBDA: usize, P> DpfPrg<LAMBDA> for P
where
    P: Prg,
{
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], bool); 2] {
        let mut buf = vec![0; 2 * LAMBDA];
        Prg::gen(self, &mut buf, seed);
        let mut iter = buf.into_iter().array_chunks::<LAMBDA>();
        let mut sl = iter.next().unwrap();
        let mut sr = iter.next().unwrap();
        assert_eq!(iter.next(), None);
        let tl = sl.view_bits::<Lsb0>()[0];
        sl[LAMBDA - 1].view_bits_mut::<Lsb0>().set(0, false);
        let tr = sr.view_bits::<Lsb0>()[0];
        sr[LAMBDA - 1].view_bits_mut::<Lsb0>().set(0, false);
        [(sl, tl), (sr, tr)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prg::Aes256HirosePrg;
    use crate::utils::xor;

    const KEYS: &[&[u8; 32]] =
        &[b"j9\x1b_\xb3X\xf33\xacW\x15\x1b\x0812K\xb3I\xb9\x90r\x1cN\xb5\xee9W\xd3\xbb@\xc6d"];
    const SEED: &[u8; 16] = b"*L\x8f%y\x12Z\x94*E\x8f$+NH\x19";

    #[test]
    fn test_prg_gen_not_zeros() {
        let prg = Aes256HirosePrg::new(KEYS);
        let out = DpfPrg::gen(&prg, SEED);
        (0..2).for_each(|i| {
            assert_ne!(out[i].0, [0; 16]);
            assert_ne!(xor(&[&out[i].0, SEED]), [0; 16]);
        });
    }
}
