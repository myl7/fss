// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! [`super::Prg`] integrated impl and [`crate::PrgBytes`] impl

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::{Aes128, Aes256};
use bitvec::prelude::*;

use super::Prg;
use crate::utils::{xor, xor_inplace};
use crate::PrgBytes;

/// Hirose double-block-length one-way compression function with AES256 and precreated keys.
/// Integrated impl of [`Prg`] with a good performance.
///
/// To avoid `#![feature(generic_const_exprs)]`, you MUST ensure `LAMBDA % 16 = 0` and `N = LAMBDA / 16`
///
/// It actually works for LAMBDA * 4 - 1 bits other than LAMBDA bytes.
/// The last bit of the output `[u8; LAMBDA]` is always set to 0.
pub struct Aes256HirosePrg<const LAMBDA: usize, const N: usize> {
    ciphers: [Aes256; N],
}

impl<const LAMBDA: usize, const N: usize> Aes256HirosePrg<LAMBDA, N> {
    pub fn new(keys: [&[u8; 32]; N]) -> Self {
        let ciphers = std::array::from_fn(|i| {
            let key_block = GenericArray::from_slice(keys[i]);
            Aes256::new(key_block)
        });
        Self { ciphers }
    }

    /// The arbitrary non-zero constant c for the arbitrary fixed-point-free permutation, typically just xor c
    const fn c() -> [u8; LAMBDA] {
        [0xff; LAMBDA]
    }
}

impl<const LAMBDA: usize, const N: usize> Prg<LAMBDA> for Aes256HirosePrg<LAMBDA, N> {
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], bool); 2] {
        // `$p(G_{i - 1})$`
        let seed_p = xor(&[seed, &Self::c()]);
        let mut result_buf0 = [[0; LAMBDA]; 2];
        let mut out_blocks = [GenericArray::default(); 2];
        (0..LAMBDA / 16).for_each(|j| {
            let in_block0 = GenericArray::from_slice(&seed[j * 16..(j + 1) * 16]);
            let in_block1 = GenericArray::from_slice(&seed_p[j * 16..(j + 1) * 16]);
            self.ciphers[j]
                .encrypt_blocks_b2b(&[*in_block0, *in_block1], &mut out_blocks)
                .unwrap();
            result_buf0[0][j * 16..(j + 1) * 16].copy_from_slice(out_blocks[0].as_ref());
            result_buf0[1][j * 16..(j + 1) * 16].copy_from_slice(out_blocks[1].as_ref());
        });
        xor_inplace(&mut result_buf0[0], &[seed]);
        xor_inplace(&mut result_buf0[1], &[&seed_p]);
        let bit0 = result_buf0[0].view_bits::<Lsb0>()[0];
        let bit1 = result_buf0[1].view_bits::<Lsb0>()[0];
        result_buf0
            .iter_mut()
            .for_each(|buf| buf[LAMBDA - 1].view_bits_mut::<Lsb0>().set(0, false));
        [(result_buf0[0], bit0), (result_buf0[1], bit1)]
    }
}

/// Matyas-Meyer-Oseas single-block-length one-way compression function with AES128 and precreated keys.
/// Integrated impl of [`Prg`].
///
/// To avoid `#![feature(generic_const_exprs)]`, you MUST ensure `LAMBDA % 16 = 0` and `N = 2 * (LAMBDA / 16)`
///
/// It actually works for LAMBDA * 4 - 1 bits other than LAMBDA bytes.
/// The last bit of the output `[u8; LAMBDA]` is always set to 0.
pub struct Aes128MatyasMeyerOseasPrg<const LAMBDA: usize, const N: usize> {
    ciphers: [Aes128; N],
}

impl<const LAMBDA: usize, const N: usize> Aes128MatyasMeyerOseasPrg<LAMBDA, N> {
    pub fn new(keys: [&[u8; 16]; N]) -> Self {
        let ciphers = std::array::from_fn(|i| {
            let key_block = GenericArray::from_slice(keys[i]);
            Aes128::new(key_block)
        });
        Self { ciphers }
    }
}

impl<const LAMBDA: usize, const N: usize> Prg<LAMBDA> for Aes128MatyasMeyerOseasPrg<LAMBDA, N> {
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], bool); 2] {
        let mut result_buf = [[0; LAMBDA]; 2];
        let mut out_block = GenericArray::default();
        (0..LAMBDA / 16).for_each(|j| {
            let in_block = GenericArray::from_slice(&seed[j * 16..(j + 1) * 16]);
            self.ciphers[j].encrypt_block_b2b(in_block, &mut out_block);
            result_buf[0][j * 16..(j + 1) * 16].copy_from_slice(out_block.as_ref());
            self.ciphers[j + LAMBDA / 16].encrypt_block_b2b(in_block, &mut out_block);
            result_buf[1][j * 16..(j + 1) * 16].copy_from_slice(out_block.as_ref());
        });
        xor_inplace(&mut result_buf[0], &[seed]);
        xor_inplace(&mut result_buf[1], &[seed]);
        let bit0 = result_buf[0].view_bits::<Lsb0>()[0];
        let bit1 = result_buf[1].view_bits::<Lsb0>()[0];
        result_buf
            .iter_mut()
            .for_each(|buf| buf[LAMBDA - 1].view_bits_mut::<Lsb0>().set(0, false));
        [(result_buf[0], bit0), (result_buf[1], bit1)]
    }
}

impl<const LAMBDA: usize, P> Prg<LAMBDA> for P
where
    P: PrgBytes,
{
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], bool); 2] {
        let mut buf = vec![0; 2 * LAMBDA];
        PrgBytes::gen(self, &mut buf, seed);
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
    use crate::prg::Aes256HirosePrgBytes;

    const KEYS: [&[u8; 32]; 1] =
        [b"j9\x1b_\xb3X\xf33\xacW\x15\x1b\x0812K\xb3I\xb9\x90r\x1cN\xb5\xee9W\xd3\xbb@\xc6d"];
    const SEED: &[u8; 16] = b"*L\x8f%y\x12Z\x94*E\x8f$+NH\x19";

    #[test]
    fn test_prg_gen_not_zeros() {
        let prg = Aes256HirosePrg::<16, 1>::new(KEYS);
        let out = prg.gen(SEED);
        (0..2).for_each(|i| {
            assert_ne!(out[i].0, [0; 16]);
            assert_ne!(xor(&[&out[i].0, SEED]), [0; 16]);
        });
    }

    #[test]
    fn test_prg_bytes_gen_not_zeros() {
        let prg = Aes256HirosePrgBytes::new(&KEYS.iter().map(|&k| k).collect::<Vec<_>>());
        let out = Prg::gen(&prg, SEED);
        (0..2).for_each(|i| {
            assert_ne!(out[i].0, [0; 16]);
            assert_ne!(xor(&[&out[i].0, SEED]), [0; 16]);
        });
    }
}
