// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! [`crate::Prg`] implementations

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes256;
use bitvec::prelude::*;

use crate::utils::{xor, xor_inplace};
use crate::Prg;

/// Hirose double-block-length one-way compression function with AES256 and precreated keys
/// as an implementation of [`Prg`].
///
/// To avoid `#![feature(generic_const_exprs)]`, it is **your responsibility**
/// to ensure `LAMBDA % 16 = 0` and `N = 2 * (LAMBDA / 16)`.
///
/// It actually works for LAMBDA * 8 - 1 bits other than LAMBDA bytes.
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

    /// Get the arbitrary non-zero constant c
    fn c() -> [u8; LAMBDA] {
        std::array::from_fn(|_| 0xff)
    }
}

impl<const LAMBDA: usize, const N: usize> Prg<LAMBDA> for Aes256HirosePrg<LAMBDA, N> {
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], [u8; LAMBDA], bool); 2] {
        // `$p(G_{i - 1})$`
        let seed_p = xor(&[seed, &Self::c()]);
        let mut result_buf0 = [[0; LAMBDA]; 2];
        let mut result_buf1 = [[0; LAMBDA]; 2];
        let mut out_blocks = [GenericArray::default(); 2];
        (0..2usize).zip(0..LAMBDA / 16).for_each(|(i, j)| {
            let in_block0 = GenericArray::from_slice(&seed[j * 16..(j + 1) * 16]);
            let in_block1 = GenericArray::from_slice(&seed_p[j * 16..(j + 1) * 16]);
            self.ciphers[i * 16 + j]
                .encrypt_blocks_b2b(&[*in_block0, *in_block1], &mut out_blocks)
                .unwrap();
            result_buf0[i][j * 16..(j + 1) * 16].copy_from_slice(out_blocks[0].as_ref());
            result_buf1[i][j * 16..(j + 1) * 16].copy_from_slice(out_blocks[1].as_ref());
            // TODO: Add a test for this
            assert_ne!(&result_buf0[i][j * 16..(j + 1) * 16], &[0; 16]);
        });
        result_buf0
            .iter_mut()
            .for_each(|buf| xor_inplace(buf, &[seed]));
        result_buf1
            .iter_mut()
            .for_each(|buf| xor_inplace(buf, &[&seed_p]));
        let bit0 = result_buf0[0].view_bits::<Lsb0>()[0];
        let bit1 = result_buf1[0].view_bits::<Lsb0>()[0];
        result_buf0
            .iter_mut()
            .chain(result_buf1.iter_mut())
            .for_each(|buf| buf[LAMBDA - 1].view_bits_mut::<Lsb0>().set(0, false));
        [
            (result_buf0[0], result_buf1[0], bit0),
            (result_buf0[1], result_buf1[1], bit1),
        ]
    }
}
