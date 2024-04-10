// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! [`crate::Prg`] implementations

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes256;
use bitvec::prelude::*;
use group::byte::utils::{xor, xor_inplace};

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
        (0..2usize).for_each(|i| {
            (0..LAMBDA / 16).for_each(|j| {
                let in_block0 = GenericArray::from_slice(&seed[j * 16..(j + 1) * 16]);
                let in_block1 = GenericArray::from_slice(&seed_p[j * 16..(j + 1) * 16]);
                self.ciphers[i * 16 + j]
                    .encrypt_blocks_b2b(&[*in_block0, *in_block1], &mut out_blocks)
                    .unwrap();
                result_buf0[i][j * 16..(j + 1) * 16].copy_from_slice(out_blocks[0].as_ref());
                result_buf1[i][j * 16..(j + 1) * 16].copy_from_slice(out_blocks[1].as_ref());
            });
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

#[cfg(test)]
mod tests {
    use super::*;

    const KEYS: [&[u8; 32]; 2] = [
        b"j9\x1b_\xb3X\xf33\xacW\x15\x1b\x0812K\xb3I\xb9\x90r\x1cN\xb5\xee9W\xd3\xbb@\xc6d",
        b"\x9b\x15\xc8\x0f\xb7\xbc!q\x9e\x89\xb8\xf7\x0e\xa0S\x9dN\xfa\x0c;\x16\xe4\x98\x82b\xfcdy\xb5\x8c{\xc2",
    ];
    const SEED: &[u8; 16] = b"*L\x8f%y\x12Z\x94*E\x8f$+NH\x19";

    #[test]
    fn test_prg_gen_not_zeros() {
        let prg = Aes256HirosePrg::<16, 2>::new(KEYS);
        let out = prg.gen(SEED);
        (0..2).for_each(|i| {
            assert_ne!(out[i].0, [0; 16]);
            assert_ne!(out[i].1, [0; 16]);
            assert_ne!(xor(&[&out[i].0, SEED]), [0; 16]);
            assert_ne!(xor(&[&out[i].1, SEED]), [0; 16]);
            assert_ne!(xor(&[&out[i].0, SEED]), [0xff; 16]);
            assert_ne!(xor(&[&out[i].1, SEED]), [0xff; 16]);
        });
    }
}
