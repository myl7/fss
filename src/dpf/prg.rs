// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! [`super::Prg`] impl.

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::{Aes128, Aes256};
use bitvec::prelude::*;

use super::Prg;
use crate::utils::{xor, xor_inplace};

/// Hirose double-block-length one-way compression function with AES256 and precreated keys.
///
/// To avoid `#![feature(generic_const_exprs)]`, you MUST ensure `OUT_BLEN % 16 = 0` and `CIPHER_N = OUT_BLEN / 16`.
///
/// It actually works for OUT_BLEN * 8 - 1 bits other than OUT_BLEN bytes.
/// The last bit of the output `[u8; OUT_BLEN]` is always set to 0.
pub struct Aes256HirosePrg<const OUT_BLEN: usize, const CIPHER_N: usize> {
    ciphers: [Aes256; CIPHER_N],
}

impl<const OUT_BLEN: usize, const CIPHER_N: usize> Aes256HirosePrg<OUT_BLEN, CIPHER_N> {
    pub fn new(keys: [&[u8; 32]; CIPHER_N]) -> Self {
        let ciphers = std::array::from_fn(|i| {
            let key_block = GenericArray::from_slice(keys[i]);
            Aes256::new(key_block)
        });
        Self { ciphers }
    }

    /// The arbitrary non-zero constant `c` for the arbitrary fixed-point-free permutation,
    /// which is typically just XOR `c`.
    const fn c() -> [u8; OUT_BLEN] {
        [0xff; OUT_BLEN]
    }
}

impl<const OUT_BLEN: usize, const CIPHER_N: usize> Prg<OUT_BLEN>
    for Aes256HirosePrg<OUT_BLEN, CIPHER_N>
{
    fn gen(&self, seed: &[u8; OUT_BLEN]) -> [([u8; OUT_BLEN], bool); 2] {
        // `$p(G_{i - 1})$`.
        let seed_p = xor(&[seed, &Self::c()]);
        let mut result_buf0 = [[0; OUT_BLEN]; 2];
        let mut out_blocks = [GenericArray::default(); 2];
        (0..OUT_BLEN / 16).for_each(|j| {
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
            .for_each(|buf| buf[OUT_BLEN - 1].view_bits_mut::<Lsb0>().set(0, false));
        [(result_buf0[0], bit0), (result_buf0[1], bit1)]
    }
}

/// Matyas-Meyer-Oseas single-block-length one-way compression function with AES128 and precreated keys.
///
/// To avoid `#![feature(generic_const_exprs)]`, you MUST ensure `OUT_BLEN % 16 = 0` and `CIPHER_N = 2 * (OUT_BLEN / 16)`.
///
/// It actually works for OUT_BLEN * 8 - 1 bits other than OUT_BLEN bytes.
/// The last bit of the output `[u8; OUT_BLEN]` is always set to 0.
pub struct Aes128MatyasMeyerOseasPrg<const OUT_BLEN: usize, const CIPHER_N: usize> {
    ciphers: [Aes128; CIPHER_N],
}

impl<const OUT_BLEN: usize, const CIPHER_N: usize> Aes128MatyasMeyerOseasPrg<OUT_BLEN, CIPHER_N> {
    pub fn new(keys: [&[u8; 16]; CIPHER_N]) -> Self {
        let ciphers = std::array::from_fn(|i| {
            let key_block = GenericArray::from_slice(keys[i]);
            Aes128::new(key_block)
        });
        Self { ciphers }
    }
}

impl<const OUT_BLEN: usize, const CIPHER_N: usize> Prg<OUT_BLEN>
    for Aes128MatyasMeyerOseasPrg<OUT_BLEN, CIPHER_N>
{
    fn gen(&self, seed: &[u8; OUT_BLEN]) -> [([u8; OUT_BLEN], bool); 2] {
        let mut result_buf = [[0; OUT_BLEN]; 2];
        let mut out_block = GenericArray::default();
        (0..OUT_BLEN / 16).for_each(|j| {
            let in_block = GenericArray::from_slice(&seed[j * 16..(j + 1) * 16]);
            self.ciphers[j].encrypt_block_b2b(in_block, &mut out_block);
            result_buf[0][j * 16..(j + 1) * 16].copy_from_slice(out_block.as_ref());
            self.ciphers[j + OUT_BLEN / 16].encrypt_block_b2b(in_block, &mut out_block);
            result_buf[1][j * 16..(j + 1) * 16].copy_from_slice(out_block.as_ref());
        });
        xor_inplace(&mut result_buf[0], &[seed]);
        xor_inplace(&mut result_buf[1], &[seed]);
        let bit0 = result_buf[0].view_bits::<Lsb0>()[0];
        let bit1 = result_buf[1].view_bits::<Lsb0>()[0];
        result_buf
            .iter_mut()
            .for_each(|buf| buf[OUT_BLEN - 1].view_bits_mut::<Lsb0>().set(0, false));
        [(result_buf[0], bit0), (result_buf[1], bit1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
