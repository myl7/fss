// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Fast PRG implementations based on one-way compression functions, AES, and precreated keys.

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::{Aes128, Aes256};
use bitvec::prelude::*;

use crate::utils::{xor, xor_inplace};
use crate::Prg;

/// Hirose double-block-length one-way compression function with AES256 and precreated keys.
///
/// To avoid `#![feature(generic_const_exprs)]`, you MUST ensure `OUT_BLEN % 16 = 0` and `CIPHER_N = (OUT_BLEN / 16) * OUT_BLEN_N`.
/// We append a prefix `OUT_` to the generic constants to emphasize the PRG output is used for the output domain.
///
/// It actually works for OUT_BLEN * 8 - 1 bits other than OUT_BLEN bytes.
/// The last bit of EVERY `[u8; OUT_BLEN]` of the output is always set to 0.
pub struct Aes256HirosePrg<const OUT_BLEN: usize, const OUT_BLEN_N: usize, const CIPHER_N: usize> {
    ciphers: [Aes256; CIPHER_N],
}

impl<const OUT_BLEN: usize, const OUT_BLEN_N: usize, const CIPHER_N: usize>
    Aes256HirosePrg<OUT_BLEN, OUT_BLEN_N, CIPHER_N>
{
    pub fn new(keys: &[&[u8; 32]; CIPHER_N]) -> Self {
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

impl<const OUT_BLEN: usize, const OUT_BLEN_N: usize, const CIPHER_N: usize>
    Prg<OUT_BLEN, OUT_BLEN_N> for Aes256HirosePrg<OUT_BLEN, OUT_BLEN_N, CIPHER_N>
{
    fn gen(&self, seed: &[u8; OUT_BLEN]) -> [([[u8; OUT_BLEN]; OUT_BLEN_N], bool); 2] {
        // `$p(G_{i - 1})$`.
        let seed_p = xor(&[seed, &Self::c()]);

        let mut res_bufs = [[[0; OUT_BLEN]; OUT_BLEN_N]; 2];
        let mut out_blocks = [GenericArray::default(); 2];
        (0..OUT_BLEN_N).for_each(|blen_i| {
            (0..OUT_BLEN / 16).for_each(|block_i| {
                // Same as j = 0..CIHPER_N.
                let cipher_i = blen_i * (OUT_BLEN / 16) + block_i;
                let in_block0 = GenericArray::from_slice(&seed[block_i * 16..(block_i + 1) * 16]);
                let in_block1 = GenericArray::from_slice(&seed_p[block_i * 16..(block_i + 1) * 16]);
                self.ciphers[cipher_i]
                    .encrypt_blocks_b2b(&[*in_block0, *in_block1], &mut out_blocks)
                    .unwrap();
                res_bufs[0][blen_i][block_i * 16..(block_i + 1) * 16]
                    .copy_from_slice(out_blocks[0].as_ref());
                res_bufs[1][blen_i][block_i * 16..(block_i + 1) * 16]
                    .copy_from_slice(out_blocks[1].as_ref());
            });
        });
        (0..OUT_BLEN_N).for_each(|k| {
            xor_inplace(&mut res_bufs[0][k], &[seed]);
            xor_inplace(&mut res_bufs[1][k], &[&seed_p]);
        });
        let bit0 = res_bufs[0][0].view_bits::<Lsb0>()[0];
        let bit1 = res_bufs[1][0].view_bits::<Lsb0>()[0];
        res_bufs.iter_mut().for_each(|bufs| {
            bufs.iter_mut()
                .for_each(|buf| buf[OUT_BLEN - 1].view_bits_mut::<Lsb0>().set(0, false))
        });
        [(res_bufs[0], bit0), (res_bufs[1], bit1)]
    }
}

/// Matyas-Meyer-Oseas single-block-length one-way compression function with AES128 and precreated keys.
///
/// To avoid `#![feature(generic_const_exprs)]`, you MUST ensure `OUT_BLEN % 16 = 0` and `CIPHER_N = (OUT_BLEN / 16) * OUT_BLEN_N * 2`.
/// We append a prefix `OUT_` to the generic constants to emphasize the PRG output is used for the output domain.
///
/// It actually works for OUT_BLEN * 8 - 1 bits other than OUT_BLEN bytes.
/// The last bit of EVERY `[u8; OUT_BLEN]` of the output is always set to 0.
pub struct Aes128MatyasMeyerOseasPrg<
    const OUT_BLEN: usize,
    const OUT_BLEN_N: usize,
    const CIPHER_N: usize,
> {
    ciphers: [Aes128; CIPHER_N],
}

impl<const OUT_BLEN: usize, const OUT_BLEN_N: usize, const CIPHER_N: usize>
    Aes128MatyasMeyerOseasPrg<OUT_BLEN, OUT_BLEN_N, CIPHER_N>
{
    pub fn new(keys: &[&[u8; 16]; CIPHER_N]) -> Self {
        let ciphers = std::array::from_fn(|i| {
            let key_block = GenericArray::from_slice(keys[i]);
            Aes128::new(key_block)
        });
        Self { ciphers }
    }
}

impl<const OUT_BLEN: usize, const OUT_BLEN_N: usize, const CIPHER_N: usize>
    Prg<OUT_BLEN, OUT_BLEN_N> for Aes128MatyasMeyerOseasPrg<OUT_BLEN, OUT_BLEN_N, CIPHER_N>
{
    fn gen(&self, seed: &[u8; OUT_BLEN]) -> [([[u8; OUT_BLEN]; OUT_BLEN_N], bool); 2] {
        let ciphers0 = &self.ciphers[0..CIPHER_N / 2];
        let ciphers1 = &self.ciphers[CIPHER_N / 2..];

        let mut res_bufs = [[[0; OUT_BLEN]; OUT_BLEN_N]; 2];
        let mut out_blocks = [GenericArray::default(); 2];
        (0..OUT_BLEN_N).for_each(|blen_i| {
            (0..OUT_BLEN / 16).for_each(|block_i| {
                // Same as j = 0..CIHPER_N / 2.
                let cipher_i = blen_i * (OUT_BLEN / 16) + block_i;
                let in_block0 = GenericArray::from_slice(&seed[block_i * 16..(block_i + 1) * 16]);
                let in_block1 = GenericArray::from_slice(&seed[block_i * 16..(block_i + 1) * 16]);
                // AES instruction-level parallelism does not give noticeable performance improvement according to our trials.
                ciphers0[cipher_i].encrypt_block_b2b(in_block0, &mut out_blocks[0]);
                ciphers1[cipher_i].encrypt_block_b2b(in_block1, &mut out_blocks[1]);
                res_bufs[0][blen_i][block_i * 16..(block_i + 1) * 16]
                    .copy_from_slice(out_blocks[0].as_ref());
                res_bufs[1][blen_i][block_i * 16..(block_i + 1) * 16]
                    .copy_from_slice(out_blocks[1].as_ref());
            });
        });
        (0..OUT_BLEN_N).for_each(|k| {
            xor_inplace(&mut res_bufs[0][k], &[seed]);
            xor_inplace(&mut res_bufs[1][k], &[seed]);
        });
        let bit0 = res_bufs[0][0].view_bits::<Lsb0>()[0];
        let bit1 = res_bufs[1][0].view_bits::<Lsb0>()[0];
        res_bufs.iter_mut().for_each(|bufs| {
            bufs.iter_mut()
                .for_each(|buf| buf[OUT_BLEN - 1].view_bits_mut::<Lsb0>().set(0, false))
        });
        [(res_bufs[0], bit0), (res_bufs[1], bit1)]
    }
}

#[cfg(test)]
mod tests {
    use arbtest::arbtest;

    use super::*;

    #[test]
    fn test_128_not_trivial() {
        arbtest(|u| {
            let keys: [[u8; 16]; 4] = u.arbitrary()?;
            let seed: [u8; 16] = u.arbitrary()?;
            let prg =
                Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
            let out = prg.gen(&seed);
            (0..2).for_each(|i| {
                (0..2).for_each(|j| {
                    assert_ne!(out[i].0[j], [0; 16]);
                    assert_ne!(xor(&[&out[i].0[j], &seed]), [0; 16]);
                    assert_ne!(xor(&[&out[i].0[j], &seed]), [0xff; 16]);
                });
            });
            Ok(())
        });
    }

    #[test]
    fn test_256_not_trivial() {
        arbtest(|u| {
            let keys: [[u8; 32]; 4] = u.arbitrary()?;
            let seed: [u8; 16] = u.arbitrary()?;
            let prg = Aes256HirosePrg::<16, 2, 2>::new(&std::array::from_fn(|i| &keys[i]));
            let out = prg.gen(&seed);
            (0..2).for_each(|i| {
                (0..2).for_each(|j| {
                    assert_ne!(out[i].0[j], [0; 16]);
                    assert_ne!(xor(&[&out[i].0[j], &seed]), [0; 16]);
                    assert_ne!(xor(&[&out[i].0[j], &seed]), [0xff; 16]);
                });
            });
            Ok(())
        });
    }
}
