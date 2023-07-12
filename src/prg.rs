// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! [`crate::Prg`] implementations

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes256;
use bitvec::prelude::*;

use crate::utils::xor_inplace;
use crate::Prg;

/// Matyas-Meyer-Oseas one-way compression function with AES256 and precreated keys as an implementation of [`Prg`].
pub struct Aes256MatyasMeyerOseasPrg {
    ciphers: [Aes256; 5],
}

impl Aes256MatyasMeyerOseasPrg {
    pub fn new(keys: [&[u8; 32]; 5]) -> Self {
        let ciphers = std::array::from_fn(|i| {
            let key_block = GenericArray::from_slice(keys[i]);
            Aes256::new(key_block)
        });
        Self { ciphers }
    }
}

impl Prg<16> for Aes256MatyasMeyerOseasPrg {
    fn gen(&self, seed: &[u8; 16]) -> [([u8; 16], [u8; 16], bool); 2] {
        let rand_blocks: Vec<[u8; 16]> = self
            .ciphers
            .iter()
            .map(|cipher| {
                let mut block = GenericArray::clone_from_slice(seed);
                cipher.encrypt_block(&mut block);
                xor_inplace(&mut block.into(), &[seed]);
                block.into()
            })
            .collect();
        assert_eq!(rand_blocks.len(), 5);
        [
            (
                rand_blocks[0],
                rand_blocks[1],
                rand_blocks[4].view_bits::<Lsb0>()[0],
            ),
            (
                rand_blocks[2],
                rand_blocks[3],
                rand_blocks[4].view_bits::<Lsb0>()[1],
            ),
        ]
    }
}
