//! Embedded PRGs.
//!
//! Including the ones from AES-series, combined with one-way compression functions.

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::{Aes128, Aes256};

use crate::owcf::{Hirose, MatyasMeyerOseas};
use crate::Prg;

/// AES128 with the Matyas-Meyer-Oseas single-block-length one-way compression function and precreated keys
pub struct Aes128MatyasMeyerOseasPrg {
    ciphers: Vec<Aes128MatyasMeyerOseasCipher>,
}

impl Aes128MatyasMeyerOseasPrg {
    pub fn new(keys: &[&[u8; 16]]) -> Self {
        Self {
            ciphers: keys
                .iter()
                .map(|&key| Aes128MatyasMeyerOseasCipher::new(key))
                .collect(),
        }
    }
}

impl Prg for Aes128MatyasMeyerOseasPrg {
    fn gen(&self, buf: &mut [u8], src: &[u8]) {
        assert_eq!(src.len() % 16, 0);
        assert_eq!(buf.len() % src.len(), 0);
        let scale = buf.len() / src.len();
        assert_eq!(self.ciphers.len(), scale * (src.len() / 16));
        for i in 0..scale {
            buf[i * src.len()..(i + 1) * src.len()]
                .array_chunks_mut::<16>()
                .zip(src.array_chunks::<16>())
                .zip(self.ciphers[i * (src.len() / 16)..(i + 1) * (src.len() / 16)].iter())
                .for_each(|((buf, src), cipher)| cipher.gen_blk(buf, src));
        }
    }
}

struct Aes128MatyasMeyerOseasCipher {
    cipher: Aes128,
}

impl Aes128MatyasMeyerOseasCipher {
    fn new(key: &[u8; 16]) -> Self {
        Self {
            cipher: Aes128::new(GenericArray::from_slice(key)),
        }
    }
}

impl MatyasMeyerOseas<16> for Aes128MatyasMeyerOseasCipher {
    fn enc_blk(&self, buf: &mut [u8; 16], input: &[u8; 16]) {
        let in_blk = GenericArray::from_slice(input);
        let buf_blk = GenericArray::from_mut_slice(buf);
        self.cipher.encrypt_block_b2b(in_blk, buf_blk)
    }
}

/// AES256 with the Hirose double-block-length one-way compression function and precreated keys
pub struct Aes256HirosePrg {
    ciphers: Vec<Aes256HiroseCipher>,
}

impl Aes256HirosePrg {
    pub fn new(keys: &[&[u8; 32]]) -> Self {
        Self {
            ciphers: keys
                .iter()
                .map(|&key| Aes256HiroseCipher::new(key))
                .collect(),
        }
    }
}

impl Prg for Aes256HirosePrg {
    fn gen(&self, buf: &mut [u8], src: &[u8]) {
        assert_eq!(src.len() % 16, 0);
        assert_eq!(buf.len() % (2 * src.len()), 0);
        let scale = buf.len() / (2 * src.len());
        assert_eq!(self.ciphers.len(), scale * (src.len() / 16));
        for i in 0..scale {
            buf[i * 2 * src.len()..(i + 1) * 2 * src.len()]
                .array_chunks_mut::<32>()
                .zip(src.array_chunks::<16>())
                .zip(self.ciphers[i * (src.len() / 16)..(i + 1) * (src.len() / 16)].iter())
                .for_each(|((buf, src), cipher)| {
                    let mut iter = buf.array_chunks_mut::<16>();
                    let buf1 = iter.next().unwrap();
                    let buf2 = iter.next().unwrap();
                    assert_eq!(iter.next(), None);
                    cipher.gen_blk([buf1, buf2], src);
                });
        }
    }
}

struct Aes256HiroseCipher {
    cipher: Aes256,
}

impl Aes256HiroseCipher {
    fn new(key: &[u8; 32]) -> Self {
        Self {
            cipher: Aes256::new(GenericArray::from_slice(key)),
        }
    }
}

impl Hirose<16> for Aes256HiroseCipher {
    fn enc_blk(&self, buf: &mut [u8; 16], input: &[u8; 16]) {
        let in_blk = GenericArray::from_slice(input);
        let buf_blk = GenericArray::from_mut_slice(buf);
        self.cipher.encrypt_block_b2b(in_blk, buf_blk)
    }
}
