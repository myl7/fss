// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

//! One-way compression function wrappers to make PRGs.
//! https://wikipedia.org/wiki/One-way_compression_function .
//!
//! Notice that we only do the 1st round, which uses a constant pre-specified initial key, due to the use cases in FSS and performance
//!
//! Requires `Sync` for multi-threading, which should be still easy for even single-threaded

use crate::utils::xor_inplace;

pub trait MatyasMeyerOseas<const BLK_SIZE: usize>: Sync {
    fn enc_blk(&self, buf: &mut [u8; BLK_SIZE], input: &[u8; BLK_SIZE]);
    fn gen_blk(&self, buf: &mut [u8; BLK_SIZE], input: &[u8; BLK_SIZE]) {
        self.enc_blk(buf, input);
        xor_inplace(buf, &[input]);
    }
}

pub trait Hirose<const BLK_SIZE: usize>: Sync {
    fn enc_blk(&self, buf: &mut [u8; BLK_SIZE], input: &[u8; BLK_SIZE]);
    /// Get the arbitrary non-zero constant c for the arbitrary fixed-point-free permutation, typically just xor c
    fn c(&self) -> &[u8; BLK_SIZE] {
        &[0xff; BLK_SIZE]
    }
    /// the arbitrary fixed-point-free permutation
    fn p(&self, buf: &mut [u8; BLK_SIZE]) {
        xor_inplace(buf, &[self.c()]);
    }
    /// `input` is unchanged, but used as a buffer and modified in the process
    fn gen_blk(&self, buf: [&mut [u8; BLK_SIZE]; 2], input: &[u8; BLK_SIZE]) {
        let [buf1, buf2] = buf;

        // Use `buf1` as a buffer
        buf1.copy_from_slice(input);
        self.p(buf1);
        self.enc_blk(buf2, buf1);
        xor_inplace(buf2, &[buf1]);

        self.enc_blk(buf1, input);
        xor_inplace(buf1, &[input]);
    }
}
