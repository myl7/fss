// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

use std::simd::{u8x16, u8x32, u8x64};

pub fn xor<const BLEN: usize>(xs: &[&[u8; BLEN]]) -> [u8; BLEN] {
    let mut res = [0; BLEN];
    xor_inplace(&mut res, xs);
    res
}

pub fn xor_inplace<const BLEN: usize>(lhs: &mut [u8; BLEN], rhss: &[&[u8; BLEN]]) {
    rhss.iter().fold(lhs, |lhs, &rhs| {
        assert_eq!(lhs.len(), rhs.len());
        let mut i = 0;
        while i < BLEN {
            let left = BLEN - i;
            if left >= 64 {
                let lhs_simd = u8x64::from_slice(&lhs[i..i + 64]);
                let rhs_simd = u8x64::from_slice(&rhs[i..i + 64]);
                lhs[i..i + 64].copy_from_slice((lhs_simd ^ rhs_simd).as_array());
                i += 64;
            } else if left >= 32 {
                let lhs_simd = u8x32::from_slice(&lhs[i..i + 32]);
                let rhs_simd = u8x32::from_slice(&rhs[i..i + 32]);
                lhs[i..i + 32].copy_from_slice((lhs_simd ^ rhs_simd).as_array());
                i += 32;
            } else if left >= 16 {
                let lhs_simd = u8x16::from_slice(&lhs[i..i + 16]);
                let rhs_simd = u8x16::from_slice(&rhs[i..i + 16]);
                lhs[i..i + 16].copy_from_slice((lhs_simd ^ rhs_simd).as_array());
                i += 16;
            } else {
                // Since a AES block is 16 bytes, and we usually use AES to construct the PRG,
                // no need to specially handle the case where OUT_BLEN % 16 != 0.
                // So we just xor them one by one in case wired situations make the program enter here.
                lhs[i..]
                    .iter_mut()
                    .zip(&rhs[i..])
                    .for_each(|(l, r)| *l ^= r);
                break;
            }
        }
        lhs
    });
}
