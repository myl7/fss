// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

pub fn xor<const LAMBDA: usize>(xs: &[&[u8; LAMBDA]]) -> [u8; LAMBDA] {
    let mut res = [0; LAMBDA];
    xor_inplace(&mut res, xs);
    res
}

pub fn xor_inplace<const LAMBDA: usize>(lhs: &mut [u8; LAMBDA], rhss: &[&[u8; LAMBDA]]) {
    for i in 0..LAMBDA {
        for rhs in rhss {
            lhs[i] ^= rhs[i];
        }
    }
}
