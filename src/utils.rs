// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

pub fn xor<const LAMBDA: usize>(xs: &[&[u8; LAMBDA]]) -> [u8; LAMBDA] {
    let mut res = [0; LAMBDA];
    for i in 0..LAMBDA {
        for x in xs {
            res[i] ^= x[i];
        }
    }
    res
}

pub fn xor_inplace<const LAMBDA: usize>(lhs: &mut [u8; LAMBDA], rhss: &[&[u8; LAMBDA]]) {
    for i in 0..LAMBDA {
        for rhs in rhss {
            lhs[i] ^= rhs[i];
        }
    }
}
