//! Group of bytes which defines addition as XOR

use std::ops::{Add, AddAssign};

use crate::Group;
use utils::xor_inplace;

/// See [`crate::byte`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGroup<const LAMBDA: usize>(pub [u8; LAMBDA]);

impl<const LAMBDA: usize> Add for ByteGroup<LAMBDA> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        xor_inplace(&mut self.0, &[&rhs.0]);
        self
    }
}

impl<const LAMBDA: usize> AddAssign for ByteGroup<LAMBDA> {
    fn add_assign(&mut self, rhs: Self) {
        xor_inplace(&mut self.0, &[&rhs.0])
    }
}

impl<const LAMBDA: usize> Group<LAMBDA> for ByteGroup<LAMBDA> {
    fn zero() -> Self {
        ByteGroup([0; LAMBDA])
    }

    fn add_inverse(self) -> Self {
        self
    }
}

impl<const LAMBDA: usize> From<[u8; LAMBDA]> for ByteGroup<LAMBDA> {
    fn from(value: [u8; LAMBDA]) -> Self {
        Self(value)
    }
}

impl<const LAMBDA: usize> From<ByteGroup<LAMBDA>> for [u8; LAMBDA] {
    fn from(value: ByteGroup<LAMBDA>) -> Self {
        value.0
    }
}

mod utils {
    // pub fn xor<const LAMBDA: usize>(xs: &[&[u8; LAMBDA]]) -> [u8; LAMBDA] {
    //     let mut res = [0; LAMBDA];
    //     xor_inplace(&mut res, xs);
    //     res
    // }

    pub fn xor_inplace<const LAMBDA: usize>(lhs: &mut [u8; LAMBDA], rhss: &[&[u8; LAMBDA]]) {
        for i in 0..LAMBDA {
            for rhs in rhss {
                lhs[i] ^= rhs[i];
            }
        }
    }
}
