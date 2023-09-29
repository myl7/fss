//! Group of bytes which defines addition as XOR

use std::ops::{Add, AddAssign};

use crate::Group;
use utils::{xor, xor_inplace};

/// See [`crate::byte`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGroup<const LAMBDA: usize>(pub [u8; LAMBDA]);

impl<const LAMBDA: usize> Add for ByteGroup<LAMBDA> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ByteGroup(xor(&[&self.0, &rhs.0]))
    }
}

impl<const LAMBDA: usize> AddAssign for ByteGroup<LAMBDA> {
    fn add_assign(&mut self, rhs: Self) {
        xor_inplace(&mut self.0, &[&rhs.0])
    }
}

impl<const LAMBDA: usize> Group<LAMBDA> for ByteGroup<LAMBDA> {
    fn convert(y: [u8; LAMBDA]) -> Self {
        ByteGroup(y)
    }

    fn zero() -> Self {
        ByteGroup([0; LAMBDA])
    }

    fn add_inverse(self) -> Self {
        self
    }
}

pub mod utils {
    //! Utilities, e.g., XOR

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
}
