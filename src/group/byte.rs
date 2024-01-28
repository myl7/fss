//! Group of bytes
//!
//! - Associative operation: Xor
//! - Identity element: All zero
//! - Inverse element: `x` itself

use std::ops::{Add, AddAssign};

use super::{Group, GroupEmbed};
use crate::utils::xor_inplace;

/// See [`self`]
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

impl<const LAMBDA: usize> GroupEmbed<LAMBDA> for ByteGroup<LAMBDA> {}

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
