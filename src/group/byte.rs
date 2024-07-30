// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Byte vectors as a group.
//!
//! - Associative operation: XOR.
//! - Identity element: All bits zero.
//! - Inverse element: `x` itself.

use std::ops::{Add, AddAssign, Neg};

use super::Group;
use crate::utils::xor_inplace;

/// See [`self`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGroup<const BLEN: usize>(pub [u8; BLEN]);

impl<const BLEN: usize> Add for ByteGroup<BLEN> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        xor_inplace(&mut self.0, &[&rhs.0]);
        self
    }
}

impl<const BLEN: usize> AddAssign for ByteGroup<BLEN> {
    fn add_assign(&mut self, rhs: Self) {
        xor_inplace(&mut self.0, &[&rhs.0])
    }
}

impl<const BLEN: usize> Neg for ByteGroup<BLEN> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}

impl<const BLEN: usize> Group<BLEN> for ByteGroup<BLEN> {
    fn zero() -> Self {
        ByteGroup([0; BLEN])
    }
}

impl<const BLEN: usize> From<[u8; BLEN]> for ByteGroup<BLEN> {
    fn from(value: [u8; BLEN]) -> Self {
        Self(value)
    }
}

impl<const BLEN: usize> From<ByteGroup<BLEN>> for [u8; BLEN] {
    fn from(value: ByteGroup<BLEN>) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_group_axioms;

    test_group_axioms!(test_group_axioms, ByteGroup<16>, 16);
}
