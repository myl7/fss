// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Byte vectors as a group.
//!
//! - Associative operation: XOR.
//! - Identity element: All bits zero.
//! - Inverse element: `x` itself.

use std::ops::{Add, AddAssign};

use super::{Group, GroupEmbed};
use crate::utils::xor_inplace;

/// See [`self`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGroup<const OUT_BLEN: usize>(pub [u8; OUT_BLEN]);

impl<const OUT_BLEN: usize> Add for ByteGroup<OUT_BLEN> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        xor_inplace(&mut self.0, &[&rhs.0]);
        self
    }
}

impl<const OUT_BLEN: usize> AddAssign for ByteGroup<OUT_BLEN> {
    fn add_assign(&mut self, rhs: Self) {
        xor_inplace(&mut self.0, &[&rhs.0])
    }
}

impl<const OUT_BLEN: usize> Group<OUT_BLEN> for ByteGroup<OUT_BLEN> {
    fn zero() -> Self {
        ByteGroup([0; OUT_BLEN])
    }

    fn add_inverse(self) -> Self {
        self
    }
}

impl<const OUT_BLEN: usize> GroupEmbed<OUT_BLEN> for ByteGroup<OUT_BLEN> {}

impl<const OUT_BLEN: usize> From<[u8; OUT_BLEN]> for ByteGroup<OUT_BLEN> {
    fn from(value: [u8; OUT_BLEN]) -> Self {
        Self(value)
    }
}

impl<const OUT_BLEN: usize> From<ByteGroup<OUT_BLEN>> for [u8; OUT_BLEN] {
    fn from(value: ByteGroup<OUT_BLEN>) -> Self {
        value.0
    }
}
