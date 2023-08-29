// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;
use std::ops::{Add, AddAssign};

use crate::utils::{xor, xor_inplace};

pub trait Group<const LAMBDA: usize>
where
    Self: Sized + Add<Output = Self> + AddAssign + Debug + Clone + PartialEq + Eq + Sync + Send,
{
    fn convert(y: [u8; LAMBDA]) -> Self;
    fn convert_ref(y: &[u8; LAMBDA]) -> Self {
        Self::convert(y.to_owned())
    }
    fn zero() -> Self;
    fn add_inverse(self) -> Self;
    fn add_inverse_if(self, t: bool) -> Self {
        if t {
            self.add_inverse()
        } else {
            self
        }
    }
}

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
