// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

#![feature(portable_simd)]

use std::fmt::Debug;
use std::ops::{Add, AddAssign};

#[cfg(feature = "byte")]
pub mod byte;
#[cfg(feature = "int")]
pub mod int;
#[cfg(feature = "int-prime")]
pub mod int_prime;

/// Group (mathematics) that can be converted from a byte array
///
/// `Into<[u8; LAMBDA]>` is not used in any fss crate so not included.
/// But it is implemented by all groups in the submodules.
pub trait Group<const LAMBDA: usize>
where
    Self: Add<Output = Self>
        + AddAssign
        + PartialEq
        + Eq
        + Debug
        + Sized
        + Clone
        + Sync
        + Send
        + From<[u8; LAMBDA]>,
{
    /// Zero in the group
    fn zero() -> Self;

    /// Additive inverse in the group, e.g., `-x` for `x` in the integer group
    fn add_inverse(self) -> Self;
    /// Helper to get the additive inverse if true.
    /// Used for expressions like `$(-1)^n x$`, in which `t` can be computed from `n`.
    fn add_inverse_if(self, t: bool) -> Self {
        if t {
            self.add_inverse()
        } else {
            self
        }
    }
}
