// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Group (mathematics).
//!
//! See [`Group`].

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg};

pub mod byte;
pub mod int;
pub mod int_prime;

/// Group that can be converted from bytes.
pub trait Group<const BLEN: usize>
where
    Self: Add<Output = Self>
        + AddAssign
        + Neg<Output = Self>
        + PartialEq
        + Eq
        + Debug
        + Sized
        + Clone
        + Sync
        + Send
        + From<[u8; BLEN]>,
{
    /// Additive identity.
    ///
    /// E.g., 0 in the integer group.
    fn zero() -> Self;

    /// Helper to get the inverse if true.
    ///
    /// Used for expressions like `$(-1)^n x$`, in which `t` can be computed from `n`.
    fn neg_if(self, t: bool) -> Self {
        if t {
            -self
        } else {
            self
        }
    }
}

/// `Into<[u8; BLEN]>` is not used in the crate.
/// But it is implemented by all included PRGs for user convenience.
pub trait GroupToBytes<const BLEN: usize>: Into<[u8; BLEN]> {}
