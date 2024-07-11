// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! See [`Group`].

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg};

pub mod byte;
pub mod int;
pub mod int_prime;

/// Group (mathematics) that can be converted from a byte array.
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
    /// Zero in the group.
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
/// We include it here and impl it for all PRG embedded in the crate for user convenience.
pub trait GroupEmbed<const BLEN: usize>
where
    Self: Group<BLEN> + Into<[u8; BLEN]>,
{
}
