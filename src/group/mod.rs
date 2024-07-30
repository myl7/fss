// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! See [Group].

use std::ops::{Add, AddAssign, Neg};

pub mod byte;
pub mod int;
pub mod int_prime;

/// Group (mathematics).
/// Which has:
///
/// - Associative operation.
/// - Identity element.
/// - Inverse element.
pub trait Group<const BLEN: usize>
where
    Self: Add<Output = Self>
        + AddAssign
        + Neg<Output = Self>
        + PartialEq
        + Eq
        + Clone
        + Sync
        + Send
        + From<[u8; BLEN]>,
{
    /// Identity element.
    ///
    /// E.g., 0 in the integer group.
    ///
    /// If the compiler cannot infer `BLEN` with this static method, you can use the fully qualified syntax like:
    ///
    /// ```
    /// use fss_rs::group::Group;
    /// use fss_rs::group::byte::ByteGroup;
    ///
    /// let e: ByteGroup<16> = Group::<16>::zero();
    /// ```
    fn zero() -> Self;

    /// Helper to get the inverse element if true.
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

#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_group_axioms {
        ($test:ident, $t_impl:ty, $blen:literal) => {
            #[test]
            fn $test() {
                arbtest::arbtest(|u| {
                    let a_bs: [u8; $blen] = u.arbitrary()?;
                    let a: $t_impl = a_bs.into();
                    let b_bs: [u8; $blen] = u.arbitrary()?;
                    let b: $t_impl = b_bs.into();
                    let c_bs: [u8; $blen] = u.arbitrary()?;
                    let c: $t_impl = c_bs.into();
                    let e: $t_impl = crate::group::Group::<$blen>::zero();
                    let a_inv = -a.clone();

                    let l = a.clone() + (b.clone() + c.clone());
                    let r = (a.clone() + b.clone()) + c.clone();
                    assert_eq!(l, r, "associativity");

                    let l0 = a.clone() + e.clone();
                    let r0 = a.clone();
                    assert_eq!(l0, r0, "identity element");
                    let l1 = e.clone() + a.clone();
                    let r1 = a.clone();
                    assert_eq!(l1, r1, "identity element");

                    let l0 = a.clone() + a_inv.clone();
                    let r0 = e.clone();
                    assert_eq!(l0, r0, "inverse element");
                    let l1 = a_inv.clone() + a.clone();
                    let r1 = e.clone();
                    assert_eq!(l1, r1, "inverse element");

                    Ok(())
                });
            }
        };
    }
}
