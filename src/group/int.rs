// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Integers as a group.
//!
//! - Associative operation: Integer wrapping addition, `$(a + b) \mod 2^N$`.
//! - Identity element: 0.
//! - Inverse element: `-x`.
//!
//! # Security
//!
//! Such a group whose cardinality is not a prime number cannot provide the attribute that: if `a` and `b` are individually indistinguishable with random elements, `a * b` (integer multiplication) is still that.
//! If you need this attribute (e.g., for some verification), use [`crate::group::int_prime`] instead.

use std::mem::size_of;
use std::ops::{Add, AddAssign};

use super::{Group, GroupEmbed};

macro_rules! decl_int_group {
    ($t:ty, $t_impl:ident) => {
        /// See [`self`]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $t_impl(pub $t);

        impl Add for $t_impl {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                $t_impl(self.0.wrapping_add(rhs.0))
            }
        }

        impl AddAssign for $t_impl {
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0.wrapping_add(rhs.0);
            }
        }

        impl<const BLEN: usize> Group<BLEN> for $t_impl {
            fn zero() -> Self {
                $t_impl(0)
            }

            fn add_inverse(self) -> Self {
                $t_impl(self.0.wrapping_neg())
            }
        }

        impl<const BLEN: usize> GroupEmbed<BLEN> for $t_impl {}

        impl<const BLEN: usize> From<[u8; BLEN]> for $t_impl {
            fn from(value: [u8; BLEN]) -> Self {
                if cfg!(not(feature = "int-be")) {
                    $t_impl(<$t>::from_le_bytes(
                        (&value[..size_of::<$t>()]).clone().try_into().unwrap(),
                    ))
                } else {
                    $t_impl(<$t>::from_be_bytes(
                        (&value[..size_of::<$t>()]).clone().try_into().unwrap(),
                    ))
                }
            }
        }

        impl<const BLEN: usize> From<$t_impl> for [u8; BLEN] {
            fn from(value: $t_impl) -> Self {
                let mut bs = [0; BLEN];
                if cfg!(not(feature = "int-be")) {
                    bs[..size_of::<$t>()].copy_from_slice(&value.0.to_le_bytes());
                } else {
                    bs[..size_of::<$t>()].copy_from_slice(&value.0.to_be_bytes());
                }
                bs
            }
        }
    };
}

decl_int_group!(u8, U8Group);
decl_int_group!(u16, U16Group);
decl_int_group!(u32, U32Group);
decl_int_group!(u64, U64Group);
decl_int_group!(u128, U128Group);

decl_int_group!(i8, I8Group);
decl_int_group!(i16, I16Group);
decl_int_group!(i32, I32Group);
decl_int_group!(i64, I64Group);
decl_int_group!(i128, I128Group);
