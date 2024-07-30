// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! Integers as a group which is a `$p$`-group.
//! `MOD` as the `$p$` is a prime number as the cardinality of the group.
//! Some prime numbers that are the max ones less than or equal to `u*::MAX` are provided as `PRIME_MAX_LE_U*_MAX`.
//!
//! - Associative operation: Integer addition modulo `MOD`, `$(a + b) \mod MOD$`.
//! - Identity element: 0.
//! - Inverse element: `-x`.

use std::mem::size_of;
use std::ops::{Add, AddAssign, Neg};

use super::Group;

macro_rules! decl_int_prime_group {
    ($t:ty, $t_impl:ident) => {
        /// See [`self`].
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $t_impl<const MOD: $t>(
            /// Always less than `MOD`.
            $t,
        );

        impl<const MOD: $t> $t_impl<MOD> {
            pub fn new(x: $t) -> Self {
                $t_impl(x % MOD)
            }
        }

        impl<const MOD: $t> Add for $t_impl<MOD> {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                $t_impl(match self.0.checked_add(rhs.0) {
                    Some(x) => x % MOD,
                    None => {
                        (self.0.wrapping_add(rhs.0) % MOD)
                            .wrapping_add(<$t>::MAX % MOD)
                            .wrapping_add(1)
                            % MOD
                    }
                })
            }
        }

        impl<const MOD: $t> AddAssign for $t_impl<MOD> {
            fn add_assign(&mut self, rhs: Self) {
                self.0 = match self.0.checked_add(rhs.0) {
                    Some(x) => x % MOD,
                    None => {
                        self.0
                            .wrapping_add(rhs.0)
                            .wrapping_add(<$t>::MAX % MOD)
                            .wrapping_add(1)
                            % MOD
                    }
                };
            }
        }

        impl<const MOD: $t> Neg for $t_impl<MOD> {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self.0 = (MOD - self.0) % MOD;
                self
            }
        }

        impl<const BLEN: usize, const MOD: $t> Group<BLEN> for $t_impl<MOD> {
            fn zero() -> Self {
                $t_impl(0)
            }
        }

        impl<const BLEN: usize, const MOD: $t> From<[u8; BLEN]> for $t_impl<MOD> {
            fn from(value: [u8; BLEN]) -> Self {
                if cfg!(not(feature = "int-be")) {
                    $t_impl(
                        <$t>::from_le_bytes(
                            (&value[..size_of::<$t>()]).clone().try_into().unwrap(),
                        ) % MOD,
                    )
                } else {
                    $t_impl(
                        <$t>::from_be_bytes(
                            (&value[..size_of::<$t>()]).clone().try_into().unwrap(),
                        ) % MOD,
                    )
                }
            }
        }

        impl<const MOD: $t> From<$t> for $t_impl<MOD> {
            fn from(value: $t) -> Self {
                <$t_impl<MOD>>::new(value)
            }
        }

        impl<const BLEN: usize, const MOD: $t> From<$t_impl<MOD>> for [u8; BLEN] {
            fn from(value: $t_impl<MOD>) -> Self {
                let mut bs = [0; BLEN];
                if cfg!(not(feature = "int-be")) {
                    bs[..size_of::<$t>()].copy_from_slice(&value.0.to_le_bytes());
                } else {
                    bs[..size_of::<$t>()].copy_from_slice(&value.0.to_be_bytes());
                }
                bs
            }
        }

        impl<const MOD: $t> From<$t_impl<MOD>> for $t {
            fn from(value: $t_impl<MOD>) -> Self {
                value.0
            }
        }
    };
}

decl_int_prime_group!(u8, U8Group);
decl_int_prime_group!(u16, U16Group);
decl_int_prime_group!(u32, U32Group);
decl_int_prime_group!(u64, U64Group);
decl_int_prime_group!(u128, U128Group);

/// `$2^8 - 5$`
pub const PRIME_MAX_LE_U8_MAX: u8 = u8::MAX - 5 + 1;
/// `$2^16 - 15$`
pub const PRIME_MAX_LE_U16_MAX: u16 = u16::MAX - 15 + 1;
/// `$2^32 - 5$`
pub const PRIME_MAX_LE_U32_MAX: u32 = u32::MAX - 5 + 1;
/// `$2^64 - 59$`
pub const PRIME_MAX_LE_U64_MAX: u64 = u64::MAX - 59 + 1;
/// `$2^128 - 159$`
pub const PRIME_MAX_LE_U128_MAX: u128 = u128::MAX - 159 + 1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_group_axioms;

    test_group_axioms!(test_u8_group_axioms, U8Group<PRIME_MAX_LE_U8_MAX>, 1);
    test_group_axioms!(test_u16_group_axioms, U16Group<PRIME_MAX_LE_U16_MAX>, 2);
    test_group_axioms!(test_u32_group_axioms, U32Group<PRIME_MAX_LE_U32_MAX>, 4);
    test_group_axioms!(test_u64_group_axioms, U64Group<PRIME_MAX_LE_U64_MAX>, 8);
    test_group_axioms!(test_u128_group_axioms, U128Group<PRIME_MAX_LE_U128_MAX>, 16);
}
