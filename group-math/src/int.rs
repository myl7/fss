//! Group of a integer which defines addition as integer (wrapping) addition

use std::mem::size_of;
use std::ops::{Add, AddAssign};

use crate::Group;

macro_rules! decl_int_group {
    ($t:ty, $t_impl:ident) => {
        /// See [`crate::int`]
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

        impl<const LAMBDA: usize> Group<LAMBDA> for $t_impl {
            fn convert(y: [u8; LAMBDA]) -> Self {
                if cfg!(not(feature = "int-be")) {
                    $t_impl(<$t>::from_le_bytes(
                        (&y[..size_of::<$t>()]).clone().try_into().unwrap(),
                    ))
                } else {
                    $t_impl(<$t>::from_be_bytes(
                        (&y[..size_of::<$t>()]).clone().try_into().unwrap(),
                    ))
                }
            }

            fn zero() -> Self {
                $t_impl(0)
            }

            fn add_inverse(self) -> Self {
                $t_impl(self.0.wrapping_neg())
            }
        }
    };
}

decl_int_group!(u8, U8Group);
decl_int_group!(u16, U16Group);
decl_int_group!(u32, U32Group);
decl_int_group!(u64, U64Group);
decl_int_group!(u128, U128Group);
decl_int_group!(usize, UsizeGroup);

decl_int_group!(i8, I8Group);
decl_int_group!(i16, I16Group);
decl_int_group!(i32, I32Group);
decl_int_group!(i64, I64Group);
decl_int_group!(i128, I128Group);
decl_int_group!(isize, IsizeGroup);
