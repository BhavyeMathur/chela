use crate::RawDataType;
use crate::ndarray::NdArrayFlags;
use crate::NdArray;
use std::ops::{AddAssign, BitAndAssign, BitOrAssign, DivAssign, MulAssign, RemAssign, ShlAssign, ShrAssign, SubAssign};

macro_rules! define_binary_iop {
    ( $trait_: ident, $operator: tt, $method: ident ) => {
    impl<T: RawDataType + $trait_> $trait_<NdArray<'_, T>> for NdArray<'_, T> {
        fn $method(&mut self, rhs: NdArray<'_, T>) {
            *self $operator &rhs
        }
    }

    impl<T: RawDataType + $trait_> $trait_<&NdArray<'_, T>> for NdArray<'_, T> {
        fn $method(&mut self, rhs: &NdArray<'_, T>) {
            if !self.flags.contains(NdArrayFlags::Writeable) {
                panic!("tensor is readonly.");
            }

            let rhs = rhs.broadcast_to(&self.shape);
            
            for (lhs, rhs) in self.flatiter_ptr().zip(rhs.flatiter()) {
                unsafe { *lhs $operator rhs; }
            }
        }
    }
    };
}

define_binary_iop!(AddAssign, +=, add_assign);
define_binary_iop!(SubAssign, -=, sub_assign);
define_binary_iop!(MulAssign, *=, mul_assign);
define_binary_iop!(DivAssign, /=, div_assign);
define_binary_iop!(RemAssign, %=, rem_assign);
define_binary_iop!(BitAndAssign, &=, bitand_assign);
define_binary_iop!(BitOrAssign, |=, bitor_assign);
define_binary_iop!(ShlAssign, <<=, shl_assign);
define_binary_iop!(ShrAssign, >>=, shr_assign);
