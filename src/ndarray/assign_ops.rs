use crate::ndarray::NdArrayFlags;
use crate::ops::binary_op_add::BinaryOpAdd;
use crate::ops::binary_op_div::BinaryOpDiv;
use crate::ops::binary_op_mul::BinaryOpMul;
use crate::ops::binary_op_sub::BinaryOpSub;
use crate::ops::binary_ops::{BinaryOpBitAnd, BinaryOpBitOr, BinaryOpRem, BinaryOpShl, BinaryOpShr};
use crate::RawDataType;
use crate::{NdArray, StridedMemory};
use paste::paste;
use std::ops::{AddAssign, BitAndAssign, BitOrAssign, DivAssign, MulAssign, RemAssign, ShlAssign, ShrAssign, SubAssign};


macro_rules! define_binary_iop {
    ( $binary_op_trait:ident, $iop_trait:ident, $operator:tt, $method:ident ) => {
        paste! {
            impl<T: RawDataType + $binary_op_trait> $iop_trait<NdArray<'_, T>> for NdArray<'_, T> {
                fn [<$method _assign>](&mut self, rhs: NdArray<'_, T>) {
                    *self $operator &rhs
                }
            }

            impl<T: RawDataType + $binary_op_trait> $iop_trait<&NdArray<'_, T>> for NdArray<'_, T> {
                fn [<$method _assign>](&mut self, rhs: &NdArray<'_, T>) {
                    if !self.flags.contains(NdArrayFlags::Writeable) {
                        panic!("tensor is readonly.");
                    }

                    let rhs = rhs.broadcast_to(&self.shape);
    
                    unsafe {
                        <T as $binary_op_trait>::$method(self.ptr(), &self.stride(),
                                                         rhs.ptr(), &rhs.stride(),
                                                         self.mut_ptr(), self.shape());
                    }
                }
            }

            impl<T: RawDataType + $binary_op_trait> $iop_trait<T> for NdArray<'_, T> {
                fn [<$method _assign>](&mut self, rhs: T) {
                    if !self.flags.contains(NdArrayFlags::Writeable) {
                        panic!("tensor is readonly.");
                    }

                    unsafe {
                        <T as $binary_op_trait>::[<$method _scalar>](self.ptr(), self.shape(), self.stride(),
                                                                     rhs, self.mut_ptr());
                    }
                }
            }
        }
    };
}

define_binary_iop!(BinaryOpAdd, AddAssign, +=, add);
define_binary_iop!(BinaryOpSub, SubAssign, -=, sub);
define_binary_iop!(BinaryOpMul, MulAssign, *=, mul);
define_binary_iop!(BinaryOpDiv, DivAssign, /=, div);
define_binary_iop!(BinaryOpRem, RemAssign, %=, rem);
define_binary_iop!(BinaryOpBitAnd, BitAndAssign, &=, bitand);
define_binary_iop!(BinaryOpBitOr, BitOrAssign, |=, bitor);
define_binary_iop!(BinaryOpShl, ShlAssign, <<=, shl);
define_binary_iop!(BinaryOpShr, ShrAssign, >>=, shr);
