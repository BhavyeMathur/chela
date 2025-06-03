use crate::broadcast::broadcast_shapes;
use crate::broadcast::broadcast_stride;
use crate::common::constructors::Constructors;
use crate::{NdArray, RawDataType, StridedMemory};
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr, Sub};

use crate::ops::binary_op::*;
use crate::ops::binary_op_add::BinaryOpAdd;
use crate::ops::binary_op_div::BinaryOpDiv;
use crate::ops::binary_op_mul::BinaryOpMul;
use crate::ops::binary_op_sub::BinaryOpSub;
use paste::paste;

macro_rules! implement_binary_ops {
    ($($binary_op:ident, $binary_op_trait:ident, $operator:tt, $method: ident;)* ) => { $(
        impl<T: RawDataType + $binary_op_trait> $binary_op<NdArray<'_, T>> for NdArray<'_, T> {
            type Output = NdArray<'static, T>;

            fn $method(self, rhs: NdArray<T>) -> Self::Output { &self $operator &rhs }
        }

        impl<T: RawDataType + $binary_op_trait> $binary_op<&NdArray<'_, T>> for NdArray<'_, T> {
            type Output = NdArray<'static, T>;

            fn $method(self, rhs: &NdArray<T>) -> Self::Output { &self $operator rhs }
        }
        
        impl<T: RawDataType + $binary_op_trait> $binary_op<NdArray<'_, T>> for &NdArray<'_, T> {
            type Output = NdArray<'static, T>;

            fn $method(self, rhs: NdArray<T>) -> Self::Output { self $operator &rhs }
        }

        impl<T: RawDataType + $binary_op_trait> $binary_op<&NdArray<'_, T>> for &NdArray<'_, T> {
            type Output = NdArray<'static, T>;

            fn $method(self, rhs: &NdArray<T>) -> Self::Output {
                let shape = broadcast_shapes(self.shape(), rhs.shape());
                let lhs_stride = broadcast_stride(self.stride(), &shape, self.shape());
                let rhs_stride = broadcast_stride(rhs.stride(), &shape, rhs.shape());

                let mut data = vec![T::default(); shape.iter().product()];

                unsafe {
                    <T as $binary_op_trait>::$method(self.ptr(), &lhs_stride,
                                                     rhs.ptr(), &rhs_stride,
                                                     data.as_mut_ptr(), &shape);
                }

                unsafe { NdArray::from_contiguous_owned_buffer(shape, data) }
            }
        }
        
        impl<T: RawDataType + $binary_op<Output=T>> $binary_op<T> for NdArray<'_, T> {
            type Output = NdArray<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { paste! { &self $operator rhs } }
        }

        impl<T: RawDataType + $binary_op<Output=T>> $binary_op<T> for &NdArray<'_, T> {
            type Output = NdArray<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { paste! {
                let data = self.flatiter().map(|lhs| lhs $operator rhs).collect();
                unsafe { NdArray::from_contiguous_owned_buffer(self.shape().to_vec(), data) }
            } }
        }
    )*};

    ($dtype1:ty, $dtype2:ty, $($trait_: ident, $method: ident;)* ) => {
        implement_binary_ops!($dtype1, $($trait_, $method;)* );
        implement_binary_ops!($dtype2, $($trait_, $method;)* );
    };

    ($dtype1:ty, $dtype2:ty, $dtype3:ty, $dtype4:ty, $($trait_: ident, $method: ident;)* ) => {
        implement_binary_ops!($dtype1, $dtype2, $($trait_, $method;)* );
        implement_binary_ops!($dtype3, $dtype4, $($trait_, $method;)* );
        implement_binary_ops!($dtype5, $dtype6, $($trait_, $method;)* );
    }
}


implement_binary_ops!(
    Add, BinaryOpAdd, +, add;
    Sub, BinaryOpSub, -, sub;
    Mul, BinaryOpMul, *, mul;
    Div, BinaryOpDiv, /, div;
    Rem, BinaryOpRem, %, rem;
    BitAnd, BinaryOpBitAnd, &, bitand;
    BitOr, BinaryOpBitOr, |, bitor;
    Shl, BinaryOpShl, <<, shl;
    Shr, BinaryOpShr, >>, shr;
);
