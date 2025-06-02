use crate::common::binary_ops::BinaryOps;
use crate::{NdArray, RawDataType};
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr, Sub};

use crate::ops::binary_op::*;
use crate::ops::binary_op_addition::BinaryOpAdd;
use crate::ops::binary_op_multiplication::BinaryOpMul;
use paste::paste;


macro_rules! implement_binary_ops {
    ($object:ident, $($trait_:ident, $binary_op_trait:ident, $operator:tt, $method: ident;)* ) => { $(
        impl<T: RawDataType + $binary_op_trait> $trait_<$object<'_, T>> for $object<'_, T> {
            type Output = $object<'static, T>;

            fn $method(self, rhs: $object<T>) -> Self::Output { &self $operator &rhs }
        }

        impl<T: RawDataType + $binary_op_trait> $trait_<&$object<'_, T>> for $object<'_, T> {
            type Output = $object<'static, T>;

            fn $method(self, rhs: &$object<T>) -> Self::Output { &self $operator rhs }
        }
        
        impl<T: RawDataType + $binary_op_trait> $trait_<$object<'_, T>> for &$object<'_, T> {
            type Output = $object<'static, T>;

            fn $method(self, rhs: $object<T>) -> Self::Output { self $operator &rhs }
        }

        impl<T: RawDataType + $binary_op_trait> $trait_<&$object<'_, T>> for &$object<'_, T> {
            type Output = $object<'static, T>;

            fn $method(self, rhs: &$object<T>) -> Self::Output { <T as BinaryOps>::$method(self, rhs) }
        }
        
        impl<T: RawDataType + $trait_<Output=T>> $trait_<T> for $object<'_, T> {
            type Output = $object<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { paste! { &self $operator rhs } }
        }

        impl<T: RawDataType + $trait_<Output=T>> $trait_<T> for &$object<'_, T> {
            type Output = $object<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { paste! { <T as BinaryOps>::[<$method _scalar>](self, rhs) } }
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
    NdArray,
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
