use crate::broadcast::broadcast_shapes;
use crate::dtype::RawDataType;
use crate::Tensor;
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr, Sub};


macro_rules! define_binary_op {
    ( $trait_: ident, $operator: tt, $method: ident ) => {
        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_ for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<&Tensor<'_, T>> for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<Tensor<'_, T>> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<&Tensor<'_, T>> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                let shape = broadcast_shapes(&self.shape, &rhs.shape);
                let lhs = self.broadcast_to(&shape);
                let rhs = rhs.broadcast_to(&shape);

                let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs $operator rhs).collect();
                unsafe { Tensor::from_contiguous_owned_buffer(shape, data) }
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<T> for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<T> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output {
                let data = self.flatiter().map(|lhs| lhs $operator rhs).collect();
                unsafe { Tensor::from_contiguous_owned_buffer(self.shape.clone(), data) }
            }
        }
    };
}

define_binary_op!(Add, +, add);
define_binary_op!(Sub, -, sub);
define_binary_op!(Mul, *, mul);
define_binary_op!(Div, /, div);
define_binary_op!(Rem, %, rem);
define_binary_op!(BitAnd, &, bitand);
define_binary_op!(BitOr, |, bitor);
define_binary_op!(Shl, <<, shl);
define_binary_op!(Shr, >>, shr);
