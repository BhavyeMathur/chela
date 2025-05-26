use crate::broadcast::broadcast_shapes;
use crate::{RawDataType, Tensor};
use std::ops::{Add, Div, Mul, Rem, Sub};

use paste::paste;

macro_rules! define_binary_ops {
    ($($trait_: ident, $operator: tt, $method: ident;)* ) => {
        $(
            fn $method<'a, 'b>(lhs: impl AsRef<Tensor<'a, T>>,
                               rhs: impl AsRef<Tensor<'b, T>>) -> Tensor<'static, T>
            where
                T: $trait_<Output=T>,
            {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();

                let shape = broadcast_shapes(&lhs.shape, &rhs.shape);
                let lhs = lhs.broadcast_to(&shape);
                let rhs = rhs.broadcast_to(&shape);

                let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs $operator rhs).collect();
                unsafe { Tensor::from_contiguous_owned_buffer(shape, data, false, false) }
            }

            paste! { fn [<$method _scalar>] <'a, 'b>(lhs: impl AsRef<Tensor<'a, T>>,
                                                     rhs: T) -> Tensor<'static, T> 
                where
                    T: $trait_<Output=T>,
                {
                    let lhs = lhs.as_ref();

                    let data = lhs.flatiter().map(|lhs| lhs $operator rhs).collect();
                    unsafe { Tensor::from_contiguous_owned_buffer(lhs.shape.clone(), data, false, false) }
                }
            }
        )*
    }
}

macro_rules! implement_binary_op_traits {
    ($($trait_: ident, $method: ident;)* ) => {
        $(
            impl<T: RawDataType + $trait_<Output=T>> $trait_ for Tensor<'_, T> {
                type Output = Tensor<'static, T>;

                fn $method(self, rhs: Tensor<T>) -> Self::Output { <T as TensorBinaryOps<T>>::$method(self, rhs) }
            }
            impl<T: RawDataType + $trait_<Output=T>> $trait_<&Tensor<'_, T>> for Tensor<'_, T> {
                type Output = Tensor<'static, T>;

                fn $method(self, rhs: &Tensor<T>) -> Self::Output { <T as TensorBinaryOps<T>>::$method(self, rhs) }
            }
            impl<T: RawDataType + $trait_<Output=T>> $trait_<Tensor<'_, T>> for &Tensor<'_, T> {
                type Output = Tensor<'static, T>;

                fn $method(self, rhs: Tensor<T>) -> Self::Output { <T as TensorBinaryOps<T>>::$method(self, rhs) }
            }
            impl<T: RawDataType + $trait_<Output=T>> $trait_<&Tensor<'_, T>> for &Tensor<'_, T> {
                type Output = Tensor<'static, T>;

                fn $method(self, rhs: &Tensor<T>) -> Self::Output { <T as TensorBinaryOps<T>>::$method(self, rhs) }
            }

            impl<T: RawDataType + $trait_<Output=T>> $trait_<T> for Tensor<'_, T> {
                type Output = Tensor<'static, T>;

                fn $method(self, rhs: T) -> Self::Output { paste! { <T as TensorBinaryOps<T>>::[<$method _scalar>](self, rhs) } }
            }

            impl<T: RawDataType + $trait_<Output=T>> $trait_<T> for &Tensor<'_, T> {
                type Output = Tensor<'static, T>;

                fn $method(self, rhs: T) -> Self::Output { paste! { <T as TensorBinaryOps<T>>::[<$method _scalar>](self, rhs) } }
            }
        )*
    }
}

pub trait TensorBinaryOps<T: RawDataType> {
    define_binary_ops!(
        Add, +, add;
        Sub, -, sub;
        Mul, *, mul;
        Div, /, div;
        Rem, %, rem;
    );
}

implement_binary_op_traits!(
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div;
    Rem, rem;
);

impl<T: RawDataType> TensorBinaryOps<T> for T {}
