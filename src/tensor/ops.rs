use crate::gradient_function::NoneBackwards;
use crate::ndarray::flags::NdArrayFlags;
use crate::{Tensor, TensorDataType};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use paste::paste;

impl<T: TensorDataType> Neg for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output { -&self }
}

impl<T: TensorDataType> Neg for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output {
        Tensor {
            array: -&self.array,

            flags: NdArrayFlags::empty(),
            grad_fn: NoneBackwards::new(),
        }
    }
}

macro_rules! implement_binary_ops {
    ($($trait_: ident, $operator:tt, $method: ident;)* ) => { $(
        impl<T: TensorDataType> $trait_<Tensor<'_, T>> for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output { &self $operator &rhs }
        }

        impl<T: TensorDataType> $trait_<&Tensor<'_, T>> for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output { &self $operator rhs }
        }
        
        impl<T: TensorDataType> $trait_<Tensor<'_, T>> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output { self $operator &rhs }
        }

        impl<T: TensorDataType> $trait_<&Tensor<'_, T>> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output { 
                unsafe { Tensor::from_raw_parts(&self.array $operator &rhs.array, false, false) }
            }
        }
        
        impl<T: TensorDataType> $trait_<T> for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { paste! { &self $operator rhs } }
        }

        impl<T: TensorDataType> $trait_<T> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { 
                unsafe { Tensor::from_raw_parts(&self.array $operator rhs, false, false) }
            }
        }
    )*};
}

implement_binary_ops!(
    Add, +, add;
    Sub, -, sub;
    Mul, *, mul;
    Div, /, div;
    Rem, %, rem;
);
