use crate::{Tensor, TensorDataType};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::add_backwards::*;
use crate::div_backwards::*;
use crate::mul_backwards::*;
use crate::neg_backwards::*;
use crate::none_backwards::*;
use crate::sub_backwards::*;
use paste::paste;

impl<T: TensorDataType> Neg for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output { -&self }
}

impl<T: TensorDataType> Neg for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { NegBackwards::new(self) } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts(-self.array.as_ref(), requires_grad, grad_fn) }
    }
}

macro_rules! implement_binary_ops {
    ($($trait_: ident, $operator:tt, $method: ident, $backwards:ident, $backwards_scalar:ident;)* ) => { $(
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
                let requires_grad = self.requires_grad() || rhs.requires_grad();
                let grad_fn = if requires_grad { $backwards::new(self, rhs) } else { NoneBackwards::new() };

                unsafe { Tensor::from_raw_parts(self.array.as_ref() $operator rhs.array.as_ref(), requires_grad, grad_fn) }
            }
        }
        
        impl<T: TensorDataType> $trait_<T> for Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output { paste! { &self $operator rhs } }
        }

        impl<T: TensorDataType> $trait_<T> for &Tensor<'_, T> {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output {
                let requires_grad = self.requires_grad();
                let grad_fn = if requires_grad { $backwards_scalar::new(self, rhs) } else { NoneBackwards::new() };

                unsafe { Tensor::from_raw_parts(self.array.as_ref() $operator rhs, requires_grad, grad_fn) }
            }
        }
    )*};
}

implement_binary_ops!(
    Add, +, add, AddBackwards, AddScalarBackwards;
    Sub, -, sub, SubBackwards, AddScalarBackwards;
    Mul, *, mul, MulBackwards, MulScalarBackwards;
    Div, /, div, DivBackwards, DivScalarBackwards;
    // Rem, %, rem, RemBackwards;
);
