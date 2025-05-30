use crate::gradient_function::NoneBackwards;
use crate::{Tensor, TensorDataType};
use std::ops::{Add, Div, Mul, Neg, Sub};

use paste::paste;
use crate::backwards::{AddBackwards, DivBackwards, MulBackwards, NegBackwards, SubBackwards};

impl<T: TensorDataType> Neg for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output { -&self }
}

impl<T: TensorDataType> Neg for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { NegBackwards::new(self) } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts(-&self.array, requires_grad, grad_fn) }
    }
}

macro_rules! implement_binary_ops {
    ($($trait_: ident, $operator:tt, $method: ident, $backwards:ident;)* ) => { $(
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

                unsafe { Tensor::from_raw_parts(&self.array $operator &rhs.array, requires_grad, grad_fn) }
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
                let grad_fn = if requires_grad { NoneBackwards::new() } else { NoneBackwards::new() };

                unsafe { Tensor::from_raw_parts(&self.array $operator rhs, requires_grad, grad_fn) }
            }
        }
    )*};
}

implement_binary_ops!(
    Add, +, add, AddBackwards;
    Sub, -, sub, SubBackwards;
    Mul, *, mul, MulBackwards;
    Div, /, div, DivBackwards;
    // Rem, %, rem, RemBackwards;
);
