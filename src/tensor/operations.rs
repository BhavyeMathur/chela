use crate::broadcast::broadcast_shapes;
use crate::{IntegerDataType, RawDataType, Tensor, TensorMethods};
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Rem, Shl, Shr, Sub};

use paste::paste;
use crate::arithmetic_backwards::{AddBackwards, MulBackwards, NegBackwards, SubBackwards};

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

macro_rules! define_float_binary_ops {
    ($dtype:ty, $($trait_: ident, $operator: tt, $method: ident, $backwards: ident;)* ) => {
        $(
            fn $method<'a, 'b>(lhs: impl AsRef<Tensor<'a, $dtype>>,
                               rhs: impl AsRef<Tensor<'b, $dtype>>) -> Tensor<'static, $dtype>
            {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();

                let shape = broadcast_shapes(&lhs.shape, &rhs.shape);
                let lhs = lhs.broadcast_to(&shape);
                let rhs = rhs.broadcast_to(&shape);

                let requires_grad = lhs.requires_grad() || rhs.requires_grad();

                let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs $operator rhs).collect();
                let mut result = unsafe { Tensor::from_contiguous_owned_buffer(shape, data, requires_grad, false) };

                if requires_grad {
                    result.grad_fn = $backwards::new(lhs, rhs);
                }

                result
            }

            paste! { fn [<$method _scalar>] <'a, 'b>(lhs: impl AsRef<Tensor<'a, $dtype>>,
                                                     rhs: $dtype) -> Tensor<'static, $dtype>
                {
                    let lhs = lhs.as_ref();

                    let requires_grad = lhs.requires_grad();

                    let data = lhs.flatiter().map(|lhs| lhs $operator rhs).collect();
                    unsafe { Tensor::from_contiguous_owned_buffer(lhs.shape.clone(), data, requires_grad, false) }
                }
            }
        )*
    }
}

macro_rules! implement_binary_ops {
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
    };

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

pub trait TensorBinaryOps<T: RawDataType> {
    define_binary_ops!(
        Add, +, add;
        Sub, -, sub;
        Mul, *, mul;
        Div, /, div;
        Rem, %, rem;
        BitAnd, &, bitand;
        BitOr, |, bitor;
        Shl, <<, shl;
        Shr, >>, shr;
    );

    fn neg<'a, 'b>(rhs: impl AsRef<Tensor<'a, T>>) -> Tensor<'static, T>
    where
        T: Neg<Output=T>,
    {
        let rhs = rhs.as_ref();

        let data = rhs.flatiter().map(|rhs| -rhs).collect();
        unsafe { Tensor::from_contiguous_owned_buffer(rhs.shape.clone(), data, false, false) }
    }
}

implement_binary_ops!(
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div;
    Rem, rem;
    BitAnd, bitand;
    BitOr, bitor;
    Shl, shl;
    Shr, shr;
);

impl<T: RawDataType + Neg<Output=T>> Neg for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output { <T as TensorBinaryOps<T>>::neg(self) }
}

impl<T: RawDataType + Neg<Output=T>> Neg for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output { <T as TensorBinaryOps<T>>::neg(self) }
}

impl<T: IntegerDataType> TensorBinaryOps<T> for T {}
impl TensorBinaryOps<bool> for bool {}

impl TensorBinaryOps<f32> for f32 {
    define_float_binary_ops!(
        f32,
        Add, +, add, AddBackwards;
        Sub, -, sub, SubBackwards;
        Mul, *, mul, MulBackwards;
    );

    fn neg<'a, 'b>(rhs: impl AsRef<Tensor<'a, f32>>) -> Tensor<'static, f32> {
        let rhs = rhs.as_ref();
        
        let requires_grad = rhs.requires_grad();

        let data = rhs.flatiter().map(|rhs| -rhs).collect();
        let mut result = unsafe { Tensor::from_contiguous_owned_buffer(rhs.shape.clone(), data, requires_grad, false) };
        
        if requires_grad {
            result.grad_fn = NegBackwards::new(rhs);
        }
        
        result
    }
}

impl TensorBinaryOps<f64> for f64 {
    define_float_binary_ops!(
        f64,
        Add, +, add, AddBackwards;
        Sub, -, sub, SubBackwards;
        Mul, *, mul, MulBackwards;
    );

    fn neg<'a, 'b>(rhs: impl AsRef<Tensor<'a, f64>>) -> Tensor<'static, f64> {
        let rhs = rhs.as_ref();

        let requires_grad = rhs.requires_grad();

        let data = rhs.flatiter().map(|rhs| -rhs).collect();
        let mut result = unsafe { Tensor::from_contiguous_owned_buffer(rhs.shape.clone(), data, requires_grad, false) };

        if requires_grad {
            result.grad_fn = NegBackwards::new(rhs);
        }

        result
    }
}
