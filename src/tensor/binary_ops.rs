use crate::broadcast::broadcast_shapes;
use crate::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;
use crate::{Tensor, TensorMethods};
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};

macro_rules! define_binary_op {
    ( $trait_: ident, $operator: tt, $method: ident ) => {
        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_ for Tensor<'_, T>
        where
            for<'a> Tensor<'a, T>: TensorMethods
        {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<&Tensor<'_, T>> for Tensor<'_, T>
        where
            for<'a> Tensor<'a, T>: TensorMethods
        {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<Tensor<'_, T>> for &Tensor<'_, T>
        where
            for<'a> Tensor<'a, T>: TensorMethods
        {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<&Tensor<'_, T>> for &Tensor<'_, T>
        where
            for<'a> Tensor<'a, T>: TensorMethods
        {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                let shape = broadcast_shapes(&self.shape, &rhs.shape);
                let lhs = self.broadcast_to(&shape);
                let rhs = rhs.broadcast_to(&shape);

                let requires_grad = self.requires_grad() || rhs.requires_grad();

                let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs $operator rhs).collect();
                unsafe { Tensor::from_contiguous_owned_buffer(shape, data, requires_grad, false) }
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<T> for Tensor<'_, T>
        where
            for<'a> Tensor<'a, T>: TensorMethods
        {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: RawDataType + $trait_<Output=T> + 'static> $trait_<T> for &Tensor<'_, T>
        where
            for<'a> Tensor<'a, T>: TensorMethods
        {
            type Output = Tensor<'static, T>;

            fn $method(self, rhs: T) -> Self::Output {
                let data = self.flatiter().map(|lhs| lhs $operator rhs).collect();
                unsafe { Tensor::from_contiguous_owned_buffer(self.shape.clone(), data, self.requires_grad(), false) }
            }
        }
    };
}

macro_rules! define_binary_iop {
    ( $trait_: ident, $operator: tt, $method: ident ) => {
    impl<T: RawDataType + $trait_ + 'static> $trait_<Tensor<'_, T>> for Tensor<'_, T> {
        fn $method(&mut self, rhs: Tensor<'_, T>) {
            *self $operator &rhs
        }
    }

    impl<T: RawDataType + $trait_ + 'static> $trait_<&Tensor<'_, T>> for Tensor<'_, T> {
        fn $method(&mut self, rhs: &Tensor<'_, T>) {
            if !self.flags.contains(TensorFlags::Writeable) {
                panic!("Tensor is readonly");
            }

            let rhs = rhs.broadcast_to(&self.shape);

            for (lhs, rhs) in self.flatiter_ptr().zip(rhs.flatiter()) {
                unsafe { *lhs $operator rhs; }
            }
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

define_binary_iop!(AddAssign, +=, add_assign);
define_binary_iop!(SubAssign, -=, sub_assign);
define_binary_iop!(MulAssign, *=, mul_assign);
define_binary_iop!(DivAssign, /=, div_assign);
define_binary_iop!(RemAssign, %=, rem_assign);
define_binary_iop!(BitAndAssign, &=, bitand_assign);
define_binary_iop!(BitOrAssign, |=, bitor_assign);
define_binary_iop!(ShlAssign, <<=, shl_assign);
define_binary_iop!(ShrAssign, >>=, shr_assign);
