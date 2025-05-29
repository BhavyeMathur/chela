use crate::gradient_function::NoneBackwards;
use crate::ndarray::flags::NdArrayFlags;
use crate::{FloatDataType, Tensor};
use std::ops::Neg;

impl<T: FloatDataType> Neg for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output { -&self }
}

impl<T: FloatDataType> Neg for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn neg(self) -> Self::Output {
        Tensor {
            array: -&self.array,

            flags: NdArrayFlags::empty(),
            grad_fn: NoneBackwards::new(),
        }
    }
}
