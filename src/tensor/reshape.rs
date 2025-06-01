use crate::none_backwards::NoneBackwards;
use crate::reshape_backwards::ReshapeBackwards;
use crate::transpose_backwards::TransposeBackwards;
use crate::{AxisType, Reshape, StridedMemory, Tensor, TensorDataType};

impl<'a, T: TensorDataType> Reshape<T> for &'a Tensor<'a, T> {
    type Output = Tensor<'a, T>;
    
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { ReshapeBackwards::new(self, self.shape()) } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts((&self.array).reshaped_view(shape, stride), requires_grad, grad_fn) }
    }

    fn transpose(self, axis1: impl AxisType, axis2: impl AxisType) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn =
            if requires_grad {
                TransposeBackwards::new(self, axis1.isize(), axis2.isize())
            } else {
                NoneBackwards::new()
            };

        let result = (&self.array).transpose(axis1, axis2);

        unsafe { Tensor::from_raw_parts(result, requires_grad, grad_fn) }
    }
}

impl<T: TensorDataType> Reshape<T> for Tensor<'_, T> {
    type Output = Tensor<'static, T>;
    
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { ReshapeBackwards::new(&self, self.shape()) } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts(self.array.reshaped_view(shape, stride), requires_grad, grad_fn) }
    }
    
    fn transpose(self, axis1: impl AxisType, axis2: impl AxisType) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn =
            if requires_grad {
                TransposeBackwards::new(&self, axis1.isize(), axis2.isize())
            } else {
                NoneBackwards::new()
            };

        let result = self.array.transpose(axis1, axis2);

        unsafe { Tensor::from_raw_parts(result, requires_grad, grad_fn) }
    }
}
