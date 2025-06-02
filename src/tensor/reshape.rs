use crate::none_backwards::NoneBackwards;
use crate::reshape_backwards::ReshapeBackwards;
use crate::transpose_backwards::TransposeBackwards;
use crate::{AxisType, Reshape, StridedMemory, Tensor, TensorDataType};
use std::rc::Rc;

impl<'a, T: TensorDataType> Reshape<T> for &'a Tensor<'a, T> {
    type Output = Tensor<'a, T>;

    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { ReshapeBackwards::new(self, self.shape()) } else { NoneBackwards::new() };

        let result = self.array.as_ref().reshaped_view(shape, stride);

        // NdArray<'static, T> needed to create a shared pointer to the result
        // this function outputs a Tensor<'a, T> where ('a: 'static) so it should be safe.
        let result = result.lifetime_cast();

        Tensor::from_raw_parts(result, requires_grad, grad_fn)
    }

    fn transpose(self, axis1: impl AxisType, axis2: impl AxisType) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn =
            if requires_grad {
                TransposeBackwards::new(self, axis1.isize(), axis2.isize())
            } else {
                NoneBackwards::new()
            };

        let result = self.array.as_ref().transpose(axis1, axis2);

        unsafe {
            // NdArray<'static, T> needed to create a shared pointer to the result
            // this function outputs a Tensor<'a, T> where ('a: 'static) so it should be safe.
            let result = result.lifetime_cast();

            Tensor::from_raw_parts(result, requires_grad, grad_fn)
        }
    }
}

impl<T: TensorDataType> Reshape<T> for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { ReshapeBackwards::new(&self, self.shape()) } else { NoneBackwards::new() };

        let result = match Rc::try_unwrap(self.array) {
            Ok(array) => array,
            Err(rc) => rc.as_ref().clone(),
        };
        let result = result.reshaped_view(shape, stride);

        Tensor::from_raw_parts(result, requires_grad, grad_fn)
    }

    fn transpose(self, axis1: impl AxisType, axis2: impl AxisType) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn =
            if requires_grad {
                TransposeBackwards::new(&self, axis1.isize(), axis2.isize())
            } else {
                NoneBackwards::new()
            };

        let result = match Rc::try_unwrap(self.array) {
            Ok(array) => array,
            Err(rc) => rc.as_ref().clone(),
        };
        let result = result.transpose(axis1, axis2);

        unsafe { Tensor::from_raw_parts(result, requires_grad, grad_fn) }
    }
}
