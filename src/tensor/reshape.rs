use crate::none_backwards::NoneBackwards;
use crate::reshape_backwards::ReshapeBackwards;
use crate::transpose_backwards::TransposeBackwards;
use crate::{AxisType, Reshape, StridedMemory, Tensor, TensorDataType};
use crate::identity_backwards::IdentityBackwards;


impl<'a, T: TensorDataType> Reshape<T> for &'a Tensor<'a, T> {
    type Output = Tensor<'a, T>;

    /// Provides a non-owning view of the tensor with the specified shape and stride.
    /// The data pointed to by the view is shared with the original tensor.
    ///
    /// # Safety
    /// - Ensure the memory layout referenced by `shape`, and `stride` is valid and owned
    ///   by the original tensor.
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { ReshapeBackwards::new(self, self.shape()) } else { NoneBackwards::new() };

        let result = self.array.as_ref().reshaped_view(shape, stride);

        // NdArray<'static, T> needed to create a shared pointer to the result
        // this function outputs a Tensor<'a, T> where ('a: 'static) so it should be safe.
        let result = result.lifetime_cast();

        Tensor::from_raw_parts(result, requires_grad, grad_fn)
    }

    /// Provides a non-owning view of the tensor that shares its data with the original tensor.
    ///
    /// # Example
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let tensor = Tensor::new([1.0, 2.0, 3.0, 4.0]);
    /// let view = (&tensor).view();
    /// assert!(view.is_view())
    /// ```
    fn view(self) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { IdentityBackwards::new(self) } else { NoneBackwards::new() };

        let result = self.array.as_ref().view();

        unsafe {
            // NdArray<'static, T> needed to create a shared pointer to the result
            // this function outputs a Tensor<'a, T> where ('a: 'static) so it should be safe.
            let result = result.lifetime_cast();

            Tensor::from_raw_parts(result, requires_grad, grad_fn)
        }
    }

    /// Returns a transposed version of the tensor, swapping the specified axes.
    ///
    /// # Panics
    /// - If `axis1` or `axis2` are out of bounds
    ///
    /// # Examples
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let array = Tensor::new([[2.0, 3.0, 4.0], [10.0, 20.0, 30.0]]);
    ///
    /// let transposed = array.transpose(0, 1);
    /// assert_eq!(transposed, Tensor::new([[2.0, 10.0], [3.0, 20.0], [4.0, 30.0]]));
    /// ```
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

    /// Provides a non-owning view of the tensor with the specified shape and stride.
    /// The data pointed to by the view is shared with the original tensor.
    ///
    /// # Safety
    /// - Ensure the memory layout referenced by `shape`, and `stride` is valid and owned
    ///   by the original tensor.
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { ReshapeBackwards::new(&self, self.shape()) } else { NoneBackwards::new() };

        let result = self.into_ndarray().reshaped_view(shape, stride);
        Tensor::from_raw_parts(result, requires_grad, grad_fn)
    }

    /// Provides a non-owning view of the tensor that shares its data with the original tensor.
    ///
    /// # Example
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let tensor = Tensor::new([1.0, 2.0, 3.0, 4.0]);
    /// let view = (&tensor).view();
    /// assert!(view.is_view())
    /// ```
    fn view(self) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn = if requires_grad { IdentityBackwards::new(&self) } else { NoneBackwards::new() };

        let result = self.into_ndarray().view();
        unsafe { Tensor::from_raw_parts(result, requires_grad, grad_fn) }
    }

    /// Returns a transposed version of the tensor, swapping the specified axes.
    ///
    /// # Panics
    /// - If `axis1` or `axis2` are out of bounds
    ///
    /// # Examples
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let array = Tensor::new([[2.0, 3.0, 4.0], [10.0, 20.0, 30.0]]);
    ///
    /// let transposed = array.transpose(0, 1);
    /// assert_eq!(transposed, Tensor::new([[2.0, 10.0], [3.0, 20.0], [4.0, 30.0]]));
    /// ```
    fn transpose(self, axis1: impl AxisType, axis2: impl AxisType) -> Self::Output {
        let requires_grad = self.requires_grad();
        let grad_fn =
            if requires_grad {
                TransposeBackwards::new(&self, axis1.isize(), axis2.isize())
            } else {
                NoneBackwards::new()
            };

        let result = self.into_ndarray().transpose(axis1, axis2);
        unsafe { Tensor::from_raw_parts(result, requires_grad, grad_fn) }
    }
}
