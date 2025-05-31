use crate::dot_backwards::DotBackwards;
use crate::{Tensor, TensorDataType};

impl<'a, T: TensorDataType> Tensor<'a, T> {
    /// Calculates the dot product of two 1D tensors.
    ///
    /// # Panics
    /// - Panics if either tensor is not 1D
    /// - Panics if the lengths of the two tensors are not equal
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    /// let tensor1 = Tensor::from([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::from([4.0, 5.0, 6.0]);
    /// let result = tensor1.dot(tensor2);
    /// assert_eq!(result.value(), 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn dot<'b, 'r>(&self, other: impl AsRef<Tensor<'b, T>>) -> Tensor<'r, T> {
        let other = other.as_ref();
        let requires_grad = self.requires_grad() || other.requires_grad();

        unsafe { Tensor::from_raw_parts(self.array.dot(&other.array), requires_grad, DotBackwards::new(self, other)) }
    }
}
