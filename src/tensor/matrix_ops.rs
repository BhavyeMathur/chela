use crate::bmm_backwards::BMMBackwards;
use crate::dot_backwards::DotBackwards;
use crate::matrix_product_backwards::MatrixProductBackwards;
use crate::matrix_vec_backwards::MatrixVecBackwards;
use crate::none_backwards::NoneBackwards;
use crate::{StridedMemory, Tensor, TensorDataType};

impl<'a, T: TensorDataType> Tensor<'a, T> {
    /// Calculates the dot product of two 1D tensors.
    ///
    /// # Panics
    /// - Panics if either tensor is not 1D
    /// - Panics if the lengths of the two tensors are not equal
    ///
    /// # Examples
    /// ```
    /// # use redstone_ml::*;
    /// let tensor1 = Tensor::new([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::new([4.0, 5.0, 6.0]);
    /// let result = tensor1.dot(tensor2);
    /// assert_eq!(result.value(), 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn dot<'b, 'r>(&self, other: impl AsRef<Tensor<'b, T>>) -> Tensor<'r, T> {
        let other = other.as_ref();

        let requires_grad = self.requires_grad() || other.requires_grad();
        let grad_fn = if requires_grad { DotBackwards::new(self, other) } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts(self.array.dot(&other.array), requires_grad, grad_fn) }
    }

    /// Calculates the matrix product of two tensors.
    ///
    /// - If both tensors are 1D, then their dot product is returned.
    /// - If both tensors are 2D, then their matrix product is returned.
    /// - If the first tensor is 2D and the second tensor is 1D, then the matrix-vector product is returned.
    ///
    /// # Panics
    /// - If the dimensions/shape of the tensors are incompatible
    ///
    /// # Example
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let a = Tensor::new(vec![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    /// ]);
    ///
    /// let b = Tensor::new(vec![
    ///     [7.0, 8.0],
    ///     [9.0, 10.0],
    ///     [11.0, 12.0],
    /// ]);
    ///
    /// let result = a.matmul(&b);
    /// assert_eq!(result, Tensor::new([
    ///     [58.0, 64.0],
    ///     [139.0, 154.0],
    /// ]));
    /// ```
    pub fn matmul<'r>(&self, other: impl AsRef<Tensor<'a, T>>) -> Tensor<'r, T> {
        let other = other.as_ref();

        if self.ndims() == 1 && other.ndims() == 1 {
            return self.dot(other);
        }

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = self.array.matmul(&other.array);

        let grad_fn = if requires_grad {
            if self.ndims() == 2 && other.ndims() == 1 {
                MatrixVecBackwards::new(self, other)
            } else if self.ndims() == 2 && other.ndims() == 2 {
                MatrixProductBackwards::new(self, other)
            } else {
                panic!("this should never happen")
            }
        } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts(result, requires_grad, grad_fn) }
    }

    /// Performs batch matrix multiplication on 3D tensors.
    ///
    /// The shape of the resulting ndarray will be `[batch_size, self.shape()[1], other.shape()[2]]`,
    /// where `batch_size` is the shared first dimension of both input tensors.
    ///
    /// # Panics
    /// - If either tensor is not 3D
    /// - If the tensors do not have dimensions compatible for batch matrix multiplication.
    ///
    /// # Example
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let arr1 = Tensor::<f32>::rand([3, 2, 4]); // 3 batches of 2x4 matrices
    /// let arr2 = Tensor::<f32>::rand([3, 4, 5]); // 3 batches of 4x5 matrices
    /// let result = arr1.bmm(&arr2);
    /// assert_eq!(result.shape(), [3, 2, 5]); // result is 3 batches of 2x5 matrices
    /// ```
    pub fn bmm<'r>(&self, other: impl AsRef<Tensor<'a, T>>) -> Tensor<'r, T> {
        let other = other.as_ref();
        
        let requires_grad = self.requires_grad() || other.requires_grad();
        let grad_fn = if requires_grad { BMMBackwards::new(self, other) } else { NoneBackwards::new() };

        unsafe { Tensor::from_raw_parts(self.array.bmm(&other.array), requires_grad, grad_fn) }
    }
}
