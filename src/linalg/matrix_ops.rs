use crate::axis::AxisType;
use crate::einsum::einsum_into_ptr;
use crate::linalg::sum_of_products::SumOfProductsType;
use crate::{Axis, IntegerDataType, NumericDataType, RawDataType, NdArray, StridedMemory};
use std::cmp::min;

impl<'a, T: MatrixOps> NdArray<'a, T> {
    /// Calculates the matrix product of two tensors.
    ///
    /// - If both tensors are 1D, then their dot product is returned.
    /// - If both tensors are 2D, then their matrix product is returned.
    /// - If the first ndarray is 2D and the second ndarray is 1D, then the matrix-vector product is returned.
    ///
    /// # Panics
    /// - If the dimensions/shape of the tensors is incompatible
    ///
    /// # Example
    /// ```
    /// # use chela::NdArray;
    ///
    /// let a = NdArray::from(vec![
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    /// let b = NdArray::from(vec![
    ///     [7, 8],
    ///     [9, 10],
    ///     [11, 12],
    /// ]);
    ///
    /// let result = a.matmul(&b);
    /// assert_eq!(result, NdArray::from(vec![
    ///     [58, 64],
    ///     [139, 154],
    /// ]));
    /// ```
    pub fn matmul<'r>(&self, other: impl AsRef<NdArray<'a, T>>) -> NdArray<'r, T> {
        let other = other.as_ref();

        if self.ndims() == 1 && other.ndims() == 1 {
            return self.dot(other);
        }

        if self.ndims() == 2 && other.ndims() == 1 {
            assert_eq!(self.shape()[1], other.shape()[0], "mismatched shape for matrix-vector product: {:?} and {:?})", self.shape(), other.shape());
            return unsafe { <T as MatrixOps>::matrix_vector_product(self, other) };
        }

        if self.ndims() == 2 && other.ndims() == 2 {
            assert_eq!(self.shape()[1], other.shape()[0], "mismatched shape for matrix-matrix product: {:?} and {:?})", self.shape(), other.shape());

            let requires_grad = self.requires_grad() || other.requires_grad();
            let output_shape = [self.shape()[0], other.shape()[1]];

            let result = NdArray::zeros_requires_grad(output_shape, requires_grad);
            unsafe { <T as MatrixOps>::matrix_matrix_product(self, other, result.stride(), result.mut_ptr()) };
            return result;
        }

        panic!("matmul requires a tensor with 1 or 2 dimensions");
    }

    /// Performs batch matrix multiplication on 3D tensors.
    ///
    /// The shape of the resulting ndarray will be `[batch_size, self.shape()[1], other.shape()[2]]`,
    /// where `batch_size` is the shared first dimension of both input tensors.
    ///
    /// # Panics
    /// - If either ndarray is not 3D
    /// - If the tensors do not have dimensions compatible for batch matrix multiplication.
    ///
    /// # Example
    /// ```rust
    /// # use chela::*;
    /// let tensor_a = NdArray::<f32>::rand([3, 2, 4]); // 3 batches of 2x4 matrices
    /// let tensor_b = NdArray::<f32>::rand([3, 4, 5]); // 3 batches of 4x5 matrices
    /// let result = tensor_a.bmm(&tensor_b);
    /// assert_eq!(result.shape(), [3, 2, 5]); // result is 3 batches of 2x5 matrices
    /// ```
    pub fn bmm<'r>(&self, other: impl AsRef<NdArray<'a, T>>) -> NdArray<'r, T> {
        let other = other.as_ref();
        assert_eq!(self.ndims(), 3, "batch matrix multiplication requires 3D tensors");
        assert_eq!(other.ndims(), 3, "batch matrix multiplication requires 3D tensors");
        assert_eq!(self.len(), other.len(), "incompatible batch sizes for batch matrix multiplication: {:?} and {:?})", self.shape(), other.shape());

        let requires_grad = self.requires_grad() || other.requires_grad();
        let output_shape = [self.len(), self.shape()[1], other.shape()[2]];

        let result = NdArray::zeros_requires_grad(output_shape, requires_grad);
        unsafe { <T as MatrixOps>::batch_matrix_matrix_product(self, other, result.stride(), result.mut_ptr()); }
        result
    }
}

impl<'a, T: SumOfProductsType> NdArray<'a, T> {
    /// Calculates the dot product of two 1D tensors.
    ///
    /// # Panics
    /// - Panics if either ndarray is not 1D
    /// - Panics if the lengths of the two tensors are not equal
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    /// let tensor1 = NdArray::from([1, 2, 3]);
    /// let tensor2 = NdArray::from([4, 5, 6]);
    /// let result = tensor1.dot(tensor2);
    /// assert_eq!(result.value(), 32); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn dot<'b, 'r>(&self, other: impl AsRef<NdArray<'b, T>>) -> NdArray<'r, T> {
        let other = other.as_ref();
        assert_eq!(self.ndims(), 1, "dot product requires a tensor with 1 dimension");
        assert_eq!(other.ndims(), 1, "dot product requires a tensor with 1 dimension");
        assert_eq!(self.len(), other.len(), "dot product requires tensors with the same length");

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = NdArray::scalar_requires_grad(T::default(), requires_grad);

        unsafe {
            <T as SumOfProductsType>::sum_of_products_in_strides_n_n_out_stride_0(&[self.mut_ptr(), other.mut_ptr(), result.mut_ptr()],
                                                                                  &[self.stride()[0], other.stride()[0], 0],
                                                                                  self.len())
        };

        result
    }
}

impl<'a, T: NumericDataType> NdArray<'a, T> {
    /// Returns the trace of the ndarray along its first 2 axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(ndarray.trace(), NdArray::scalar(1 + 5 + 9));
    pub fn trace<'r>(&self) -> NdArray<'r, T> {
        self.offset_trace(0)
    }

    /// Returns the sum of an offset ndarray diagonal along its first 2 axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(ndarray.offset_trace(-1), NdArray::scalar(4 + 8));
    pub fn offset_trace<'r>(&self, offset: isize) -> NdArray<'r, T> {
        self.offset_trace_along(offset, 0, 1)
    }

    /// Returns the trace of an ndarray along the specified axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(ndarray.trace_along(0, 1), NdArray::scalar(1 + 5 + 9));
    pub fn trace_along<'r>(&self, axis1: impl AxisType, axis2: impl AxisType) -> NdArray<'r, T> {
        self.offset_trace_along(0, axis1, axis2)
    }

    /// Returns the sum of an offset ndarray diagonal along the specified axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(ndarray.offset_trace_along(1, 0, 1), NdArray::scalar(2 + 6));
    pub fn offset_trace_along<'r>(&self, offset: isize, axis1: impl AxisType, axis2: impl AxisType) -> NdArray<'r, T> {
        let diagonal = self.offset_diagonal_along(offset, axis1, axis2);
        diagonal.sum_along(-1)
    }
}

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Returns a diagonal view of the ndarray along its first 2 axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = ndarray.diagonal();
    /// assert_eq!(diagonal, NdArray::from([1, 5, 9]));
    pub fn diagonal(&'a self) -> NdArray<'a, T> {
        self.diagonal_along(0, 1)
    }

    /// Returns an offset diagonal view of the ndarray along its first 2 axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = ndarray.offset_diagonal(1);
    /// assert_eq!(diagonal, NdArray::from([2, 6]));
    pub fn offset_diagonal(&'a self, offset: isize) -> NdArray<'a, T> {
        self.offset_diagonal_along(offset, 0, 1)
    }

    /// Returns a diagonal view of the ndarray along the specified axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = ndarray.diagonal_along(Axis(0), Axis(1));  // or .diagonal_along(0, 1)
    /// assert_eq!(diagonal, NdArray::from([1, 5, 9]));
    pub fn diagonal_along(&'a self, axis1: impl AxisType, axis2: impl AxisType) -> NdArray<'a, T> {
        self.offset_diagonal_along(0, axis1, axis2)
    }

    /// Returns an offset diagonal view of the ndarray along the specified axes.
    ///
    /// # Panics
    /// - if the ndarray has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = ndarray.offset_diagonal_along(-1, Axis(0), Axis(1));  // or .offset_diagonal_along(-1, 0, 1)
    /// assert_eq!(diagonal, NdArray::from([4, 8]));
    pub fn offset_diagonal_along(&'a self, offset: isize, axis1: impl AxisType, axis2: impl AxisType) -> NdArray<'a, T> {
        assert!(self.ndims() >= 2, "diagonals require a tensor with at least 2 dimensions");

        let axis1 = axis1.get_absolute(self.ndims());
        let axis2 = axis2.get_absolute(self.ndims());

        assert_ne!(axis1, axis2, "axis1 and axis2 cannot be the same");


        // get the new dimensions and strides of the two axes

        let mut dim1 = self.shape()[axis1];
        let mut dim2 = self.shape()[axis2];

        let stride1 = self.stride()[axis1];
        let stride2 = self.stride()[axis2];


        // modify the dimensions and data pointer based on offset

        let ptr_offset = if offset >= 0 {
            let offset = offset as usize;
            if offset >= dim2 {
                panic!("invalid offset {} for axis with dimension {}", offset, dim2);
            }

            dim2 -= offset;
            offset * stride2
        } else {
            let offset = -offset as usize;
            if offset >= dim1 {
                panic!("invalid offset -{} for axis with dimension {}", offset, dim1);
            }

            dim1 -= offset;
            offset * stride1
        };


        // compute the resultant shape and stride

        let mut result_shape = Vec::with_capacity(self.ndims() - 1);
        let mut result_stride = Vec::with_capacity(self.ndims() - 1);

        for axis in 0..self.ndims() {
            if axis == axis1 || axis == axis2 {
                continue;
            }

            result_shape.push(self.shape()[axis]);
            result_stride.push(self.stride()[axis]);
        }

        result_shape.push(min(dim1, dim2));
        result_stride.push(stride1 + stride2);

        // create and return the diagonal view
        unsafe { self.reshaped_view_with_offset(ptr_offset, result_shape, result_stride) }
    }
}


trait MatrixOps: SumOfProductsType {
    /// Performs an unchecked batched matrix-matrix product operation
    /// and writes the result to the given pointer
    ///
    /// # Safety
    ///
    /// - The dimensions of `lhs` and `rhs` must be `(b, i, j)` and `(b, j, k)`.
    /// - `result` must point to a valid data buffer with dimension `(b, i, k)`
    /// - `result_stride` must represent a valid layout for the results buffer with 
    ///   the last 2 dimensions being contiguous.
    /// - `result` must not overlap with `lhs` or `rhs`.
    unsafe fn batch_matrix_matrix_product<'a>(lhs: &NdArray<'a, Self>,
                                              rhs: &NdArray<'a, Self>,
                                              result_stride: &[usize],
                                              mut result: *mut Self) {
        let mut lhs_slice = lhs.slice_along(Axis(0), 0);
        let mut rhs_slice = rhs.slice_along(Axis(0), 0);

        for _ in 0..lhs.len() {
            Self::matrix_matrix_product(&lhs_slice, &rhs_slice, &result_stride[1..], result);

            result = result.add(result_stride[0]);
            lhs_slice.offset_ptr(lhs.stride()[0] as isize);
            rhs_slice.offset_ptr(rhs.stride()[0] as isize);
        }
    }

    /// Performs an unchecked matrix-matrix product and writes the result to the given pointer
    ///
    /// # Safety
    ///
    /// - The dimensions of `lhs` and `rhs` must be `(i, j)` and `(j, k)`.
    /// - `result` must point to a valid data buffer with dimension `(i, k)`.
    /// - `result_stride` must represent a contiguous layout for the results buffer.
    /// - `result` must not overlap with `lhs` or `rhs`.
    unsafe fn matrix_matrix_product<'a>(lhs: &NdArray<'a, Self>,
                                        rhs: &NdArray<'a, Self>,
                                        result_stride: &[usize],
                                        result: *mut Self)
    {
        einsum_into_ptr([lhs, rhs], (["ij", "jk"], "ik"), result_stride, result)
    }

    /// Performs an unchecked matrix-vector product and returns the result.
    ///
    /// # Safety
    ///
    /// - The dimensions of `lhs` and `rhs` must be `(i, j)` and `(j)`.
    unsafe fn matrix_vector_product<'a, 'b, 'r>(matrix: &NdArray<'a, Self>,
                                                vector: &NdArray<'b, Self>) -> NdArray<'r, Self> {
        let rows = matrix.shape()[0];
        let cols = matrix.shape()[1];
        let mut result = vec![Self::default(); rows];

        let strides = &[matrix.stride()[1], vector.stride()[0], 0];

        let mut matrix_row = matrix.mut_ptr();
        let mut dst = result.as_mut_ptr();

        let requires_grad = matrix.requires_grad() || vector.requires_grad();

        for _ in 0..rows {
            Self::sum_of_products_in_strides_n_n_out_stride_0(&[matrix_row, vector.mut_ptr(), dst], strides, cols);
            matrix_row = matrix_row.add(matrix.stride()[0]);
            dst = dst.add(1);
        }

        NdArray::from_contiguous_owned_buffer(vec![rows], result, requires_grad, false)
    }
}

impl<T: IntegerDataType> MatrixOps for T {}

impl MatrixOps for f32 {
    #[cfg(use_apple_blas)]
    unsafe fn matrix_matrix_product<'a>(lhs: &NdArray<'a, Self>,
                                        rhs: &NdArray<'a, Self>,
                                        result_stride: &[usize],
                                        result: *mut Self) {
        use crate::accelerate::cblas::{cblas_sgemm, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};

        // BLAS does not support matrices that don't have contiguous rows
        if lhs.stride()[1] != 1 || rhs.stride()[1] != 1 {
            return einsum_into_ptr([lhs, rhs], (["ij", "jk"], "ik"), result_stride, result);
        }

        let m = lhs.shape()[0];
        let n = rhs.shape()[1];

        unsafe {
            cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                        m as i32, n as i32, lhs.shape()[1] as i32,
                        1.0,
                        lhs.mut_ptr(), lhs.stride()[0] as i32,
                        rhs.mut_ptr(), rhs.stride()[0] as i32,
                        0.0, result, n as i32);
        }
    }

    #[cfg(all(use_apple_blas, not(use_neon_simd)))]
    unsafe fn matrix_vector_product<'a, 'b, 'r>(matrix: &NdArray<'a, Self>,
                                                vector: &NdArray<'b, Self>) -> NdArray<'r, Self> {
        use crate::accelerate::cblas::{cblas_sgemv, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};
        use crate::einsum;

        // BLAS does not support matrices that don't have contiguous rows
        if matrix.stride()[1] != 1 {
            return einsum([matrix, vector], (["ij", "j"], "i"));
        }

        let rows = matrix.shape()[0];
        let cols = matrix.shape()[1] as i32;
        let mut result = vec![Self::default(); rows];

        let requires_grad = matrix.requires_grad() || vector.requires_grad();

        unsafe {
            cblas_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
                        rows as i32, cols, 1.0, matrix.ptr(), matrix.stride()[0] as i32,
                        vector.ptr(), vector.stride()[0] as i32,
                        0.0, result.as_mut_ptr(), 1
            );

            NdArray::from_contiguous_owned_buffer(vec![rows], result, requires_grad)
        }
    }
}

impl MatrixOps for f64 {
    #[cfg(use_apple_blas)]
    unsafe fn matrix_matrix_product<'a>(lhs: &NdArray<'a, Self>,
                                        rhs: &NdArray<'a, Self>,
                                        result_stride: &[usize],
                                        result: *mut Self) {
        use crate::accelerate::cblas::{cblas_dgemm, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};

        // BLAS does not support matrices that don't have contiguous rows
        if lhs.stride()[1] != 1 || rhs.stride()[1] != 1 {
            return einsum_into_ptr([lhs, rhs], (["ij", "jk"], "ik"), result_stride, result);
        }

        let m = lhs.shape()[0];
        let n = rhs.shape()[1];

        unsafe {
            cblas_dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                        m as i32, n as i32, lhs.shape()[1] as i32,
                        1.0,
                        lhs.mut_ptr(), lhs.stride()[0] as i32,
                        rhs.mut_ptr(), rhs.stride()[0] as i32,
                        0.0, result, n as i32);
        }
    }

    #[cfg(all(use_apple_blas, not(use_neon_simd)))]
    unsafe fn matrix_vector_product<'a, 'b, 'r>(matrix: &NdArray<'a, Self>,
                                                vector: &NdArray<'b, Self>) -> NdArray<'r, Self> {
        use crate::accelerate::cblas::{cblas_dgemv, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};
        use crate::einsum;

        // BLAS does not support matrices that don't have contiguous rows
        if matrix.stride()[1] != 1 {
            return einsum([matrix, vector], (["ij", "j"], "i"));
        }

        let rows = matrix.shape()[0];
        let cols = matrix.shape()[1] as i32;
        let mut result = vec![Self::default(); rows];

        let requires_grad = matrix.requires_grad() || vector.requires_grad();

        unsafe {
            cblas_dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
                        rows as i32, cols, 1.0, matrix.ptr(), matrix.stride()[0] as i32,
                        vector.ptr(), vector.stride()[0] as i32,
                        0.0, result.as_mut_ptr(), 1
            );

            NdArray::from_contiguous_owned_buffer(vec![rows], result, requires_grad)
        }
    }
}
