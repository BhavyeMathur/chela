use crate::axes_traits::AxisType;
use crate::{NumericDataType, RawDataType, Tensor, TensorMethods, TensorNumericReduce};
use std::cmp::min;


impl<'a, T: NumericDataType> Tensor<'a, T>
where
    Tensor<'a, T>: TensorNumericReduce<T>
{
    /// Returns the trace of the tensor along its first 2 axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(tensor.trace(), Tensor::scalar(1 + 5 + 9));
    pub fn trace<'b>(&'a self) -> Tensor<'b, T> {
        self.offset_trace(0)
    }

    /// Returns the sum of an offset tensor diagonal along its first 2 axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(tensor.offset_trace(-1), Tensor::scalar(4 + 8));
    pub fn offset_trace<'b>(&'a self, offset: isize) -> Tensor<'b, T> {
        self.offset_trace_along(offset, 0, 1)
    }

    /// Returns the trace of a tensor along the specified axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(tensor.trace_along(0, 1), Tensor::scalar(1 + 5 + 9));
    pub fn trace_along<'b>(&'a self, axis1: impl AxisType, axis2: impl AxisType) -> Tensor<'b, T> {
        self.offset_trace_along(0, axis1, axis2)
    }

    /// Returns the sum of an offset tensor diagonal along the specified axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(tensor.offset_trace_along(1, 0, 1), Tensor::scalar(2 + 6));
    pub fn offset_trace_along<'b>(&'a self, offset: isize, axis1: impl AxisType, axis2: impl AxisType) -> Tensor<'b, T> {
        let diagonal = self.offset_diagonal_along(offset, axis1, axis2);
        diagonal.sum_along(-1)
    }
}

impl<T: RawDataType> Tensor<'_, T> {
    /// Returns a diagonal view of the tensor along its first 2 axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = tensor.diagonal();
    /// assert_eq!(diagonal, Tensor::from([1, 5, 9]));
    pub fn diagonal(&self) -> Tensor<T> {
        self.diagonal_along(0, 1)
    }

    /// Returns an offset diagonal view of the tensor along its first 2 axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = tensor.offset_diagonal(1);
    /// assert_eq!(diagonal, Tensor::from([2, 6]));
    pub fn offset_diagonal(&self, offset: isize) -> Tensor<T> {
        self.offset_diagonal_along(offset, 0, 1)
    }

    /// Returns a diagonal view of the tensor along the specified axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = tensor.diagonal_along(Axis(0), Axis(1));  // or .diagonal_along(0, 1)
    /// assert_eq!(diagonal, Tensor::from([1, 5, 9]));
    pub fn diagonal_along(&self, axis1: impl AxisType, axis2: impl AxisType) -> Tensor<T> {
        self.offset_diagonal_along(0, axis1, axis2)
    }

    /// Returns an offset diagonal view of the tensor along the specified axes.
    ///
    /// # Panics
    /// - if the tensor has fewer than 2 dimensions.
    /// - if `axis1` and `axis2` are the same or are out-of-bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    ///
    /// let diagonal = tensor.offset_diagonal_along(-1, Axis(0), Axis(1));  // or .offset_diagonal_along(-1, 0, 1)
    /// assert_eq!(diagonal, Tensor::from([4, 8]));
    pub fn offset_diagonal_along(&self, offset: isize, axis1: impl AxisType, axis2: impl AxisType) -> Tensor<T> {
        assert!(self.ndims() >= 2, "diagonals require a tensor with at least 2 dimensions");

        let axis1 = axis1.get_absolute(self.ndims());
        let axis2 = axis2.get_absolute(self.ndims());

        assert_ne!(axis1, axis2, "axis1 and axis2 cannot be the same");


        // get the new dimensions and strides of the two axes

        let mut dim1 = self.shape()[axis1];
        let mut dim2 = self.shape()[axis2];

        let stride1 = self.stride()[axis1];
        let stride2 = self.stride()[axis2];

        let ptr_offset;

        if offset >= 0 {
            let offset = offset as usize;
            if offset >= dim2 {
                panic!("invalid offset {} for axis with dimension {}", offset, dim2);
            }

            dim2 -= offset;
            ptr_offset = offset * stride2;
        } else {
            let offset = -offset as usize;
            if offset >= dim1 {
                panic!("invalid offset -{} for axis with dimension {}", offset, dim1);
            }

            dim1 -= offset;
            ptr_offset = offset * stride1;
        }


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
