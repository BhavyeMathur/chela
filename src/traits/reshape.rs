use crate::{AxisType, NdArray, StridedMemory, RawDataType};
use crate::util::to_vec::ToVec;

pub(crate) trait ReshapeImpl<'a, T: RawDataType>: StridedMemory {
    /// Provides a non-owning view of the ndarray with the specified shape and stride.
    /// The data pointed to by the view is shared with the original ndarray.
    ///
    /// # Safety
    /// - Ensure the memory layout referenced by `shape`, and `stride` is valid and owned
    ///   by the original ndarray.
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> NdArray<'a, T>;
}

pub trait Reshape<'a, T: RawDataType>: ReshapeImpl<'a, T> {
    /// Provides a non-owning view of the ndarray that shares its data with the original ndarray.
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from(vec![1, 2, 3, 4]);
    /// let view = (&ndarray).view();
    /// assert!(view.is_view())
    /// ```
    fn view(self) -> NdArray<'a, T> {
        let shape = self.shape().to_vec();
        let stride = self.stride().to_vec();
        unsafe { self.reshaped_view(shape, stride) }
    }

    /// Reshapes the ndarray into the specified shape.
    ///
    /// This method returns a view.
    ///
    /// # Panics
    ///
    /// * If the total number of elements in the current ndarray does not match the
    ///   total number of elements in `new_shape`.
    ///
    /// # Example
    ///
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from([[4, 5], [6, 7], [8, 9]]);  // shape is [3, 2]
    /// let reshaped_array = ndarray.reshape([1, 2, 3]);
    /// assert_eq!(reshaped_array, NdArray::from([[[4, 5, 6], [7, 8, 9]]]));
    ///
    /// let ndarray = NdArray::from([0, 1, 2, 3]);
    /// let reshaped_array = (&ndarray ).reshape([2, 2]);  // reshape without consuming ndarray
    /// assert_eq!(ndarray.shape(), &[4]);
    /// assert_eq!(reshaped_array, NdArray::from([[0, 1], [2, 3]]));
    /// ```
    fn reshape(self, new_shape: impl ToVec<usize>) -> NdArray<'a, T> {
        let new_shape = new_shape.to_vec();

        if self.size() != new_shape.iter().product() {
            panic!("total number of elements must not change during reshape");
        }

        let mut new_stride = vec![0; new_shape.len()];
        let mut acc = 1;
        for (i, dim) in new_shape.iter().rev().enumerate() {
            new_stride[new_shape.len() - 1 - i] = acc;
            acc *= *dim;
        }

        unsafe { self.reshaped_view(new_shape, new_stride) }
    }

    /// Removes all singleton dimensions (dimensions of size 1) from the ndarray's shape.
    ///
    /// This method returns a view.
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from([[[[1], [3]], [[1], [4]]]]);  // shape [1, 2, 2, 1]
    /// let squeezed = ndarray.squeeze();
    /// assert_eq!(squeezed, NdArray::from([[1, 3], [1, 4]]));  // shape [2, 2]
    ///
    /// let ndarray = NdArray::from([[3], [5], [7], [9]]);
    /// let squeezed = (&ndarray ).squeeze();  // squeeze without consuming ndarray
    /// assert_eq!(ndarray.shape(), &[4, 1]);
    /// assert_eq!(squeezed, NdArray::from([3, 5, 7, 9]));
    /// ```
    fn squeeze(self) -> NdArray<'a, T> {
        let mut shape = self.shape().to_vec();
        let mut stride = self.stride().to_vec();

        (shape, stride) = shape.iter()
                               .zip(stride.iter())
                               .filter(|&(&axis_length, _)| axis_length != 1)
                               .unzip();

        unsafe { self.reshaped_view(shape, stride) }
    }

    /// Adds a singleton dimension (dimensions of size 1) to the ndarray at the specified axis.
    ///
    /// This method returns a view.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from([2, 3]);  // shape is [2]
    /// let unsqueezed = ndarray.unsqueeze(-1);  // add dimension after the last axis
    /// assert_eq!(unsqueezed.shape(), &[2, 1]);
    ///
    /// let ndarray = NdArray::from([[1, 2, 3], [9, 8, 7]]);  // shape is [2, 3]
    /// let unsqueezed = (&ndarray ).unsqueeze(1);  // unsqueeze without consuming ndarray
    /// assert_eq!(ndarray.shape(), &[2, 3]);
    /// assert_eq!(unsqueezed.shape(), &[2, 1, 3]);
    /// ```
    fn unsqueeze(self, axis: impl AxisType) -> NdArray<'a, T> {
        let axis = axis.as_absolute(self.ndims() + 1);

        let mut shape = self.shape().to_vec();
        let mut stride = self.stride().to_vec();

        if axis == self.ndims() {
            shape.push(1);
            stride.push(1)
        } else {
            shape.insert(axis, 1);
            stride.insert(axis, stride[axis] * shape[axis + 1]);
        }

        unsafe { self.reshaped_view(shape, stride) }
    }

    /// Transposes the array along the first 2 dimensions.
    ///
    /// # Panics
    /// - If the array is 1-dimensional or a scalar.
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    ///
    /// let array = NdArray::from([[2, 3, 4], [10, 20, 30]]);
    ///
    /// let transposed = array.T();
    /// assert_eq!(transposed, NdArray::from([[2, 10], [3, 20], [4, 30]]));
    /// ```
    #[allow(non_snake_case)]
    fn T(self) -> NdArray<'a, T> {
        self.transpose(0, 1)
    }

    /// Returns a transposed version of this `NdArray`, swapping the specified axes.
    ///
    /// # Panics
    /// - If `axis1` or `axis2` are out of bounds
    ///
    /// # Examples
    /// ```rust
    /// # use chela::*;
    ///
    /// let array = NdArray::from([[2, 3, 4], [10, 20, 30]]);
    ///
    /// let transposed = array.transpose(0, 1);
    /// assert_eq!(transposed, NdArray::from([[2, 10], [3, 20], [4, 30]]));
    /// ```
    fn transpose(self, axis1: impl AxisType, axis2: impl AxisType) -> NdArray<'a, T> {
        let axis1 = axis1.as_absolute(self.ndims());
        let axis2 = axis2.as_absolute(self.ndims());

        let mut shape = self.shape().to_vec();
        let mut stride = self.stride().to_vec();

        shape.swap(axis1, axis2);
        stride.swap(axis1, axis2);

        unsafe { self.reshaped_view(shape, stride) }
    }
}

impl<'a, T: RawDataType> Reshape<'a, T> for &'a NdArray<'a, T> {}
impl<T: RawDataType> Reshape<'static, T> for NdArray<'static, T> {}
