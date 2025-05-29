use crate::dtype::RawDataType;
use crate::slice::update_flags_with_contiguity;
use crate::ndarray::flags::NdArrayFlags;
use crate::util::to_vec::ToVec;
use crate::{AxisType, NdArray, TensorMethods};

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Returns a 1D copy of a flattened multidimensional tensor.
    ///
    /// If copying the data is not desirable, it may be possible to return a view.
    /// See `Tensor::ravel()`.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = NdArray::from([[1, 2, 3], [4, 5, 6]]);
    /// let flat_tensor = tensor.flatten();
    /// assert_eq!(flat_tensor, NdArray::from([1, 2, 3, 4, 5, 6]));
    /// ```
    pub fn flatten(&self) -> NdArray<'static, T> {
        unsafe {
            NdArray::from_contiguous_owned_buffer(vec![self.size()],
                                                  self.clone_data(),
                                                  self.requires_grad(),
                                                  false)
        }
    }

    /// Provides a non-owning view of the tensor that shares its data with the original tensor.
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = NdArray::from(vec![1, 2, 3, 4]);
    /// let view = tensor.view();
    /// assert!(view.is_view())
    /// ```
    pub fn view(&'a self) -> NdArray<'a, T> {
        unsafe { self.reshaped_view(self.shape.clone(), self.stride.clone()) }
    }

    /// Provides a non-owning view of the tensor with the specified shape and stride.
    /// The data pointed to by the view is shared with the original tensor
    /// but offset by the specified amount.
    ///
    /// # Safety
    /// - Ensure the memory referenced by `offset`, `shape`, and `stride` is valid and owned
    ///   by the original tensor.
    pub(crate) unsafe fn reshaped_view_with_offset(&'a self,
                                                   offset: usize,
                                                   shape: Vec<usize>,
                                                   stride: Vec<usize>) -> NdArray<'a, T> {
        let mut flags = update_flags_with_contiguity(self.flags, &shape, &stride);
        flags -= NdArrayFlags::UserCreated;
        flags -= NdArrayFlags::Owned;

        NdArray {
            ptr: self.ptr.add(offset),
            len: shape.iter().product(),
            capacity: 0,

            shape,
            stride,
            flags,

            grad_fn: self.grad_fn.clone(),

            _marker: self._marker,
        }
    }
}

pub(crate) trait ReshapeImpl<'a, T: RawDataType>: TensorMethods {
    /// Provides a non-owning view of the tensor with the specified shape and stride.
    /// The data pointed to by the view is shared with the original tensor.
    ///
    /// # Safety
    /// - Ensure the memory layout referenced by `shape`, and `stride` is valid and owned
    ///   by the original tensor.
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> NdArray<'a, T>;
}

pub trait Reshape<'a, T: RawDataType>: ReshapeImpl<'a, T> {
    /// Reshapes the tensor into the specified shape.
    ///
    /// This method returns a view.
    ///
    /// # Panics
    ///
    /// * If the total number of elements in the current tensor does not match the
    ///   total number of elements in `new_shape`.
    ///
    /// # Example
    ///
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = NdArray::from([[4, 5], [6, 7], [8, 9]]);  // shape is [3, 2]
    /// let reshaped_tensor = tensor.reshape([1, 2, 3]);
    /// assert_eq!(reshaped_tensor, NdArray::from([[[4, 5, 6], [7, 8, 9]]]));
    ///
    /// let tensor = NdArray::from([0, 1, 2, 3]);
    /// let reshaped_tensor = (&tensor).reshape([2, 2]);  // reshape without consuming tensor
    /// assert_eq!(tensor.shape(), &[4]);
    /// assert_eq!(reshaped_tensor, NdArray::from([[0, 1], [2, 3]]));
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

    /// Removes all singleton dimensions (dimensions of size 1) from the tensor's shape.
    ///
    /// This method returns a view.
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = NdArray::from([[[[1], [3]], [[1], [4]]]]);  // shape [1, 2, 2, 1]
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed, NdArray::from([[1, 3], [1, 4]]));  // shape [2, 2]
    ///
    /// let tensor = NdArray::from([[3], [5], [7], [9]]);
    /// let squeezed = (&tensor).squeeze();  // squeeze without consuming tensor
    /// assert_eq!(tensor.shape(), &[4, 1]);
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

    /// Adds a singleton dimension (dimensions of size 1) to the tensor at the specified axis.
    ///
    /// This method returns a view.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let tensor = NdArray::from([2, 3]);  // shape is [2]
    /// let unsqueezed = tensor.unsqueeze(-1);  // add dimension after the last axis
    /// assert_eq!(unsqueezed.shape(), &[2, 1]);
    ///
    /// let tensor = NdArray::from([[1, 2, 3], [9, 8, 7]]);  // shape is [2, 3]
    /// let unsqueezed = (&tensor).unsqueeze(1);  // unsqueeze without consuming tensor
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// assert_eq!(unsqueezed.shape(), &[2, 1, 3]);
    /// ```
    fn unsqueeze(self, axis: impl AxisType) -> NdArray<'a, T> {
        let axis = axis.get_absolute(self.ndims() + 1);

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
}

impl<'a, T: RawDataType> ReshapeImpl<'static, T> for NdArray<'a, T> {
    unsafe fn reshaped_view(mut self, shape: Vec<usize>, stride: Vec<usize>) -> NdArray<'static, T> {
        let flags = update_flags_with_contiguity(self.flags, &shape, &stride);

        // prevent tensor's data from being deallocated once this method ends
        self.flags -= NdArrayFlags::Owned;

        NdArray {
            ptr: self.ptr,
            len: shape.iter().product(),
            capacity: self.capacity,

            shape,
            stride,
            flags,

            grad_fn: self.grad_fn.clone(),

            _marker: Default::default(),
        }
    }
}

impl<'a, T: RawDataType> ReshapeImpl<'a, T> for &'a NdArray<'a, T> {
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> NdArray<'a, T> {
        let mut flags = update_flags_with_contiguity(self.flags, &shape, &stride);
        flags -= NdArrayFlags::UserCreated;
        flags -= NdArrayFlags::Owned;

        NdArray {
            ptr: self.ptr,
            len: shape.iter().product(),
            capacity: self.capacity,

            shape,
            stride,
            flags,

            grad_fn: self.grad_fn.clone(),

            _marker: Default::default(),
        }
    }
}

impl<'a, T: RawDataType> Reshape<'a, T> for &'a NdArray<'a, T> {}
impl<T: RawDataType> Reshape<'static, T> for NdArray<'static, T> {}
