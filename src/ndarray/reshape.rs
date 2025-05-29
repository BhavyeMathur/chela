use crate::dtype::RawDataType;
use crate::slice::update_flags_with_contiguity;
use crate::ndarray::flags::NdArrayFlags;
use crate::util::to_vec::ToVec;
use crate::{AxisType, NdArray, NdArrayMethods};

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Returns a 1D copy of a flattened multidimensional ndarray .
    ///
    /// If copying the data is not desirable, it may be possible to return a view.
    /// See `NdArray::ravel()`.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from([[1, 2, 3], [4, 5, 6]]);
    /// let flat_array = ndarray .flatten();
    /// assert_eq!(flat_array, NdArray::from([1, 2, 3, 4, 5, 6]));
    /// ```
    pub fn flatten(&self) -> NdArray<'static, T> {
        unsafe {
            NdArray::from_contiguous_owned_buffer(vec![self.size()],
                                                  self.clone_data(),
                                                  self.requires_grad(),
                                                  false)
        }
    }

    /// Provides a non-owning view of the ndarray that shares its data with the original ndarray .
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from(vec![1, 2, 3, 4]);
    /// let view = ndarray .view();
    /// assert!(view.is_view())
    /// ```
    pub fn view(&'a self) -> NdArray<'a, T> {
        unsafe { self.reshaped_view(self.shape.clone(), self.stride.clone()) }
    }

    /// Provides a non-owning view of the ndarray with the specified shape and stride.
    /// The data pointed to by the view is shared with the original ndarray
    /// but offset by the specified amount.
    ///
    /// # Safety
    /// - Ensure the memory referenced by `offset`, `shape`, and `stride` is valid and owned
    ///   by the original ndarray .
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

pub(crate) trait ReshapeImpl<'a, T: RawDataType>: NdArrayMethods {
    /// Provides a non-owning view of the ndarray with the specified shape and stride.
    /// The data pointed to by the view is shared with the original ndarray .
    ///
    /// # Safety
    /// - Ensure the memory layout referenced by `shape`, and `stride` is valid and owned
    ///   by the original ndarray .
    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> NdArray<'a, T>;
}

pub trait Reshape<'a, T: RawDataType>: ReshapeImpl<'a, T> {
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
    /// let reshaped_array = ndarray .reshape([1, 2, 3]);
    /// assert_eq!(reshaped_array, NdArray::from([[[4, 5, 6], [7, 8, 9]]]));
    ///
    /// let ndarray = NdArray::from([0, 1, 2, 3]);
    /// let reshaped_array = (&ndarray ).reshape([2, 2]);  // reshape without consuming ndarray
    /// assert_eq!(ndarray .shape(), &[4]);
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
    /// let squeezed = ndarray .squeeze();
    /// assert_eq!(squeezed, NdArray::from([[1, 3], [1, 4]]));  // shape [2, 2]
    ///
    /// let ndarray = NdArray::from([[3], [5], [7], [9]]);
    /// let squeezed = (&ndarray ).squeeze();  // squeeze without consuming ndarray
    /// assert_eq!(ndarray .shape(), &[4, 1]);
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
    /// let unsqueezed = ndarray .unsqueeze(-1);  // add dimension after the last axis
    /// assert_eq!(unsqueezed.shape(), &[2, 1]);
    ///
    /// let ndarray = NdArray::from([[1, 2, 3], [9, 8, 7]]);  // shape is [2, 3]
    /// let unsqueezed = (&ndarray ).unsqueeze(1);  // unsqueeze without consuming ndarray
    /// assert_eq!(ndarray .shape(), &[2, 3]);
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

        // prevent ndarray's data from being deallocated once this method ends
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
