use crate::dtype::RawDataType;
use crate::slice::update_flags_with_contiguity;
use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArray, StridedMemory};
use crate::traits::ReshapeImpl;

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Returns a 1D copy of a flattened multidimensional ndarray.
    ///
    /// If copying the data is not desirable, it may be possible to return a view.
    /// See `NdArray::ravel()`.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::from([[1, 2, 3], [4, 5, 6]]);
    /// let flat_array = ndarray.flatten();
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

    /// Provides a non-owning view of the ndarray with the specified shape and stride.
    /// The data pointed to by the view is shared with the original ndarray
    /// but offset by the specified amount.
    ///
    /// # Safety
    /// - Ensure the memory referenced by `offset`, `shape`, and `stride` is valid and owned
    ///   by the original ndarray.
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
