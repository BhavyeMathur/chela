use crate::dtype::RawDataType;
use crate::ndarray::flags::NdArrayFlags;
use crate::slice::update_flags_with_contiguity;
use crate::{NdArray, Reshape, StridedMemory};
use crate::common::constructors::Constructors;

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Returns a 1D copy of a flattened multidimensional ndarray.
    ///
    /// If copying the data is not desirable, it may be possible to return a view.
    /// See `NdArray::ravel()`.
    ///
    /// # Examples
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let ndarray = NdArray::new([[1, 2, 3], [4, 5, 6]]);
    /// let flat_array = ndarray.flatten();
    /// assert_eq!(flat_array, NdArray::new([1, 2, 3, 4, 5, 6]));
    /// ```
    pub fn flatten(&self) -> NdArray<'static, T> {
        unsafe {
            NdArray::from_contiguous_owned_buffer(vec![self.size()], self.clone_data())
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

            _marker: self._marker,
        }
    }
}

impl<T: RawDataType> Reshape<T> for NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    unsafe fn reshaped_view(mut self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
        let flags = update_flags_with_contiguity(self.flags, &shape, &stride);

        // prevent ndarray's data from being deallocated once this method ends
        self.flags -= NdArrayFlags::Owned;

        NdArray {
            ptr: self.ptr,
            len: self.len,
            capacity: 0,

            shape,
            stride,
            flags,

            _marker: Default::default(),
        }
    }
}

impl<'a, T: RawDataType> Reshape<T> for &'a NdArray<'a, T> {
    type Output = NdArray<'a, T>;

    unsafe fn reshaped_view(self, shape: Vec<usize>, stride: Vec<usize>) -> Self::Output {
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

            _marker: Default::default(),
        }
    }
}
