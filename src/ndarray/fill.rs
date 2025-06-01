use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::{NdArray, StridedMemory};


/// Fills a sequence of memory locations with a given value, following a specified stride.
///
/// # Safety
/// - Ensure the memory described by `start`, `stride`, and `n` is valid.
///
/// # Parameters
/// * `start`: A mutable raw pointer to the starting location of memory to be filled.
/// * `value`: The value to write to the memory locations.
/// * `stride`: The stride (in number of elements of type `T`) between consecutive writes.
/// * `n`: The number of values to write.
unsafe fn fill_strided<T: Copy>(mut start: *mut T, value: T, stride: usize, n: usize) {
    for _ in 0..n {
        std::ptr::write(start, value);
        start = start.add(stride);
    }
}

/// Fills memory described by the layout of `shape` and `stride` with a given value.
///
/// # Safety
/// - Ensure the memory described by `start`, `shape`, and `stride` is valid.
///
/// # Parameters
/// * `start`: A mutable raw pointer to the starting location of memory to be filled.
/// * `value`: The value to write to the memory locations.
/// * `shape`: The shape along each dimension of the memory layout
/// * `stride`: The stride between elements along each dimension of the memory layout
pub(crate) unsafe fn fill_shape_and_stride<T: Copy>(mut start: *mut T, value: T, shape: &[usize], stride: &[usize]) {
    if shape.len() == 1 {
        return fill_strided(start, value, stride[0], shape[0]);
    }

    for _ in 0..shape[0] {
        fill_shape_and_stride(start, value, &shape[1..], &stride[1..]);
        start = start.add(stride[0]);
    }
}

impl<T: RawDataType> NdArray<'_, T> {
    /// Fills the entire array with a specified `value`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use chela::*;
    ///
    /// let mut arr = NdArray::new([1, 2, 4]);
    /// arr.fill(10);
    /// assert_eq!(arr, NdArray::new([10, 10, 10]));
    /// ```
    pub fn fill(&mut self, value: T) {
        if let Some(stride) = self.has_uniform_stride() {
            return unsafe { fill_strided(self.mut_ptr(), value, stride, self.size()); };
        }

        let (shape, stride) = collapse_to_uniform_stride(&self.shape, &self.stride);
        unsafe { fill_shape_and_stride(self.mut_ptr(), value, &shape, &stride); }
    }
}

impl<T: RawDataType + From<bool>> NdArray<'_, T> {
    /// Fills the entire array with a zero (or `false` if dtype is boolean).
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use chela::*;
    ///
    /// let mut arr = NdArray::new([1, 2, 4]);
    /// arr.zero();
    /// assert_eq!(arr, NdArray::new([0, 0, 0]));
    /// ```
    pub fn zero(&mut self) {
        self.fill(false.into());
    }
}
