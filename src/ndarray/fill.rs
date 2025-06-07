use crate::dtype::RawDataType;
use crate::{NdArray, StridedMemory};
use crate::ops::fill::Fill;


impl<T: RawDataType> NdArray<'_, T> {
    /// Fills the entire array with a specified `value`.
    ///
    /// # Example
    ///
    /// ```
    /// # use redstone::*;
    ///
    /// let mut arr = NdArray::new([1, 2, 4]);
    /// arr.fill(10);
    /// assert_eq!(arr, NdArray::new([10, 10, 10]));
    /// ```
    pub fn fill(&mut self, value: T) {
        unsafe { <T as Fill>::fill(self.mut_ptr(), self.shape(), self.stride(), self.len, value) }
    }
}

impl<T: RawDataType + From<bool>> NdArray<'_, T> {
    /// Fills the entire array with a zero (or `false` if dtype is boolean).
    ///
    /// # Example
    ///
    /// ```
    /// # use redstone::*;
    ///
    /// let mut arr = NdArray::new([1, 2, 4]);
    /// arr.zero();
    /// assert_eq!(arr, NdArray::new([0, 0, 0]));
    /// ```
    pub fn zero(&mut self) {
        self.fill(false.into());
    }
}
