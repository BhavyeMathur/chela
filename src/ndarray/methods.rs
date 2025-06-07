use crate::dtype::RawDataType;
use crate::ndarray::flags::NdArrayFlags;
use crate::common::methods::StridedMemory;
use crate::NdArray;

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Retrieves the single value contained within an ndarray with a singular element.
    ///
    /// # Panics
    /// If the ndarray contains more than one element (i.e., it is not a scalar or an ndarray with a
    /// single element)
    ///
    /// # Example
    /// ```
    /// # use redstone::*;
    ///
    /// let ndarray = NdArray::scalar(50f32);
    /// let value = ndarray.value();
    /// assert_eq!(value, 50.0);
    /// ```
    ///
    /// # Notes
    /// This function is only meant for arrays that are guaranteed to have
    /// exactly one element. For arrays with multiple elements, consider using
    /// appropriate methods to access individual elements or slices safely.
    pub fn value(&self) -> T {
        assert_eq!(self.size(), 1, "cannot get value of an ndarray with more than one element");
        unsafe { self.ptr.read() }
    }

    /// Returns a slice of the ndarray's (flattened) data buffer
    ///
    /// # Example
    /// ```
    /// # use redstone::*;
    ///
    /// let ndarray = NdArray::new([[50, 60], [-5, -10]]);
    /// let data = ndarray.data_slice();
    /// assert_eq!(data, &[50, 60, -5, -10]);
    /// ```
    pub fn data_slice(&self) -> &'a [T] {
        assert!(self.is_contiguous(), "cannot get data slice of non-contiguous tensor");
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len) }
    }

    /// Converts an `NdArray` into its underlying data vector by flattening its dimensions.
    ///
    /// # Panics
    /// - If the ndarray does not own its data (it is a NdArray view).
    ///
    /// # Example
    /// ```
    /// # use redstone::*;
    ///
    /// let ndarray = NdArray::new([[50, 60], [-5, -10]]);
    /// let data = ndarray.into_data_vector();
    /// assert_eq!(data, vec![50, 60, -5, -10]);
    /// ```
    pub fn into_data_vector(mut self) -> Vec<T> {
        if !self.flags.contains(NdArrayFlags::Owned) {
            panic!("cannot return data vector of non-owned tensor");
        }
        assert!(self.is_contiguous(), "cannot get data vector of non-contiguous tensor");

        // ensure the vector's data is not dropped when self goes out of scope and is destroyed
        self.flags -= NdArrayFlags::Owned;

        unsafe { Vec::from_raw_parts(self.mut_ptr(), self.len, self.capacity) }
    }
}

impl<T: RawDataType> StridedMemory for NdArray<'_, T> {
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn stride(&self) -> &[usize] {
        &self.stride
    }

    #[inline]
    fn flags(&self) -> NdArrayFlags {
        self.flags
    }
}

impl<T: RawDataType> StridedMemory for &NdArray<'_, T> {
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn stride(&self) -> &[usize] {
        &self.stride
    }

    #[inline]
    fn flags(&self) -> NdArrayFlags {
        self.flags
    }
}

impl<'a, T: RawDataType> NdArray<'a, T> {
    pub(crate) unsafe fn mut_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) unsafe fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}
