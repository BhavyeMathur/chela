use crate::dtype::RawDataType;
use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArray};
use crate::traits::methods::StridedMemory;

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
