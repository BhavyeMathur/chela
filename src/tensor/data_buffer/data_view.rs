use crate::tensor::data_buffer::{DataBuffer, DataOwned};
use crate::tensor::dtype::RawDataType;
use std::ops::Index;
use std::ptr::NonNull;

#[derive(Debug, Clone)]
pub struct DataView<T: RawDataType> {
    pub(super) ptr: NonNull<T>,
    pub(super) len: usize,
}

impl<T: RawDataType> DataView<T> {
    pub(in crate::tensor) fn from_buffer<B>(value: &B, offset: usize, len: usize) -> Self
    where
        B: DataBuffer<DType=T>,
    {
        assert!(offset + len <= value.len());
        Self {
            ptr: unsafe { value.ptr().offset(offset as isize) },
            len,
        }
    }
}

impl<T: RawDataType> From<&DataOwned<T>> for DataView<T> {
    fn from(value: &DataOwned<T>) -> Self {
        Self {
            ptr: value.ptr,
            len: value.len,
        }
    }
}

impl<T> Index<usize> for DataView<T>
where
    T: RawDataType,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        assert!(index < self.len, "Index '{index}' out of bounds"); // type implies 0 <= index
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}
