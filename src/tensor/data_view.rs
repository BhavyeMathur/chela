use crate::tensor::data_owned::DataOwned;
use crate::tensor::dtype::RawDataType;
use std::ops::Index;
use std::ptr::NonNull;

#[derive(Debug, Clone)]
pub struct DataView<T: RawDataType> {
    pub(crate) ptr: NonNull<T>,
    pub(crate) len: usize,
}

impl<T: RawDataType> DataView<T> {
    pub(crate) fn from_owned(value: &DataOwned<T>, offset: usize, len: usize) -> Self {
        assert!(offset + len <= value.len);
        Self {
            ptr: unsafe { value.ptr.offset(offset as isize) },
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
        unsafe { &*self.ptr.as_ptr().offset(index as isize) }
    }
}
