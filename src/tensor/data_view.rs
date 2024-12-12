use crate::tensor::data_owned::DataOwned;
use crate::tensor::dtype::RawDataType;
use std::ptr::NonNull;

#[derive(Debug, Clone)]
pub(crate) struct DataView<T: RawDataType> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T: RawDataType> From<&DataOwned<T>> for DataView<T> {
    fn from(value: &DataOwned<T>) -> Self {
        Self {
            ptr: value.ptr,
            len: value.len,
        }
    }
}
