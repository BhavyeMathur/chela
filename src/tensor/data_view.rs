use std::ptr::NonNull;

use crate::tensor::dtype::RawDataType;

pub(crate) struct DataView<T: RawDataType> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}
