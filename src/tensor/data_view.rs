use std::ptr::NonNull;

use crate::tensor::dtype::RawDataType;

pub struct DataView<T: RawDataType> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}
