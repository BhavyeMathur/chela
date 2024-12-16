use crate::data_buffer::{DataOwned, DataView};
use crate::dtype::RawDataType;
use std::ops::Index;
use std::ptr::NonNull;

pub trait DataBuffer: Index<usize> {
    type DType: RawDataType;

    fn len(&self) -> usize;

    fn ptr(&self) -> NonNull<Self::DType>;

    fn const_ptr(&self) -> *const Self::DType;

    fn to_view(&self) -> DataView<Self::DType>;

    // fn clone(&self) -> DataOwned<Self::DType>;
}

// Two kinds of data buffers
// DataOwned: owns its data & responsible for cleaning it up
// DataView: reference to data owned by another buffer

impl<T: RawDataType> DataBuffer for DataOwned<T> {
    type DType = T;

    fn len(&self) -> usize {
        self.len
    }

    fn ptr(&self) -> NonNull<T> {
        self.ptr
    }

    fn const_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn to_view(&self) -> DataView<T> {
        let ptr = self.ptr;
        let len = self.len;
        DataView { ptr, len }
    }
}
impl<T: RawDataType> DataBuffer for DataView<T> {
    type DType = T;

    fn len(&self) -> usize {
        self.len
    }

    fn ptr(&self) -> NonNull<T> {
        self.ptr
    }

    fn const_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn to_view(&self) -> DataView<T> {
        (*self).clone()
    }
}
