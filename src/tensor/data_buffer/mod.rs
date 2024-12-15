pub(super) mod clone;
pub(super) mod data_owned;
pub(super) mod data_view;

pub(super) use crate::data_buffer::data_owned::DataOwned;
pub(super) use crate::data_buffer::data_view::DataView;

use crate::tensor::dtype::RawDataType;

use std::ops::Index;
use std::ptr::NonNull;

pub trait DataBuffer: Index<usize> {
    type DType: RawDataType;

    fn len(&self) -> usize;

    fn ptr(&self) -> NonNull<Self::DType>;

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

    fn to_view(&self) -> DataView<T> {
        (*self).clone()
    }
}
