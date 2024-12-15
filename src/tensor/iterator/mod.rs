use crate::dtype::RawDataType;
use crate::iterator::flat_iterator::FlatIterator;
use crate::iterator::shape_stride_iterator::ShapeStrideRange;
use crate::{Tensor, TensorView};
use std::ops::Range;

pub mod flat_iterator;
mod collapse_contiguous;
mod shape_stride_iterator;

impl<T: RawDataType> Tensor<T> {
    pub fn flat_iter(&self) -> FlatIterator<T, Range<isize>> {
        FlatIterator::from(self, 0..self.size() as isize)
    }
}

impl<T: RawDataType> TensorView<T> {
    pub fn flat_iter(&self) -> FlatIterator<T, ShapeStrideRange> {
        let indices = ShapeStrideRange::from(&self.shape, &self.stride);
        FlatIterator::from(self, indices)
    }
}
