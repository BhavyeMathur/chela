use crate::dtype::RawDataType;
use crate::iterator::flat_index_iterator::FlatIndexIterator;
use crate::iterator::flat_iterator::FlatIterator;
use crate::{Tensor, TensorView};
use std::ops::Range;

pub mod flat_iterator;
pub(super) mod collapse_contiguous;
pub mod flat_index_iterator;

impl<T: RawDataType> Tensor<T> {
    pub fn flat_iter(&self) -> FlatIterator<T, Range<isize>> {
        FlatIterator::from(self, 0..self.size() as isize)
    }
}

impl<T: RawDataType> TensorView<T> {
    pub fn flat_iter(&self) -> FlatIterator<T, FlatIndexIterator> {
        let indices = FlatIndexIterator::from(&self.shape, &self.stride);
        FlatIterator::from(self, indices)
    }
}
