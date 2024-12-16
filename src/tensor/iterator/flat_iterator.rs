use crate::dtype::RawDataType;
use crate::iterator::buffer_iterator::BufferIterator;
use crate::iterator::flat_index_generator::FlatIndexGenerator;
use crate::{Tensor, TensorView};
use std::ops::Range;

pub trait FlatIterator<T: RawDataType> {
    type Indices: Iterator<Item=isize>;
    fn flat_iter(&self) -> BufferIterator<T, Self::Indices>;
}

impl<T: RawDataType> FlatIterator<T> for Tensor<T> {
    type Indices = Range<isize>;
    fn flat_iter(&self) -> BufferIterator<T, Self::Indices> {
        BufferIterator::from(self, 0..self.size() as isize)
    }
}

impl<T: RawDataType> FlatIterator<T> for TensorView<T> {
    type Indices = FlatIndexGenerator;
    fn flat_iter(&self) -> BufferIterator<T, Self::Indices> {
        let indices = FlatIndexGenerator::from(&self.shape, &self.stride);
        BufferIterator::from(self, indices)
    }
}
