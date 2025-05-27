use crate::buffer_iterator::BufferIterator;
use crate::dtype::RawDataType;
use crate::Tensor;

pub struct FlatIterator<T: RawDataType> {
    buffer_iterator: BufferIterator<T>,
}

impl<T: RawDataType> FlatIterator<T> {
    pub(super) fn from(tensor: &Tensor<T>) -> Self {
        Self {
            buffer_iterator: BufferIterator::from(tensor),
        }
    }
}

impl<T: RawDataType> Iterator for FlatIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer_iterator.next().map(|ptr| unsafe { *ptr })
    }
}
