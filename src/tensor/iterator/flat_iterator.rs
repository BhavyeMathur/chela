use crate::buffer_iterator::BufferIteratorMut;
use crate::dtype::RawDataType;
use crate::Tensor;

pub struct FlatIterator<T: RawDataType> {
    buffer_iterator: BufferIteratorMut<T>,
}

impl<T: RawDataType> FlatIterator<T> {
    pub(super) fn from(tensor: &Tensor<T>) -> Self {
        Self {
            buffer_iterator: BufferIteratorMut::from(tensor),
        }
    }
}

impl<T: RawDataType> Iterator for FlatIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.buffer_iterator.next() {
            None => None,
            Some(ptr) => Some(unsafe { *ptr })
        }
    }
}
