use crate::dtype::RawDataType;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::Tensor;

pub struct BufferIterator<T: RawDataType> {
    ptr: *mut T,
    indices: FlatIndexGenerator,
}

impl<T: RawDataType> BufferIterator<T> {
    pub(super) fn from(tensor: &Tensor<T>) -> Self {
        Self {
            ptr: tensor.ptr.as_ptr(),
            indices: FlatIndexGenerator::from(&tensor.shape, &tensor.stride),
        }
    }
}

impl<T: RawDataType> Iterator for BufferIterator<T> {
    type Item = *mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.next() {
            None => None,
            Some(i) => Some(unsafe { self.ptr.add(i) })
        }
    }
}
