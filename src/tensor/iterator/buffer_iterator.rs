use crate::dtype::RawDataType;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::Tensor;


pub struct BufferIterator<T: RawDataType> {
    ptr: *mut T,
    indices: FlatIndexGenerator,
}

impl<T: RawDataType> BufferIterator<T> {
    pub(crate) fn from(tensor: &Tensor<T>) -> Self {
        Self {
            ptr: tensor.ptr.as_ptr(),
            indices: FlatIndexGenerator::from(&tensor.shape, &tensor.stride),
        }
    }

    pub(crate) unsafe fn from_reshaped_view(tensor: &Tensor<T>, shape: &[usize], stride: &[usize]) -> Self {
        Self {
            ptr: tensor.ptr.as_ptr(),
            indices: FlatIndexGenerator::from(shape, stride),
        }
    }

    #[inline(always)]
    fn advance_by(&mut self, n: usize) {
        self.indices.advance_by(n);
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
