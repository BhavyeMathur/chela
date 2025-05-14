use crate::dtype::RawDataType;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::Tensor;


pub struct BufferIteratorMut<T: RawDataType> {
    ptr: *mut T,
    indices: FlatIndexGenerator,
}

pub struct BufferIterator<T: RawDataType> {
    ptr: *const T,
    indices: FlatIndexGenerator,
}

unsafe impl<T: RawDataType> Sync for BufferIterator<T> {}
unsafe impl<T: RawDataType> Send for BufferIterator<T> {}

impl<T: RawDataType> BufferIteratorMut<T> {
    pub(super) fn from(tensor: &Tensor<T>) -> Self {
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
}

impl<T: RawDataType> BufferIterator<T> {
    pub(super) fn from(tensor: &Tensor<T>) -> Self {
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
}

impl<T: RawDataType> Iterator for BufferIteratorMut<T> {
    type Item = *mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.next() {
            None => None,
            Some(i) => Some(unsafe { self.ptr.add(i) })
        }
    }
}

impl<T: RawDataType> Iterator for BufferIterator<T> {
    type Item = *const T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.next() {
            None => None,
            Some(i) => Some(unsafe { self.ptr.add(i) })
        }
    }
}

impl<T: RawDataType> Clone for BufferIterator<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            indices: self.indices.clone(),
        }
    }
}
