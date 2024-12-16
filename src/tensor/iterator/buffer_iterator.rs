use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::TensorBase;

#[non_exhaustive]
pub struct BufferIterator<T, I>
where
    T: RawDataType,
    I: Iterator<Item=isize>,
{
    ptr: *const T,
    indices: I,
}

impl<T, I> BufferIterator<T, I>
where
    T: RawDataType,
    I: Iterator<Item=isize>,
{
    pub(super) fn from<B>(tensor: &TensorBase<B>, indices: I) -> Self
    where
        B: DataBuffer<DType=T>,
    {
        Self {
            ptr: tensor.data.const_ptr(),
            indices,
        }
    }
}

impl<T, I> Iterator for BufferIterator<T, I>
where
    T: RawDataType,
    I: Iterator<Item=isize>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.next() {
            None => None,
            Some(i) => Some(unsafe { *self.ptr.offset(i) })
        }
    }
}
