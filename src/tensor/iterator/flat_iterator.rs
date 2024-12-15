use crate::dtype::RawDataType;
use crate::Tensor;

#[non_exhaustive]
pub struct TensorFlatIter<T>
where
    T: RawDataType,
{
    ptr: *const T,
    size: isize,
    index: isize,
}

impl<T: RawDataType> Tensor<T> {
    pub fn flat_iter(&self) -> TensorFlatIter<T> {
        TensorFlatIter::from(&self)
    }
}

impl<T: RawDataType> TensorFlatIter<T> {
    fn from(tensor: &Tensor<T>) -> Self {
        Self {
            ptr: tensor.data.ptr(),
            size: tensor.size() as isize,
            index: 0,
        }
    }
}

impl<T> Iterator for TensorFlatIter<T>
where
    T: RawDataType,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.size {
            return None;
        }

        let rvalue = unsafe { *self.ptr.offset(self.index) };
        self.index += 1;
        Some(rvalue)
    }
}
