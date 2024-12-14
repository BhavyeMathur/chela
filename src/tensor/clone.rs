use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::{Tensor, TensorBase};

impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType = T>,
    T: RawDataType,
{
    pub fn clone(&self) -> Tensor<T> {
        TensorBase {
            data: self.data.clone(),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            ndims: self.ndims,
        }
    }
}
