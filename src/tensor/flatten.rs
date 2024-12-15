use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::{Tensor, TensorBase};
use crate::clone::TensorClone;

impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType=T>,
    T: RawDataType,
    TensorBase<B>: TensorClone<T>,
{
    pub fn flatten(&self) -> Tensor<T> {
        Tensor {
            data: self.copy_data(),
            shape: vec![self.size()],
            stride: vec![1],
            ndims: 1,
        }
    }
}
