use crate::data_buffer::{DataBuffer, DataOwned};
use crate::dtype::RawDataType;
use crate::{Tensor, TensorBase, TensorView};


pub(super) trait TensorClone<T: RawDataType> {
    fn copy_data(&self) -> DataOwned<T>;
}

impl<T: RawDataType> TensorClone<T> for Tensor<T> {
    fn copy_data(&self) -> DataOwned<T> {
        self.data.clone()
    }
}

impl<T: RawDataType> TensorClone<T> for TensorView<T> {
    fn copy_data(&self) -> DataOwned<T> {
        let data: Vec<T> = self.flat_iter().collect();
        DataOwned::new(data)
    }
}


impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType=T>,
    T: RawDataType,
    TensorBase<B>: TensorClone<T>,
{
    pub fn clone(&self) -> Tensor<B::DType> {
        Tensor {
            data: self.copy_data(),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            ndims: self.ndims.clone(),
        }
    }
}
