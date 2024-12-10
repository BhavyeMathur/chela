use crate::tensor::dtype::RawDataType;
use crate::tensor::Tensor;

impl<T: RawDataType> Tensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> &usize {
        &self.shape[0]
    }
}
