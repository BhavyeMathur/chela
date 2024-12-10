use crate::tensor::dtype::RawData;
use crate::tensor::Tensor;

impl<T, A> Tensor<T>
where
    T: RawData<DType = A>,
{
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
