use crate::tensor::dtype::RawData;
use crate::tensor::DataOwned;

impl<T, A> DataOwned<T>
where
    T: RawData<DType = A>,
{
    // pub fn shape(&self) -> &[usize] {
    //     &self.shape
    // }
}
