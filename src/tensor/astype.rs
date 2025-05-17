use crate::dtype::NumericDataType;
use crate::{Tensor, TensorMethods};
use num::NumCast;

impl<T: NumericDataType> Tensor<'_, T> {
    pub fn astype<'b, F: NumericDataType>(&self) -> Tensor<'b, F>
    {
        let mut data = vec![F::default(); self.size()];

        for (dst, src) in data.iter_mut().zip(self.flatiter()) {
            *dst = NumCast::from(src).expect("astype conversion failed");
        }

        unsafe { Tensor::from_contiguous_owned_buffer(self.shape().to_vec(), data) }
    }
}
