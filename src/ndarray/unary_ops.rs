use crate::ops::unary_ops::UnaryOps;
use crate::{Constructors, NdArray, RawDataType, StridedMemory};
use std::ops::Neg;

impl<T: RawDataType + UnaryOps> Neg for NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    fn neg(self) -> Self::Output { -&self }
}

impl<T: RawDataType + UnaryOps> Neg for &NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    fn neg(self) -> Self::Output {
        let mut data = vec![T::default(); self.size()];

        unsafe {
            <T as UnaryOps>::neg(self.ptr(), self.shape(), self.stride(), data.as_mut_ptr());
            NdArray::from_contiguous_owned_buffer(self.shape().to_vec(), data)
        }
    }
}
