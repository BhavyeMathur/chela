use crate::dtype::NumericDataType;
use crate::{NdArray, RawDataType, TensorMethods};
use num::NumCast;

impl<T: NumericDataType> NdArray<'_, T> {
    pub fn astype<'b, F: NumericDataType>(&self) -> NdArray<'b, F>
    {
        let mut data = vec![F::default(); self.size()];

        for (dst, src) in data.iter_mut().zip(self.flatiter()) {
            *dst = NumCast::from(src).expect("astype conversion failed");
        }
        
        // TODO need to figure out behaviour of requires_grad and user_created for this method
        unsafe { NdArray::from_contiguous_owned_buffer(self.shape().to_vec(), data, self.requires_grad(), true) }
    }
}

impl<'a, T: RawDataType> AsRef<NdArray<'a, T>> for NdArray<'a, T> {
    fn as_ref(&self) -> &NdArray<'a, T> {
        self
    }
}
