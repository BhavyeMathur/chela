use crate::dtype::NumericDataType;
use crate::{NdArray, RawDataType, StridedMemory};
use num::NumCast;

impl<T: NumericDataType> NdArray<'_, T> {
    pub fn astype<'b, F: NumericDataType>(&self) -> NdArray<'b, F>
    {
        let mut data = vec![F::default(); self.size()];

        for (dst, src) in data.iter_mut().zip(self.flatiter()) {
            *dst = NumCast::from(src).expect("astype conversion failed");
        }
        
        unsafe { NdArray::from_contiguous_owned_buffer(self.shape().to_vec(), data) }
    }
}

impl<'a, T: RawDataType> AsRef<NdArray<'a, T>> for NdArray<'a, T> {
    fn as_ref(&self) -> &NdArray<'a, T> {
        self
    }
}
