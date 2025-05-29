use crate::{NdArray, NdArrayMethods, RawDataType};
use std::ops::Neg;

pub(crate) trait UnaryOps<T: RawDataType> {
    fn neg<'a, 'b>(rhs: impl AsRef<NdArray<'a, T>>) -> NdArray<'static, T>
    where
        T: Neg<Output=T>,
    {
        let rhs = rhs.as_ref();

        let data = rhs.flatiter().map(|rhs| -rhs).collect();
        unsafe { NdArray::from_contiguous_owned_buffer(rhs.shape().to_vec(), data, false, false) }
    }
}

impl<T: RawDataType> UnaryOps<T> for T {}
