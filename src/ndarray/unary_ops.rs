use crate::{NdArray, RawDataType};
use std::ops::Neg;

trait UnaryOps<T: RawDataType> {
    fn neg<'a, 'b>(rhs: impl AsRef<NdArray<'a, T>>) -> NdArray<'static, T>
    where
        T: Neg<Output=T>,
    {
        let rhs = rhs.as_ref();

        let data = rhs.flatiter().map(|rhs| -rhs).collect();
        unsafe { NdArray::from_contiguous_owned_buffer(rhs.shape.clone(), data, false, false) }
    }
}


impl<T: RawDataType + Neg<Output=T>> Neg for NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    fn neg(self) -> Self::Output { <T as UnaryOps<T>>::neg(self) }
}

impl<T: RawDataType + Neg<Output=T>> Neg for &NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    fn neg(self) -> Self::Output { <T as UnaryOps<T>>::neg(self) }
}

impl<T: RawDataType> UnaryOps<T> for T {}
