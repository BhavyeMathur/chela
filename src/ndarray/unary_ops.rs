use crate::{NdArray, RawDataType};
use std::ops::Neg;
use crate::unary_ops::UnaryOps;

impl<T: RawDataType + Neg<Output=T>> Neg for NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    fn neg(self) -> Self::Output { -&self }
}

impl<T: RawDataType + Neg<Output=T>> Neg for &NdArray<'_, T> {
    type Output = NdArray<'static, T>;

    fn neg(self) -> Self::Output { <T as UnaryOps<T>>::neg(self) }
}
