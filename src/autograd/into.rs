use crate::util::to_vec::ToVec;
use crate::{NdArray, RawDataType};

pub(crate) trait IntoNdArray<'a, T: RawDataType> {
    #[allow(clippy::wrong_self_convention)]
    fn as_ndarray(self) -> NdArray<'a, T>;
}

impl<'a, T: RawDataType> IntoNdArray<'a, T> for NdArray<'a, T> {
    fn as_ndarray(self) -> NdArray<'a, T> {
        self
    }
}

impl<'a, T: RawDataType> IntoNdArray<'a, T> for T {
    fn as_ndarray(self) -> NdArray<'a, T> {
        NdArray::scalar(self)
    }
}

impl<'a, T: RawDataType, const N: usize> IntoNdArray<'a, T> for [T; N] {
    fn as_ndarray(self) -> NdArray<'a, T> {
        unsafe { NdArray::from_contiguous_owned_buffer(vec![N], self.to_vec()) }
    }
}

impl<'a, T: RawDataType> IntoNdArray<'a, T> for Vec<T> {
    fn as_ndarray(self) -> NdArray<'a, T> {
        unsafe { NdArray::from_contiguous_owned_buffer(vec![self.len()], self) }
    }
}
