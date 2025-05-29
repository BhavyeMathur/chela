use crate::{RawDataType, NdArray};

pub(crate) trait IntoTensor<'a, T: RawDataType> {
    #[allow(clippy::wrong_self_convention)]
    fn as_tensor(self) -> NdArray<'a, T>;
}

impl<'a, T: RawDataType> IntoTensor<'a, T> for NdArray<'a, T> {
    fn as_tensor(self) -> NdArray<'a, T> {
        self
    }
}

impl<'a, T: RawDataType> IntoTensor<'a, T> for T {
    fn as_tensor(self) -> NdArray<'a, T> {
        NdArray::scalar(self)
    }
}
