use crate::{RawDataType, Tensor};

pub(crate) trait IntoTensor<'a, T: RawDataType> {
    #[allow(clippy::wrong_self_convention)]
    fn as_tensor(self) -> Tensor<'a, T>;
}

impl<'a, T: RawDataType> IntoTensor<'a, T> for Tensor<'a, T> {
    fn as_tensor(self) -> Tensor<'a, T> {
        self
    }
}

impl<'a, T: RawDataType> IntoTensor<'a, T> for T {
    fn as_tensor(self) -> Tensor<'a, T> {
        Tensor::scalar(self)
    }
}
