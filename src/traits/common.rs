use crate::{NdArray, RawDataType, Tensor, TensorDataType};

impl<'a, T: RawDataType> AsRef<NdArray<'a, T>> for NdArray<'a, T> {
    fn as_ref(&self) -> &NdArray<'a, T> {
        self
    }
}

impl<'a, T: TensorDataType> AsRef<Tensor<'a, T>> for Tensor<'a, T> {
    fn as_ref(&self) -> &Tensor<'a, T> {
        self
    }
}
