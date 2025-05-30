use crate::{NdArray, Tensor, TensorDataType};

impl<T: TensorDataType> PartialEq<Tensor<'_, T>> for Tensor<'_, T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.array == other.array
    }
}

impl<T: TensorDataType> PartialEq<NdArray<'_, T>> for Tensor<'_, T> {
    fn eq(&self, other: &NdArray<T>) -> bool {
        self.array == other
    }
}

impl<T: TensorDataType> PartialEq<Tensor<'_, T>> for NdArray<'_, T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self == other.array
    }
}

impl<T: TensorDataType> PartialEq<&Tensor<'_, T>> for NdArray<'_, T> {
    fn eq(&self, other: &&Tensor<T>) -> bool {
        self == other.array
    }
}

impl<T: TensorDataType> PartialEq<Tensor<'_, T>> for &NdArray<'_, T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        *self == other.array
    }
}
