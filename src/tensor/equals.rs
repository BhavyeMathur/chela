use crate::{Tensor, TensorDataType};

impl<T1, T2> PartialEq<Tensor<'_, T1>> for Tensor<'_, T2>
where
    T1: TensorDataType,
    T2: TensorDataType + From<T1>,
{
    fn eq(&self, other: &Tensor<T1>) -> bool {
        self.array == other.array
    }
}
