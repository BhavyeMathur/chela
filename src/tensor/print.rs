use crate::{Tensor, TensorDataType};
use std::fmt;

impl<T: TensorDataType> fmt::Debug for Tensor<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.array.fmt(f)
    }
}
