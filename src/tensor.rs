pub mod constructors;
pub mod dtype;
pub mod shape;

use crate::tensor::dtype::RawData;

use std::ptr::NonNull;

pub(crate) struct Tensor<T>
where
    T: RawData,
{
    data: T,
    shape: Vec<usize>,
    ptr: NonNull<T::DType>,
}
