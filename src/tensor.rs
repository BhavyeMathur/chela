pub mod data_buffer;
pub mod data_owned;
pub mod data_view;
pub mod dtype;

pub mod shape;

use crate::tensor::data_buffer::DataBuffer;
use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;

#[derive(Debug, Clone)]
pub struct TensorBase<T: DataBuffer, const D: usize> {
    data: T,
    shape: Vec<usize>,
}

pub type Tensor<T, const D: usize> = TensorBase<DataOwned<T>, D>;
pub type TensorView<T, const D: usize> = TensorBase<DataView<T>, D>;
