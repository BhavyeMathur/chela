pub mod constructors;
pub mod data_buffer;
pub mod data_owned;
pub mod data_view;
pub mod dtype;

pub mod shape;

use crate::tensor::data_buffer::DataBuffer;
use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;

#[derive(Debug, Clone)]
pub struct TensorBase<T: DataBuffer> {
    data: T,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

pub type Tensor<T> = TensorBase<DataOwned<T>>;
pub type TensorView<T> = TensorBase<DataView<T>>;
