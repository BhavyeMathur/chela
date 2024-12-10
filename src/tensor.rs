mod data_buffer;
pub mod data_owned;
pub mod data_view;
pub mod dtype;

pub mod shape;
mod constructors;

use crate::tensor::data_buffer::DataBuffer;
use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;

pub struct TensorBase<T: DataBuffer> {
    data: T,
    shape: Vec<usize>,
}

pub type Tensor<T> = TensorBase<DataOwned<T>>;
pub type TensorView<T> = TensorBase<DataView<T>>;
