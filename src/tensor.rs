pub mod constructors;
pub mod data_buffer;
pub mod data_owned;
pub mod data_view;
pub mod dtype;

pub mod index_impl;
pub mod shape;
pub mod slice;
pub mod view_flat_iterator;
pub mod flat_iterator;
pub mod flatten;
pub mod clone;

use crate::tensor::data_buffer::DataBuffer;
use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;

#[derive(Debug)]
pub struct TensorBase<T: DataBuffer> {
    data: T,
    shape: Vec<usize>,
    stride: Vec<usize>,
    ndims: usize,
}

pub type Tensor<T> = TensorBase<DataOwned<T>>;
pub type TensorView<T> = TensorBase<DataView<T>>;
