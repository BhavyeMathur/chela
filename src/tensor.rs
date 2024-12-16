pub mod constructors;
pub mod data_buffer;
pub mod dtype;

pub mod index_impl;
pub mod shape;
pub mod slice;
pub mod iterator;
pub mod flatten;
pub mod clone;
pub mod squeeze;
pub mod equals;

pub use iterator::*;

use crate::tensor::data_buffer::{DataBuffer, DataOwned, DataView};

#[derive(Debug)]
pub struct TensorBase<T: DataBuffer> {
    data: T,
    shape: Vec<usize>,
    stride: Vec<usize>,
    ndims: usize,
}

pub type Tensor<T> = TensorBase<DataOwned<T>>;
pub type TensorView<T> = TensorBase<DataView<T>>;
