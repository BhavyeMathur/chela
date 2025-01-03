pub mod constructors;
pub mod dtype;

pub mod index_impl;
pub mod methods;
pub mod slice;
pub mod iterator;
pub mod fill;
pub mod reshape;
pub mod clone;
pub mod equals;
mod flags;

use crate::dtype::RawDataType;

pub use iterator::*;

use crate::tensor::flags::TensorFlags;
use std::ptr::NonNull;

#[derive(Debug)]
pub struct Tensor<T: RawDataType> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,

    shape: Vec<usize>,
    stride: Vec<usize>,
    flags: TensorFlags,
}
