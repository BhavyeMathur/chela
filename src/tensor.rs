use std::marker::PhantomData;
use std::ptr::NonNull;

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
pub mod broadcast;

mod flags;

use crate::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;

pub use iterator::*;

#[derive(Debug)]
pub struct Tensor<'a, T: RawDataType> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,

    shape: Vec<usize>,
    stride: Vec<usize>,
    flags: TensorFlags,

    _marker: PhantomData<&'a T>,
}
