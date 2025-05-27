use std::marker::PhantomData;
use std::ptr::NonNull;

pub mod dtype;
pub use dtype::*;

pub mod methods;
pub use methods::*;

pub mod iterator;
pub use iterator::*;

pub mod reduce;

pub mod constructors;
pub mod index_impl;
pub mod slice;
pub mod fill;
pub mod reshape;
pub mod clone;
pub mod equals;
pub mod broadcast;
pub mod operations;
pub mod random;
pub mod astype;

mod flags;
use flags::TensorFlags;

mod print;
mod backward;

pub(crate) const MAX_DIMS: usize = 32;
pub(crate) const MAX_ARGS: usize = 16;

use crate::gradient_function::{GradientFunction};

pub struct Tensor<'a, T: RawDataType> {
    pub(crate) ptr: NonNull<T>,
    len: usize,
    capacity: usize,

    shape: Vec<usize>,
    stride: Vec<usize>,
    flags: TensorFlags,
    
    grad_fn: GradientFunction<T>,

    _marker: PhantomData<&'a T>,
}
