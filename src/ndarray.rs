use std::marker::PhantomData;
use std::ptr::NonNull;

pub mod methods;
pub use methods::*;

pub mod iterator;
pub use iterator::*;

pub mod reshape;

pub(crate) mod flags;
use flags::NdArrayFlags;

pub mod reduce;

pub mod constructors;
pub mod index_impl;
pub mod slice;
pub mod fill;
pub mod clone;
pub mod equals;
pub mod broadcast;
pub mod binary_ops;
pub mod random;
pub mod astype;

mod print;
mod unary_ops;
mod assign_ops;

pub(crate) const MAX_DIMS: usize = 32;
pub(crate) const MAX_ARGS: usize = 16;

use crate::dtype::RawDataType;
use crate::gradient_function::GradientFunction;

pub struct NdArray<'a, T: RawDataType> {
    pub(crate) ptr: NonNull<T>,
    len: usize,
    capacity: usize,

    shape: Vec<usize>,
    stride: Vec<usize>,
    pub(crate) flags: NdArrayFlags,

    pub(crate) grad_fn: GradientFunction<T>,

    _marker: PhantomData<&'a T>,
}
