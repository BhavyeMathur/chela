use std::marker::PhantomData;
use std::ptr::NonNull;

pub mod constructors;
pub use constructors::*;

pub mod dtype;
pub use dtype::*;

pub mod methods;
pub use methods::*;

pub mod iterator;
pub use iterator::*;

pub mod reduce;
pub use reduce::*;

pub mod index_impl;
pub mod slice;
pub mod fill;
pub mod reshape;
pub mod clone;
pub mod equals;
pub mod broadcast;
pub mod binary_ops;
pub mod random;
pub mod astype;

mod flags;
use flags::TensorFlags;

mod print;

pub(crate) const MAX_DIMS: usize = 32;

pub struct Tensor<'a, T: RawDataType> {
    pub(crate) ptr: NonNull<T>,
    len: usize,
    capacity: usize,

    shape: Vec<usize>,
    stride: Vec<usize>,
    flags: TensorFlags,

    _marker: PhantomData<&'a T>,
}
