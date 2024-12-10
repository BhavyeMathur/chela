mod dtype;
mod flatten_vector;
mod homogenous_vector;
pub mod shape;

#[macro_use]
mod recursive_vector_trait;

use crate::tensor::dtype::{RawData, RawDataType};
use crate::tensor::flatten_vector::FlattenVec;
use crate::tensor::homogenous_vector::HomogenousVec;
use crate::tensor::shape::Shape;
use std::ptr::NonNull;

pub(crate) struct Tensor<T>
where
    T: RawData,
{
    data: T,
    shape: Vec<usize>,
    ptr: NonNull<T::DType>,
}

impl<T, A> Tensor<T>
where
    T: RawData<DType = A>,
{
    fn from_data_ptr(data: T, shape: Vec<usize>, ptr: *mut A) -> Self {
        let ptr = NonNull::new(ptr);
        match ptr {
            Some(ptr) => Self { data, shape, ptr },
            None => panic!("Tensor::new failed, received null pointer"),
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

impl<A> Tensor<Vec<A>>
where
    Vec<A>: RawData<DType = A>,
    A: RawDataType,
{
    pub(crate) fn from_vector(data: impl FlattenVec<A> + HomogenousVec) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from_vector failed, found inhomogeneous dimensions"
        );
        let shape = data.shape();

        let mut data = data.flatten();
        let ptr = data.as_mut_ptr();
        Self::from_data_ptr(data, shape, ptr)
    }
}

impl<A, const N: usize> Tensor<[A; N]>
where
    [A; N]: RawData<DType = A> + Shape,
{
    pub(crate) fn from_array(mut data: [A; N]) -> Self {
        let ptr = data.as_mut_ptr();
        let shape = data.shape();
        Self::from_data_ptr(data, shape, ptr)
    }
}
