mod dtype;
mod flatten_vector;

mod recursive_vector_trait;

use dtype::*;
use flatten_vector::*;

use std::ptr::NonNull;

pub(crate) struct Tensor<T>
where
    T: RawData,
{
    data: T,
    ptr: NonNull<T::DType>,
}

impl<T, A> Tensor<T>
where
    T: RawData<DType = A>,
{
    fn from_data_ptr(data: T, ptr: *mut A) -> Self {
        let ptr = NonNull::new(ptr);
        match ptr {
            Some(ptr) => Self { data, ptr },
            None => panic!("Tensor::new failed, received null pointer"),
        }
    }
}

impl<A> Tensor<Vec<A>>
where
    Vec<A>: RawData<DType = A>,
    A: RawDataType,
{
    pub(crate) fn from_vector(data: impl FlattenVec<A>) -> Self {
        let mut data = data.flatten();
        let ptr = data.as_mut_ptr();
        Self::from_data_ptr(data, ptr)
    }
}

impl<A, const N: usize> Tensor<[A; N]>
where
    [A; N]: RawData<DType = A>,
{
    pub(crate) fn from_array(mut data: [A; N]) -> Self {
        let ptr = data.as_mut_ptr();
        Self::from_data_ptr(data, ptr)
    }
}