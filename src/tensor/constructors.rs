use crate::tensor::dtype::{RawData, RawDataType};
use crate::tensor::Tensor;

use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;
use crate::traits::shape::Shape;

use std::ptr::NonNull;

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
}

impl<A> Tensor<Vec<A>>
where
    Vec<A>: RawData<DType = A>,
    A: RawDataType,
{
    pub(crate) fn from_vector(data: impl Flatten<A> + Homogenous) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from_vector failed, found inhomogeneous dimensions"
        );
        let shape = data.shape().to_vec();

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
        let shape = data.shape().to_vec();

        let ptr = data.as_mut_ptr();
        Self::from_data_ptr(data, shape, ptr)
    }
}
