use std::ptr::NonNull;

use crate::tensor::dtype::{RawData, RawDataType};
use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;
use crate::traits::shape::Shape;

pub(crate) struct DataOwned<T>
where
    T: RawData,
{
    data: T,
    ptr: NonNull<T::DType>,
    len: usize,
    capacity: usize,
}

impl<T, A> DataOwned<T>
where
    T: RawData<DType = A>,
{
    pub fn len(&self) -> &usize {
        &self.len
    }

    pub fn capacity(&self) -> &usize {
        &self.capacity
    }
}

impl<A> DataOwned<Vec<A>>
where
    Vec<A>: RawData<DType = A>,
    A: RawDataType,
{
    pub(crate) fn from_vector<Q>(data: Vec<Q>) -> Self
    where
        Vec<Q>: Flatten<A> + Homogenous,
    {
        if data.len() == 0 {
            panic!("cannot create data buffer from empty vector");
        }

        assert!(
            data.check_homogenous(),
            "Tensor::from_vector failed, found inhomogeneous dimensions"
        );
        let mut data = data.flatten();

        // safe to unwrap because we've checked length above
        let ptr = data.as_mut_ptr();
        let ptr = NonNull::new(ptr).unwrap();

        let len = data.len();
        let capacity = data.capacity();

        Self {
            data,
            len,
            capacity,
            ptr,
        }
    }
}

impl<A, const N: usize> DataOwned<[A; N]>
where
    [A; N]: RawData<DType = A>,
{
    pub(crate) fn from_array(mut data: [A; N]) -> Self
    where
        [A; N]: Shape,
    {
        if data.len() == 0 {
            panic!("cannot create data buffer from empty vector");
        }

        let ptr = data.as_mut_ptr();
        let ptr = NonNull::new(ptr).unwrap();

        let shape = data.shape();
        let len = shape.iter().product();
        let capacity = len;

        Self {
            data,
            len,
            capacity,
            ptr,
        }
    }
}

// impl<A> DataOwned<A> {
//     pub(crate) fn from_vector(v: Vec<A>) -> Self {
//         let mut v = ManuallyDrop::new(v);
//         let ptr = NonNull::new(v.as_mut_ptr()).unwrap();
//
//         Self { ptr, len, capacity }
//     }
// }

// impl<A> Drop for DataOwned<A> {
//     fn drop(&mut self) {
//         unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };
//
//         self.len = 0;
//         self.capacity = 0;
//     }
// }
