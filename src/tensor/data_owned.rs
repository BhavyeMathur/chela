use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use crate::tensor::dtype::RawDataType;
use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;
use crate::traits::shape::Shape;

pub(crate) struct DataOwned<T>
where
    T: RawDataType,
{
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> DataOwned<T>
where
    T: RawDataType,
{
    pub fn len(&self) -> &usize {
        &self.len
    }

    pub fn capacity(&self) -> &usize {
        &self.capacity
    }
}

impl<T> DataOwned<T>
where
    T: RawDataType,
{
    pub(crate) fn from(data: impl Flatten<T> + Homogenous) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from() failed, found inhomogeneous dimensions"
        );

        let data = data.flatten();

        if data.len() == 0 {
            panic!("cannot create data buffer from empty data");
        }

        let mut data = ManuallyDrop::new(data);

        // safe to unwrap because we've checked length above
        let ptr = data.as_mut_ptr();
        let ptr = NonNull::new(ptr).unwrap();

        let len = data.len();
        let capacity = data.capacity();

        Self { len, capacity, ptr }
    }

    fn drop(&mut self) {
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };

        self.len = 0;
        self.capacity = 0;
    }
}
