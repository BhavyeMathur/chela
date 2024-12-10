use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use crate::tensor::dtype::RawDataType;

use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;
use crate::traits::shape::Shape;

pub(crate) struct DataOwned<T: RawDataType> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T: RawDataType> DataOwned<T> {
    pub fn len(&self) -> &usize {
        &self.len
    }

    pub fn capacity(&self) -> &usize {
        &self.capacity
    }
}

impl<T: RawDataType> DataOwned<T> {
    pub(crate) fn from(data: impl Flatten<T> + Homogenous) -> Self {
        let data = data.flatten();

        if data.len() == 0 {
            panic!("Tensor::from() failed, cannot create data buffer from empty data");
        }

        let mut data = ManuallyDrop::new(data);

        // safe to unwrap because we've checked length above
        let ptr = data.as_mut_ptr();
        let ptr = NonNull::new(ptr).unwrap();

        let len = data.len();
        let capacity = data.capacity();

        Self { len, capacity, ptr }
    }
}

impl<T: RawDataType> Drop for DataOwned<T> {
    fn drop(&mut self) {
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };

        self.len = 0;
        self.capacity = 0;
    }
}
