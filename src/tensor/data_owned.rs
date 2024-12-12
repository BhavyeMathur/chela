use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use crate::tensor::dtype::RawDataType;

use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;

#[derive(Debug, Clone)]
pub(crate) struct DataOwned<T: RawDataType> {
    pub(super) ptr: NonNull<T>,
    pub(super) len: usize,
    pub(super) capacity: usize,
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
    pub fn from(data: impl Flatten<T> + Homogenous) -> Self {
        let data = data.flatten();

        if data.len() == 0 {
            panic!("Tensor::from() failed, cannot create data buffer from empty data");
        }

        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
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
        // drops the data
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };

        self.len = 0;
        self.capacity = 0;
    }
}
