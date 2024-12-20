use std::mem::ManuallyDrop;
use std::ops::Index;
use std::ptr::NonNull;

use crate::tensor::dtype::RawDataType;

use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;

#[derive(Debug)]
pub struct DataOwned<T: RawDataType> {
    pub(super) ptr: NonNull<T>,
    pub(super) len: usize,
    pub(super) capacity: usize,
}

impl<T: RawDataType> DataOwned<T> {
    pub(in crate::tensor) fn len(&self) -> &usize {
        &self.len
    }

    pub(in crate::tensor) fn capacity(&self) -> &usize {
        &self.capacity
    }
}

impl<T: RawDataType> DataOwned<T> {
    pub fn new(data: Vec<T>) -> Self {
        if data.is_empty() {
            panic!("Tensor::from() failed, cannot create data buffer from empty data");
        }

        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
        let mut data = ManuallyDrop::new(data);

        // safe to unwrap because we've checked length above
        let ptr = data.as_mut_ptr();

        Self {
            len: data.len(),
            capacity: data.capacity(),
            ptr: NonNull::new(ptr).unwrap(),
        }
    }

    // pub fn new_from_ref(data: Vec<&T>) -> Self {
    //     if data.is_empty() {
    //         panic!("Tensor::from() failed, cannot create data buffer from empty data");
    //     }
    //
    //     // take control of the data so that Rust doesn't drop it once the vector goes out of scope
    //     let mut data = ManuallyDrop::new(data);
    //
    //     // safe to unwrap because we've checked length above
    //     let ptr = data.as_mut_ptr();
    //
    //     Self {
    //         len: data.len(),
    //         capacity: data.capacity(),
    //         ptr: NonNull::new(ptr).unwrap(),
    //     }
    // }

    pub fn from(data: impl Flatten<T> + Homogenous) -> Self {
        Self::new(data.flatten())
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

impl<T> Index<usize> for DataOwned<T>
where
    T: RawDataType,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        assert!(index < self.len, "Index '{index}' out of bounds"); // type implies 0 <= index
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}
