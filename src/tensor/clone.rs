use crate::data_owned::DataOwned;
use crate::dtype::RawDataType;
use crate::{Tensor, TensorView};
use std::intrinsics::copy_nonoverlapping;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

impl<T: RawDataType> Clone for DataOwned<T> {
    fn clone(&self) -> DataOwned<T> {
        let mut data = Vec::with_capacity(self.len);

        let src = self.ptr.as_ptr();
        let dst = data.as_mut_ptr();

        unsafe {
            copy_nonoverlapping(src, dst, self.len);
            data.set_len(self.len);
        }

        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
        let data = ManuallyDrop::new(data);

        Self {
            len: data.len(),
            capacity: data.capacity(),
            ptr: NonNull::new(dst).unwrap(),
        }
    }
}

impl<T: RawDataType> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            ndims: self.ndims.clone(),
        }
    }
}

impl<T: RawDataType> TensorView<T> {
    pub fn clone(&self) -> Tensor<T> {
        let data: Vec<T> = self.flat_iter().collect();
        Tensor {
            data: DataOwned::new(data),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            ndims: self.ndims.clone(),
        }
    }
}
