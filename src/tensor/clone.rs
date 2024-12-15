use crate::data_buffer::DataBuffer;
use crate::data_owned::DataOwned;
use crate::dtype::RawDataType;
use crate::{Tensor, TensorBase, TensorView};
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

pub(super) trait TensorClone<T: RawDataType> {
    fn copy_data(&self) -> DataOwned<T>;
}

impl<T: RawDataType> TensorClone<T> for Tensor<T> {
    fn copy_data(&self) -> DataOwned<T> {
        self.data.clone()
    }
}

impl<T: RawDataType> TensorClone<T> for TensorView<T> {
    fn copy_data(&self) -> DataOwned<T> {
        let data: Vec<T> = self.flat_iter().collect();
        DataOwned::new(data)
    }
}


impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType=T>,
    T: RawDataType,
    TensorBase<B>: TensorClone<T>,
{
    pub fn clone(&self) -> Tensor<B::DType> {
        Tensor {
            data: self.copy_data(),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            ndims: self.ndims.clone(),
        }
    }
}
