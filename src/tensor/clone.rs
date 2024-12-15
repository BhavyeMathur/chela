use crate::data_buffer::{DataBuffer, DataOwned};
use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_contiguous;
use crate::iterator::flat_index_iterator::FlatIndexIterator;
use crate::{Tensor, TensorBase, TensorView};
use std::ptr::copy_nonoverlapping;

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
        let (mut shape, mut stride) = collapse_contiguous(&self.shape, &self.stride);

        let mut contiguous_stride = *stride.last().unwrap();
        if contiguous_stride == 1 {
            contiguous_stride = shape.pop().unwrap();
            stride.pop();
        }

        let size = self.size();
        let mut data = Vec::with_capacity(size);

        let src = self.data.const_ptr();
        let mut dst = data.as_mut_ptr();

        for i in FlatIndexIterator::from(&shape, &stride) {
            unsafe {
                copy_nonoverlapping(src.offset(i), dst, contiguous_stride);
                dst = dst.add(contiguous_stride);
            }
        }

        unsafe { data.set_len(size); }
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
