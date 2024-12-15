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

        match stride.last_mut() {
            Some(&mut mut contiguous_stride) => {
                // if elements along the last axis are located contiguously,
                // we can collapse the last dimension and copy contiguous_stride elements at once
                // collapse_contiguous() call before guarantees only the last dimension (if any) is contiguous
                if contiguous_stride == 1 {
                    contiguous_stride = shape.pop().unwrap();
                    stride.pop();
                }
                // if elements along the last axis aren't located contiguously,
                // they must correspond to a TensorView with a step-size along the last axis of > 1
                // this is equivalent to 1 contiguous element along the last axis
                else {
                    contiguous_stride = 1;
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
            None => {
                DataOwned::new(vec![self.data[0]])  // zero-dimensional tensor
            }
        }
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
