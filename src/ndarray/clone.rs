use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::iterator::flat_index_generator::FlatIndexGenerator;
use crate::{NdArray, TensorMethods};
use std::ptr::copy_nonoverlapping;

impl<'a, T: RawDataType> NdArray<'a, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn clone<'r>(&self) -> NdArray<'r, T> {
        unsafe { NdArray::from_contiguous_owned_buffer(self.shape.clone(), self.clone_data(), self.requires_grad(), false) }
    }

    pub(super) fn clone_data(&self) -> Vec<T> {
        if self.is_contiguous() {
            return unsafe { self.clone_data_contiguous() };
        }
        unsafe { self.clone_data_non_contiguous() }
    }

    /// Safety: expects tensor buffer is contiguously stored
    unsafe fn clone_data_contiguous(&self) -> Vec<T> {
        let mut data = Vec::with_capacity(self.len);

        copy_nonoverlapping(self.ptr(), data.as_mut_ptr(), self.len);
        data.set_len(self.len);
        data
    }

    /// Safety: expects tensor buffer is not contiguously stored
    unsafe fn clone_data_non_contiguous(&self) -> Vec<T> {
        let size = self.size();
        let mut data = Vec::with_capacity(size);

        let (mut shape, mut stride) = collapse_to_uniform_stride(&self.shape, &self.stride);

        // safe to unwrap because if stride has no elements, this would be a scalar tensor
        // however, scalar tensors are contiguously stored so this method wouldn't be called
        let &mut mut contiguous_stride = stride.last_mut().unwrap();

        // if elements along the last axis are located contiguously,
        // we can collapse the last dimension and copy contiguous_stride elements at once
        if contiguous_stride == 1 {
            contiguous_stride = shape.pop().unwrap();
            stride.pop();
        }

        // if elements along the last axis aren't located contiguously,
        // they must correspond to a Tensor view with a step-size along the last axis of > 1
        // this is equivalent to 1 contiguous element along the last axis
        else {
            contiguous_stride = 1;
        }

        let src = self.ptr();
        let mut dst = data.as_mut_ptr();

        for i in FlatIndexGenerator::from(&shape, &stride) {
            copy_nonoverlapping(src.add(i), dst, contiguous_stride);
            dst = dst.add(contiguous_stride);
        }

        data.set_len(size);
        data
    }
}
