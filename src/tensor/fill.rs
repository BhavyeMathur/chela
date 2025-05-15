use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::{Tensor, TensorMethods};

unsafe fn fill_strided<T: Copy>(mut start: *mut T, value: T, stride: usize, n: usize) {
    for _ in 0..n {
        std::ptr::write(start, value);
        start = start.add(stride);
    }
}

unsafe fn fill_shape_and_stride<T: Copy>(mut start: *mut T, value: T, shape: &[usize], stride: &[usize]) {
    if shape.len() == 1 {
        return fill_strided(start, value, stride[0], shape[0]);
    }

    for _ in 0..shape[0] {
        fill_shape_and_stride(start, value, &shape[1..], &stride[1..]);
        start = start.add(stride[0]);
    }
}

impl<T: RawDataType> Tensor<'_, T> {
    pub fn fill(&mut self, value: T) {
        if self.is_contiguous() {
            return unsafe { fill_strided(self.ptr.as_ptr(), value, 1, self.len); };
        }

        let (shape, stride) = collapse_to_uniform_stride(&self.shape, &self.stride);
        unsafe { fill_shape_and_stride(self.ptr.as_ptr(), value, &shape, &stride); }
    }
}
