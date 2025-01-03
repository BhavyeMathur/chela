use crate::tensor::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;
use crate::tensor::Tensor;
use crate::traits::flatten::Flatten;
use crate::traits::nested::Nested;
use crate::traits::shape::Shape;
use crate::traits::to_vec::ToVec;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

// calculates the stride from the tensor's shape
// shape [5, 3, 2, 1] -> stride [10, 2, 1, 1]
fn stride_from_shape(shape: &[usize]) -> Vec<usize> {
    let ndims = shape.len();
    let mut stride = vec![0; ndims];

    let mut p = 1;
    for i in (0..ndims).rev() {
        stride[i] = p;
        p *= shape[i];
    }

    stride
}

impl<T: RawDataType> Tensor<T> {
    /// Safety: ensure data is non-empty and shape & stride matches data buffer
    pub(super) unsafe fn from_owned_buffer(shape: Vec<usize>, stride: Vec<usize>, data: Vec<T>) -> Self {
        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
        let mut data = ManuallyDrop::new(data);

        Self {
            ptr: NonNull::new_unchecked(data.as_mut_ptr()),
            len: data.len(),
            capacity: data.capacity(),

            shape,
            stride,
            flags: TensorFlags::Owned | TensorFlags::Contiguous,
        }
    }

    /// Safety: ensure data is non-empty and shape matches data buffer
    pub(super) unsafe fn from_contiguous_owned_buffer(shape: Vec<usize>, data: Vec<T>) -> Self {
        let stride = stride_from_shape(&shape);
        Self::from_owned_buffer(shape, stride, data)
    }

    pub fn from<const D: usize>(data: impl Flatten<T> + Shape + Nested<{ D }>) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from() failed, found inhomogeneous dimensions"
        );

        let shape = data.shape();
        let data = data.flatten();

        assert!(
            !data.is_empty(),
            "Tensor::from() failed, cannot create data buffer from empty data"
        );

        unsafe { Tensor::from_contiguous_owned_buffer(shape, data) }
    }

    pub fn full(n: T, shape: impl ToVec<usize>) -> Self {
        let shape = shape.to_vec();

        let data = vec![n; shape.iter().product()];
        assert!(!data.is_empty(), "Cannot create an empty tensor!");

        unsafe { Tensor::from_contiguous_owned_buffer(shape, data) }
    }

    pub fn zeros(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>,
    {
        Self::full(false.into(), shape)
    }

    pub fn ones(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>,
    {
        Self::full(true.into(), shape)
    }

    pub fn scalar(n: T) -> Self {
        Tensor::full(n, [])
    }
}

impl<T: RawDataType> Drop for Tensor<T> {
    fn drop(&mut self) {
        if self.flags.contains(TensorFlags::Owned) {
            // drops the data
            unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };
        }

        self.len = 0;
        self.capacity = 0;
    }
}
