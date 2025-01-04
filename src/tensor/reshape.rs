use crate::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;
use crate::{Axis, Tensor};

impl<T: RawDataType> Tensor<T> {
    pub(super) unsafe fn reshaped_view(&self, shape: Vec<usize>, stride: Vec<usize>) -> Tensor<T> {
        Tensor {
            ptr: self.ptr,
            len: self.len,
            capacity: self.capacity,

            shape,
            stride,
            flags: self.flags - TensorFlags::Owned,
        }
    }

    pub(super) fn copy_view(&self) -> Tensor<T> {
        unsafe { self.reshaped_view(self.shape.clone(), self.stride.clone()) }
    }

    pub fn flatten(&self) -> Tensor<T> {
        unsafe { Tensor::from_contiguous_owned_buffer(vec![self.size()], self.clone_data()) }
    }

    pub fn squeeze(&self) -> Tensor<T> {
        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        (shape, stride) = shape.iter().zip(stride.iter())
            .filter(|&(&axis_length, _)| axis_length != 1)
            .unzip();

        unsafe { self.reshaped_view(shape, stride) }
    }

    pub fn unsqueeze(&self, axis: Axis) -> Tensor<T> {
        let axis = axis.0;
        assert!(axis <= self.ndims(), "Tensor::unsqueeze(), axis out of bounds");

        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        if axis == self.ndims() {
            shape.push(1);
            stride.push(1)
        } else {
            shape.insert(axis, 1);
            stride.insert(axis, stride[axis] * shape[axis + 1]);
        }

        unsafe { self.reshaped_view(shape, stride) }
    }
}
