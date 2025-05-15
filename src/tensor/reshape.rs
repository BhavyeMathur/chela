use crate::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;
use crate::{Axis, Tensor, TensorMethods};
use crate::slice::update_flags_with_contiguity;

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub fn flatten<'b>(&self) -> Tensor<'b, T> {
        unsafe { Tensor::from_contiguous_owned_buffer(vec![self.size()], self.clone_data()) }
    }

    pub(super) unsafe fn reshaped_view_with_flags(&self, shape: Vec<usize>, stride: Vec<usize>, mut flags: TensorFlags) -> Tensor<T> {
        flags = update_flags_with_contiguity(flags, &shape, &stride);

        Tensor {
            ptr: self.ptr,
            len: shape.iter().product(),
            capacity: 0,

            shape,
            stride,
            flags,

            _marker: self._marker,
        }
    }

    pub(crate) unsafe fn reshaped_view(&self, shape: Vec<usize>, stride: Vec<usize>) -> Tensor<T> {
        self.reshaped_view_with_flags(shape, stride, self.flags - TensorFlags::Owned - TensorFlags::Writeable)
    }

    pub(crate) unsafe fn mut_reshaped_view(&self, shape: Vec<usize>, stride: Vec<usize>) -> Tensor<T> {
        self.reshaped_view_with_flags(shape, stride, self.flags - TensorFlags::Owned)
    }

    pub fn view(&'a self) -> Tensor<'a, T> {
        unsafe { self.mut_reshaped_view(self.shape.clone(), self.stride.clone()) }
    }

    pub fn squeeze(&'a self) -> Tensor<'a, T> {
        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        (shape, stride) = shape.iter().zip(stride.iter())
            .filter(|&(&axis_length, _)| axis_length != 1)
            .unzip();

        unsafe { self.mut_reshaped_view(shape, stride) }
    }

    pub fn unsqueeze(&'a self, axis: Axis) -> Tensor<'a, T> {
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

        unsafe { self.mut_reshaped_view(shape, stride) }
    }
}
