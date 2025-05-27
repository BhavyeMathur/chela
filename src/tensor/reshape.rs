use crate::dtype::RawDataType;
use crate::slice::update_flags_with_contiguity;
use crate::tensor::flags::TensorFlags;
use crate::util::to_vec::ToVec;
use crate::{Axis, AxisType, Tensor, TensorMethods};

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub fn flatten<'b>(&self) -> Tensor<'b, T> {
        unsafe { Tensor::from_contiguous_owned_buffer(vec![self.size()], self.clone_data(), self.requires_grad(), false) }
    }

    pub(super) unsafe fn reshaped_view_with_flags_and_offset(&'a self,
                                                             offset: usize,
                                                             shape: Vec<usize>,
                                                             stride: Vec<usize>,
                                                             mut flags: TensorFlags) -> Tensor<'a, T> {
        flags = update_flags_with_contiguity(flags, &shape, &stride) - TensorFlags::UserCreated;

        Tensor {
            ptr: self.ptr.add(offset),
            len: shape.iter().product(),
            capacity: 0,

            shape,
            stride,
            flags,

            grad_fn: self.grad_fn.clone(),

            _marker: self._marker,
        }
    }

    pub(crate) unsafe fn reshaped_view_with_offset(&'a self,
                                                   offset: usize,
                                                   shape: Vec<usize>,
                                                   stride: Vec<usize>) -> Tensor<'a, T> {
        self.reshaped_view_with_flags_and_offset(offset, shape, stride, self.flags - TensorFlags::Owned)
    }

    pub(crate) unsafe fn reshaped_view(&'a self,
                                       shape: Vec<usize>,
                                       stride: Vec<usize>) -> Tensor<'a, T> {
        self.reshaped_view_with_flags_and_offset(0, shape, stride, self.flags - TensorFlags::Owned)
    }

    pub fn view(&'a self) -> Tensor<'a, T> {
        unsafe { self.reshaped_view(self.shape.clone(), self.stride.clone()) }
    }

    pub fn squeeze(&'a self) -> Tensor<'a, T> {
        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        (shape, stride) = shape.iter().zip(stride.iter())
                               .filter(|&(&axis_length, _)| axis_length != 1)
                               .unzip();

        unsafe { self.reshaped_view(shape, stride) }
    }

    pub fn unsqueeze(&'a self, axis: Axis) -> Tensor<'a, T> {
        let axis = axis.get_absolute(self.ndims() + 1);

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

    pub fn reshape(&'a self, new_shape: impl ToVec<usize>) -> Tensor<'a, T> {
        let new_shape = new_shape.to_vec();

        if self.size() != new_shape.iter().product() {
            panic!("total number of elements must not change during reshape");
        }

        let mut new_stride = vec![0; new_shape.len()];
        let mut acc = 1;
        for (i, dim) in new_shape.iter().rev().enumerate() {
            new_stride[new_shape.len() - 1 - i] = acc;
            acc *= *dim;
        }

        unsafe { self.reshaped_view(new_shape, new_stride) }
    }
}
