use crate::dtype::RawDataType;
use crate::traits::to_vec::ToVec;
use crate::Tensor;

impl<T: RawDataType> Tensor<'_, T> {
    pub fn reduce<D: RawDataType>(&self, func: fn(Tensor<T>) -> D, axes: impl ToVec<usize>) -> Tensor<D> {
        let axes = axes.to_vec();

        let mut shape_mask = vec![false; self.ndims()];
        for i in 0..axes.len() {
            shape_mask[axes[i]] = true;
        }

        let mut reduced_shape = Vec::with_capacity(self.ndims() - axes.len());

        for i in 0..self.shape.len() {
            if shape_mask[i] {
                reduced_shape.push(self.shape[i]);
            }
        }

        let mut output = Vec::with_capacity(reduced_shape.iter().product());
        for slice in self.nditer(axes) {
            output.push(func(slice));
        }

        unsafe { Tensor::from_contiguous_owned_buffer(reduced_shape, output) }
    }
}
