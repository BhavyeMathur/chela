use crate::data_buffer::DataBuffer;
use crate::tensor::dtype::RawDataType;
use crate::{Axis, TensorBase, TensorView};

impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType = T>,
    T: RawDataType,
{
    pub fn squeeze(&self) -> TensorView<T> {
        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        (shape, stride) = shape
            .iter()
            .zip(stride.iter())
            .filter(|&(axis_length, _)| axis_length != &1)
            .unzip();

        let ndims = shape.len();

        TensorView {
            data: self.data.to_view(),
            shape,
            stride,
            ndims,
        }
    }

    pub fn unsqueeze(&self, axis: Axis) -> TensorView<T> {
        assert!(axis.0 <= self.ndims, "dimension out of bounds");

        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        if shape.len() > axis.0 {
            shape.insert(axis.0, 1)
        } else {
            shape.push(1)
        };

        if stride.len() > axis.0 {
            stride.insert(axis.0, stride[axis.0] * shape[axis.0 + 1])
        } else {
            stride.push(1)
        };

        let ndims = shape.len();

        TensorView {
            data: self.data.to_view(),
            shape,
            stride,
            ndims,
        }
    }
}
