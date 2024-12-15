use crate::data_buffer::{DataBuffer, DataOwned, DataView};
use crate::tensor::dtype::RawDataType;
use crate::tensor::{Tensor, TensorBase, TensorView};
use crate::traits::flatten::Flatten;
use crate::traits::nested::Nested;
use crate::traits::shape::Shape;

impl<T: RawDataType> Tensor<T> {
    pub fn from<const D: usize>(data: impl Flatten<T> + Shape + Nested<{ D }>) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from() failed, found inhomogeneous dimensions"
        );

        let shape: Vec<usize> = data.shape().into();

        // calculates the stride from the tensor's shape
        // shape [5, 3, 2, 1] -> stride [10, 2, 1, 1]
        let mut stride = vec![0; D];
        let mut p = 1;

        for i in (0..D).rev() {
            stride[i] = p;
            p *= shape[i];
        }

        Self {
            data: DataOwned::from(data),
            shape,
            stride,
            ndims: D,
        }
    }
}

impl<T: RawDataType> TensorView<T> {
    pub(super) fn from<B>(tensor: &TensorBase<B>, offset: usize, shape: Vec<usize>, stride: Vec<usize>) -> Self
    where
        B: DataBuffer<DType=T>,
    {
        let ndims = shape.len();

        // let mut len = 1;
        // for i in 0..ndims {
        //     len += stride[i] * (shape[i] - 1);
        // }
        //
        // the following code is equivalent to the above loop
        let len = shape.iter().zip(stride.iter())
            .map(|(&axis_length, &axis_stride)| axis_stride * (axis_length - 1))
            .sum::<usize>() + 1;

        let data = DataView::from_buffer(&tensor.data, offset, len);

        TensorView {
            data,
            shape,
            stride,
            ndims,
        }
    }
}

impl<B: DataBuffer> From<&TensorBase<B>> for TensorView<B::DType> {
    fn from(value: &TensorBase<B>) -> Self {
        Self {
            data: value.data.to_view(),
            shape: value.shape.clone(),
            stride: value.stride.clone(),
            ndims: value.ndims,
        }
    }
}
