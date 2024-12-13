use crate::data_view::DataView;
use crate::tensor::data_owned::DataOwned;
use crate::tensor::dtype::RawDataType;
use crate::tensor::{Tensor, TensorView};
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
    pub(crate) fn from(tensor: &Tensor<T>, offset: usize, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        let ndims = shape.len();
        let len = shape.iter().product();

        let data = DataView::from_owned(&tensor.data, offset, len);

        TensorView {
            data,
            shape,
            stride,
            ndims,
        }
    }
}
