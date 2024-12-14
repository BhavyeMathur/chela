use crate::data_buffer::DataBuffer;
use crate::data_view::DataView;
use crate::tensor::dtype::RawDataType;
use crate::{TensorBase, TensorView};

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
            data: DataView::from(&self.data),
            shape,
            stride,
            ndims,
        }
    }

    pub fn unsqueeze(&mut self, axis: usize) -> TensorView<T> {
        assert!(axis <= self.ndims, "dimension out of bounds");

        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        shape.insert(axis, 1);

        let mut p = 1;

        for i in (0..self.ndims).rev() {
            stride[i] = p;
            p *= shape[i];
        }

        TensorView {
            data: DataView::from(&self.data),
            shape,
            stride,
            ndims,
        }
    }
}
