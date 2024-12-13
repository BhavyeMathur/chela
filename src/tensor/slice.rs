use crate::axis::index::Index;

use crate::tensor::dtype::RawDataType;
use crate::tensor::{Tensor, TensorView};

use crate::axis::Axis;

impl<T> Tensor<T>
where
    T: RawDataType,
{
    pub fn slice_along<S: Index>(&self, axis: Axis, index: S) -> TensorView<T> {
        let (shape, stride) = index.indexed_shape_and_stride(&axis, &self.shape, &self.stride);
        let offset = self.stride[axis.0] * index.index_of_first_element();

        TensorView::from(&self, offset, shape, stride)
    }

    // pub fn slice<S, const N: usize>(&self, index: [S; N]) -> TensorView<T>
    // where
    //     S: Index,
    // {
    //     let slice_dims = self.ndims - N;
    //     let shape = vec![0; slice_dims];
    //     let stride = vec![0; slice_dims];
    //
    //     TensorView::from(&self, shape, stride)
    // }
}