use crate::axis::indexer::Indexer;

use crate::tensor::TensorView;

use crate::axis::Axis;
use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::TensorBase;

impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType=T>,
    T: RawDataType,
{
    pub fn slice_along<S>(&self, axis: Axis, index: S) -> TensorView<T>
    where
        S: Indexer,
    {
        let (shape, stride) = index.indexed_shape_and_stride(&axis, &self.shape, &self.stride);
        let offset = self.stride[axis.0] * index.index_of_first_element();

        TensorView::from(self, offset, shape, stride)
    }

    pub fn slice<S, I>(&self, index: I) -> TensorView<T>
    where
        S: Indexer,
        I: IntoIterator<Item=S>,
    {
        // repeatedly calls the slice_along() method for each element in the index
        // we keep a track of which axis to slice along using the axis variable
        // if the dimension of the tensor is preserved during the slice along, we increment the axis
        // otherwise the axis isn't incremented because the previous dimension has collapsed

        let mut axis = 0;
        let mut ndims = self.ndims;
        let mut result: TensorView<T> = self.into();

        for idx in index {
            result = result.slice_along(Axis(axis), idx.clone());

            if result.ndims() == ndims {
                axis += 1;
            } else {
                ndims = result.ndims();
            }
        }
        result
    }
}
