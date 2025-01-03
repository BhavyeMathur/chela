use crate::axis::indexer::Indexer;

use crate::axis::Axis;
use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::is_contiguous;
use crate::tensor::flags::TensorFlags;
use crate::Tensor;


impl<T: RawDataType> Tensor<T> {
    pub fn slice_along<S>(&self, axis: Axis, index: S) -> Tensor<T>
    where
        S: Indexer,
    {
        let (shape, stride) = index.indexed_shape_and_stride(&axis, &self.shape, &self.stride);
        let offset = self.stride[axis.0] * index.index_of_first_element();

        let mut flags = self.flags - TensorFlags::Owned;
        if is_contiguous(&shape, &stride) {
            flags |= TensorFlags::Contiguous;
        }
        else {
            flags -= TensorFlags::Contiguous;
        }

        Tensor {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len,
            capacity: self.capacity,

            shape,
            stride,
            flags,
        }
    }

    pub fn slice<S, I>(&self, index: I) -> Tensor<T>
    where
        S: Indexer,
        I: IntoIterator<Item=S>,
    {
        // repeatedly calls the slice_along() method for each element in the index
        // we keep a track of which axis to slice along using the axis variable
        // if the dimension of the tensor is preserved during the slice along, we increment the axis
        // otherwise the axis isn't incremented because the previous dimension has collapsed

        let mut axis = 0;
        let mut ndims = self.ndims();
        let mut result = self.copy_view();

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
