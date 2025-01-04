use crate::axis::indexer::Indexer;

use crate::axis::Axis;
use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::is_contiguous;
use crate::tensor::flags::TensorFlags;
use crate::Tensor;


impl<'a, T: RawDataType> Tensor<'a, T> {
    pub fn slice_along<S>(&self, axis: Axis, index: S) -> Tensor<T>
    where
        S: Indexer,
    {
        let (shape, stride) = index.indexed_shape_and_stride(&axis, &self.shape, &self.stride);
        let offset = self.stride[axis.0] * index.index_of_first_element();

        // let mut len = 1;
        // for i in 0..ndims {
        //     len += stride[i] * (shape[i] - 1);
        // }
        //
        // the following code is equivalent to the above loop
        let len = shape.iter().zip(stride.iter())
            .map(|(&axis_length, &axis_stride)| axis_stride * (axis_length - 1))
            .sum::<usize>() + 1;

        let mut flags = self.flags - TensorFlags::Owned;
        if is_contiguous(&shape, &stride) {
            flags |= TensorFlags::Contiguous;
        } else {
            flags -= TensorFlags::Contiguous;
        }

        Tensor {
            ptr: unsafe { self.ptr.add(offset) },
            len,
            capacity: len,

            shape,
            stride,
            flags,

            _marker: self._marker,
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

        // let mut axis = 0;
        // let mut ndims = self.ndims();
        let mut result = self.view();

        // for idx in index {
        //     result = result.slice_along(Axis(axis), idx.clone());
        //
        //     if result.ndims() == ndims {
        //         axis += 1;
        //     } else {
        //         ndims = result.ndims();
        //     }
        // }
        result
    }
}
