use crate::axis::indexer::Indexer;

use crate::axis::Axis;
use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::is_contiguous;
use crate::tensor::flags::TensorFlags;
use crate::Tensor;


impl<'a, T: RawDataType> Tensor<'a, T> {
    pub fn slice_along<S: Indexer>(&self, axis: Axis, index: S) -> Tensor<T> {
        let axis = axis.0;

        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        if index.collapse_dimension() {
            shape.remove(axis);
            stride.remove(axis);
        }
        else {
            shape[axis] = index.indexed_length(shape[axis]);
        }

        let offset = self.stride[axis] * index.index_of_first_element();

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

        let ndims = self.ndims();
        let mut offset = 0;
        let mut i = 0;

        let mut new_shape = Vec::with_capacity(ndims);
        let mut new_stride = Vec::with_capacity(ndims);

        for idx in index {
            offset += self.stride[i] * idx.index_of_first_element();

            if idx.collapse_dimension() {} else {
                let new_length = idx.indexed_length(self.shape[i]);
                // TODO what if new_length is 0?
                new_shape.push(new_length);
                new_stride.push(self.stride[i]);
            }

            i += 1;
        }

        for j in i..ndims {
            new_shape.push(self.shape[j]);
            new_stride.push(self.stride[j]);
        }

        // let mut len = 1;
        // for i in 0..ndims {
        //     len += stride[i] * (shape[i] - 1);
        // }
        //
        // the following code is equivalent to the above loop
        let len = new_shape.iter().zip(new_stride.iter())
            .map(|(&axis_length, &axis_stride)| axis_stride * (axis_length - 1))
            .sum::<usize>() + 1;

        let mut flags = self.flags - TensorFlags::Owned;
        if is_contiguous(&new_shape, &new_stride) {
            flags |= TensorFlags::Contiguous;
        } else {
            flags -= TensorFlags::Contiguous;
        }

        Tensor {
            ptr: unsafe { self.ptr.add(offset) },
            len,
            capacity: len,

            shape: new_shape,
            stride: new_stride,
            flags,

            _marker: self._marker,
        }
    }
}
