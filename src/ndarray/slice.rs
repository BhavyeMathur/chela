use crate::axis::Axis;
use crate::dtype::RawDataType;
use crate::index::Indexer;
use crate::iterator::collapse_contiguous::has_uniform_stride;
use crate::ndarray::flags::NdArrayFlags;
use crate::{AxisType, NdArray, StridedMemory};
use crate::gradient_function::NoneBackwards;

pub(super) fn update_flags_with_contiguity(mut flags: NdArrayFlags, shape: &[usize], stride: &[usize]) -> NdArrayFlags {
    match has_uniform_stride(shape, stride) {
        None => {
            flags -= NdArrayFlags::UniformStride;
            flags -= NdArrayFlags::Contiguous;
        }
        Some(stride) => {
            flags |= NdArrayFlags::UniformStride;

            if stride <= 1 {
                flags |= NdArrayFlags::Contiguous;
            } else {
                flags -= NdArrayFlags::Contiguous;
            }
        }
    }

    flags
}

fn calculate_strided_buffer_length(shape: &[usize], stride: &[usize]) -> usize {
    // let mut len = 1;
    // for i in 0..ndims {
    //     len += stride[i] * (shape[i] - 1);
    // }
    //
    // the following code is equivalent to the above loop
    shape.iter().zip(stride.iter())
         .map(|(&axis_length, &axis_stride)| axis_stride * (axis_length - 1))
         .sum::<usize>() + 1
}


impl<'a, T: RawDataType> NdArray<'a, T> {
    pub fn slice_along<S: Indexer>(&self, axis: Axis, index: S) -> NdArray<'a, T>
    {
        let axis = axis.get_absolute(self.ndims());

        let mut new_shape = self.shape.clone();
        let mut new_stride = self.stride.clone();

        if index.collapse_dimension() {
            new_shape.remove(axis);
            new_stride.remove(axis);
        } else {
            new_shape[axis] = index.indexed_length(new_shape[axis]);
        }

        let offset = self.stride[axis] * index.index_of_first_element();

        let len = calculate_strided_buffer_length(&new_shape, &new_stride);
        
        let mut flags = update_flags_with_contiguity(self.flags, &new_shape, &new_stride);
        flags -= NdArrayFlags::Owned;
        flags -= NdArrayFlags::UserCreated;

        NdArray {
            ptr: unsafe { self.ptr.add(offset) },
            len,
            capacity: len,

            shape: new_shape,
            stride: new_stride,
            flags,

            _marker: self._marker,
        }
    }

    pub fn slice<S, I>(&self, index: I) -> NdArray<'a, T>
    where
        S: Indexer,
        I: IntoIterator<Item=S>,
    {
        let ndims = self.ndims();
        let mut offset = 0;
        let mut axis = 0;

        let mut new_shape = Vec::with_capacity(ndims);
        let mut new_stride = Vec::with_capacity(ndims);

        for idx in index {
            if !idx.collapse_dimension() {
                let new_length = idx.indexed_length(self.shape[axis]);
                new_shape.push(new_length);
                new_stride.push(self.stride[axis]);
            }

            offset += self.stride[axis] * idx.index_of_first_element();
            axis += 1;
        }

        for j in axis..ndims {
            new_shape.push(self.shape[j]);
            new_stride.push(self.stride[j]);
        }

        let len = calculate_strided_buffer_length(&new_shape, &new_stride);
        let mut flags = update_flags_with_contiguity(self.flags, &new_shape, &new_stride);
        flags -= NdArrayFlags::Owned;
        flags -= NdArrayFlags::UserCreated;

        NdArray {
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
