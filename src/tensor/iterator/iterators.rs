use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::iterator::buffer_iterator::BufferIterator;
use crate::iterator::flat_index_generator::FlatIndexGenerator;
use crate::tensor_iterator::TensorIterator;
use crate::traits::haslength::HasLength;
use crate::{AxisType, Tensor, TensorBase, TensorView};
use std::ops::Range;

pub trait FlatIterator<T: RawDataType> {
    type Indices: Iterator<Item=isize>;
    fn flat_iter(&self) -> BufferIterator<T, Self::Indices>;
}

impl<T: RawDataType> FlatIterator<T> for Tensor<T> {
    type Indices = Range<isize>;
    fn flat_iter(&self) -> BufferIterator<T, Self::Indices> {
        BufferIterator::from(self, 0..self.size() as isize)
    }
}

impl<T: RawDataType> FlatIterator<T> for TensorView<T> {
    type Indices = FlatIndexGenerator;
    fn flat_iter(&self) -> BufferIterator<T, Self::Indices> {
        let indices = FlatIndexGenerator::from(&self.shape, &self.stride);
        BufferIterator::from(self, indices)
    }
}

impl<B, T> TensorBase<B>
where
    B: DataBuffer<crate::tensor::data_buffer::buffer::DType=T>,
    T: RawDataType,
{
    pub fn iter(&self) -> TensorIterator<T> {
        TensorIterator::from(self, [0])
    }

    pub fn iter_along(&self, axis: impl AxisType) -> TensorIterator<T> {
        TensorIterator::from(self, [axis.usize()])
    }

    pub fn nditer(&self, axes: impl IntoIterator<Item=usize> + HasLength + Clone) -> TensorIterator<T> {
        TensorIterator::from(self, axes)
    }
}
