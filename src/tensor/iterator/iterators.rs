use crate::dtype::RawDataType;
use crate::iterator::flat_iterator::FlatIterator;
use crate::tensor_iterator::NdIterator;
use crate::Tensor;
use crate::axes_traits::{AxesType, AxisType};
use crate::buffer_iterator::BufferIterator;

impl<T: RawDataType> Tensor<'_, T> {
    pub fn flatiter(&self) -> FlatIterator<T> {
        FlatIterator::from(self)
    }

    pub fn flatiter_ptr(&self) -> BufferIterator<T> {
        BufferIterator::from(self)
    }
}

impl<T: RawDataType> Tensor<'_, T> {
    pub fn iter(&self) -> NdIterator<T> {
        NdIterator::from(self, [0])
    }

    pub fn iter_along(&self, axis: impl AxisType) -> NdIterator<T> {
        NdIterator::from(self, [axis.isize()])
    }

    pub fn nditer(&self, axes: impl AxesType) -> NdIterator<T> {
        NdIterator::from(self, axes)
    }
}
