use crate::dtype::RawDataType;
use crate::iterator::flat_iterator::FlatIterator;
use crate::tensor_iterator::NdIterator;
use crate::NdArray;
use crate::axis::{AxesType, AxisType};
use crate::buffer_iterator::BufferIterator;

impl<T: RawDataType> NdArray<'_, T> {
    pub fn flatiter(&self) -> FlatIterator<T> {
        FlatIterator::from(self)
    }

    pub fn flatiter_ptr(&self) -> BufferIterator<T> {
        BufferIterator::from(self)
    }
}

impl<'a, T: RawDataType> NdArray<'a, T> {
    pub fn iter(&'a self) -> NdIterator<'a, T> {
        NdIterator::from(self, [0])
    }

    pub fn iter_along(&'a self, axis: impl AxisType) -> NdIterator<'a, T> {
        NdIterator::from(self, [axis.get_absolute(self.shape.len())])
    }

    pub fn nditer(&'a self, axes: impl AxesType) -> NdIterator<'a, T> {
        NdIterator::from(self, axes)
    }
}
