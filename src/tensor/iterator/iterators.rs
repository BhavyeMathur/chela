use crate::dtype::RawDataType;
use crate::iterator::flat_iterator::FlatIterator;
use crate::tensor_iterator::NdIterator;
use crate::traits::haslength::HasLength;
use crate::{AxisType, Tensor};

impl<T: RawDataType> Tensor<T> {
    pub fn flatiter(&self) -> FlatIterator<T> {
        FlatIterator::from(self)
    }
}

impl<T: RawDataType> Tensor<T> {
    pub fn iter(&self) -> NdIterator<T> {
        NdIterator::from(self, [0])
    }

    pub fn iter_along(&self, axis: impl AxisType) -> NdIterator<T> {
        NdIterator::from(self, [axis.usize()])
    }

    pub fn nditer(&self, axes: impl IntoIterator<Item=usize> + HasLength + Clone) -> NdIterator<T> {
        NdIterator::from(self, axes)
    }
}
