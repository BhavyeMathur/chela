use crate::data_buffer::{DataBuffer, DataOwned};
use crate::dtype::RawDataType;
use crate::iterator::iterator_base::IteratorBase;
use crate::{Axis, Tensor};

pub trait TensorIterator<T: RawDataType> {
    type Buffer: DataBuffer<DType = T>;
    fn iter(&self, axis: Axis) -> IteratorBase<T, Self::Buffer>;
}

impl<T: RawDataType> TensorIterator<T> for Tensor<T> {
    type Buffer = DataOwned<T>;
    fn iter(&self, axis: Axis) -> IteratorBase<T, Self::Buffer> {
        assert!(
            axis.0 < self.ndims,
            "Axis must be smaller than number of dimensions!"
        );
        IteratorBase::from(
            &self.data,
            axis.0,
            self.shape.clone(),
            self.stride.clone(),
            self.size() / self.shape[axis.0],
        )
    }
}
