use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::iterator::iterators::FlatIterator;
use crate::TensorBase;

impl<B1, T1, B2, T2> PartialEq<TensorBase<B1>> for TensorBase<B2>
where
    TensorBase<B1>: FlatIterator<T1>,
    TensorBase<B2>: FlatIterator<T2>,
    B1: DataBuffer<DType=T1>,
    B2: DataBuffer<DType=T2>,
    T1: RawDataType,
    T2: RawDataType + From<T1>,
{
    fn eq(&self, other: &TensorBase<B1>) -> bool {
        if self.shape != other.shape {
            return false;
        }
        self.flat_iter().zip(other.flat_iter()).all(|(a, b)| a == b.into())
    }
}
