use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;
use crate::tensor::dtype::RawDataType;

pub trait DataBuffer {}

// Two kinds of data buffers
// DataOwned: owns its data & responsible for cleaning it up
// DataView: reference to data owned by another buffer

impl<T: RawDataType> DataBuffer for DataOwned<T> {}
impl<T: RawDataType> DataBuffer for DataView<T> {}
