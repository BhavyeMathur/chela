use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;
use crate::tensor::dtype::RawDataType;

pub trait DataBuffer {}

impl<T: RawDataType> DataBuffer for DataOwned<T> {}
impl<T: RawDataType> DataBuffer for DataView<T> {}
