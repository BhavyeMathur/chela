mod data_buffer;
pub mod data_owned;
pub mod data_view;
pub mod dtype;

pub mod shape;
mod constructors;

use crate::tensor::data_buffer::DataBuffer;
use crate::tensor::data_owned::DataOwned;
use crate::tensor::data_view::DataView;
use crate::tensor::dtype::RawDataType;

pub(crate) struct TensorBase<T: DataBuffer> {
    data: T,
    shape: Vec<usize>,
}

type Tensor<T: RawDataType> = TensorBase<DataOwned<T>>;
type TensorView<T: RawDataType> = TensorBase<DataView<T>>;
