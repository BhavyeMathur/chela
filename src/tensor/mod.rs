pub mod methods;
pub mod ops;
pub mod constructors;
pub mod equals;
pub mod autograd;
pub mod print;

use crate::gradient_function::GradientFunction;
use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArray, TensorDataType};

pub struct Tensor<'a, T: TensorDataType> {
    array: NdArray<'a, T>,

    pub(super) flags: NdArrayFlags,
    pub(super) grad_fn: GradientFunction<T>,
}
