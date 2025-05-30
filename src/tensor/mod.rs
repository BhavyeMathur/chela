mod flags;

pub mod methods;
pub mod ops;
pub mod constructors;
pub mod equals;

use crate::gradient_function::GradientFunction;
use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArray, TensorDataType};

pub struct Tensor<'a, T: TensorDataType> {
    array: NdArray<'a, T>,

    flags: NdArrayFlags,
    grad_fn: GradientFunction<T>,
}