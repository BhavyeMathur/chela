pub mod methods;
pub mod ops;
pub mod constructors;
pub mod equals;
pub mod autograd;
pub mod print;
pub mod matrix_ops;
pub mod reshape;

use std::marker::PhantomData;
use std::rc::Rc;
use crate::gradient_function::GradientFunction;
use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArray, TensorDataType};

pub struct Tensor<'a, T: TensorDataType> {
    array: Rc<NdArray<'static, T>>,

    pub(super) flags: NdArrayFlags,
    pub(super) grad_fn: GradientFunction<T>,

    _marker: PhantomData<&'a T>,
}
