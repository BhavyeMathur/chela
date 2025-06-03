use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{NdArray, TensorDataType};
use std::cell::RefCell;
use std::rc::Rc;


/// The default backwards node for non-leaf NdArrays.
pub(crate) struct NoneBackwards {}

impl<T: TensorDataType> GradientFuncTrait<T> for NoneBackwards {
    /// Backwards method for ndarray with `requires_grad = false`, does nothing.
    fn backward(&mut self, _: &NdArray<T>) {}

    fn is_none(&self) -> bool {
        true
    }
}

impl NoneBackwards {
    pub(crate) fn new<T: TensorDataType>() -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {}))
    }
}
