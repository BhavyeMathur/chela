use crate::{NdArray, TensorDataType};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) trait GradientFuncTrait<T: TensorDataType> {
    /// Computes the gradient of this function with respect to its sources using the chain rule.
    ///
    /// # Parameters
    ///
    /// - `grad`: the gradient of the function being differentiated with respect to `self`.
    fn backward(&mut self, grad: &NdArray<T>);

    /// Returns the gradient of the function being differentiated with respect to `self`
    /// if this function is a leaf. Otherwise, returns `None`.
    fn gradient(&self) -> Option<NdArray<T>> {
        None
    }

    /// Sets the gradient of this tensor to zero.
    fn zero_gradient(&mut self) {}
}


pub(crate) type GradientFunction<T> = Rc<RefCell<dyn GradientFuncTrait<T>>>;
