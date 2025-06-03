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

    /// Whether the gradient function is NoneBackwards
    fn is_none(&self) -> bool {
        false
    }
}


pub(crate) type GradientFunction<T> = Rc<RefCell<dyn GradientFuncTrait<T>>>;

#[macro_export]
macro_rules! call_next_backward {
    ($grad:expr, $next:expr) => {
        if !$next.borrow().is_none() {
            $next.borrow_mut().backward(&$grad);
        }
    };
    
    ($grad:expr, $shape:expr, $next:expr) => {
        if !$next.borrow().is_none() {
            if $shape == $grad.shape() {
                $next.borrow_mut().backward(&$grad);
            } else {
                let grad = $grad;
                let next_grad = reduce_gradient(&grad, $shape);
                $next.borrow_mut().backward(&next_grad);
            };
        }
    };
}
