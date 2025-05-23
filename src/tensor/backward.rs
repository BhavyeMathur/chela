use std::cell::RefCell;
use std::rc::Rc;
use crate::{RawDataType, Tensor};
use crate::gradient_function::GradientFunction;
use crate::intro_gradient::IntoGradient;

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub(crate) fn set_grad<'b>(&mut self, gradient: &Tensor<'b, T>) {
        let gradient = gradient.clone();
        self.grad = Some(Rc::new(RefCell::new(gradient)));
    }

    pub(crate) fn get_grad_fn<'b>(&self) -> GradientFunction<T> {
        self.grad_fn.clone()
    }

    pub fn gradient(&self) -> Option<Tensor<'a, T>> {
        match self.grad {
            None => None,
            Some(ref grad) => Some(grad.borrow().clone())
        }
    }
    
    pub fn backward(&mut self, gradient: impl IntoGradient<'a, T>) {
        let gradient = gradient.as_tensor();
        self.grad_fn.borrow_mut().backward(&gradient);
    }
}