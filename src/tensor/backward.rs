use crate::{RawDataType, Tensor};
use crate::gradient_function::GradientFunction;
use crate::intro_gradient::IntoTensor;

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub(crate) fn get_grad_fn<'b>(&self) -> GradientFunction<T> {
        self.grad_fn.clone()
    }

    pub fn gradient(&self) -> Option<Tensor<T>> {
        self.grad_fn.borrow().gradient()
    }
    
    pub fn backward(&self, gradient: impl IntoTensor<'a, T>) {
        let gradient = gradient.as_tensor();
        self.grad_fn.borrow_mut().backward(&gradient);
    }
}