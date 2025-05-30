use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct IdentityBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,
}

impl<T: FloatDataType> GradientFuncTrait<T> for IdentityBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        self.next_function.borrow_mut().backward(grad);
    }
}

impl<T: FloatDataType> IdentityBackwards<T> {
    pub(crate) fn new(tensor: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self { next_function: tensor.get_grad_fn() };
        Rc::new(RefCell::new(grad_fn))
    }
}
