use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::util::to_vec::ToVec;
use crate::{call_next_backward, FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct ReshapeBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,
    shape: Vec<usize>,
}

impl<T: FloatDataType> GradientFuncTrait<T> for ReshapeBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad.reshape(&self.shape);
        call_next_backward!(grad, self.next_function);
    }
}

impl<T: FloatDataType> ReshapeBackwards<T> {
    pub(crate) fn new(tensor: &Tensor<T>, old_shape: impl ToVec<usize>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_function: tensor.grad_fn(),
            shape: old_shape.to_vec(),
        }))
    }
}
