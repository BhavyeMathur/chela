use crate::autograd::util::reduce_gradient;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct NegBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,
    shape: Vec<usize>,
}

impl<T: FloatDataType> GradientFuncTrait<T> for NegBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        call_next_backward!(-grad, &self.shape, self.next_function);
    }
}

impl<T: FloatDataType> NegBackwards<T> {
    pub(crate) fn new(rhs: &Tensor<T>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_function: rhs.grad_fn(),
            shape: rhs.shape().to_vec(),
        }))
    }
}
