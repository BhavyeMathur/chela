use crate::autograd::util::reduce_gradient;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct NegBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
}

impl<T: FloatDataType> GradientFuncTrait<T> for NegBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = -grad;
        let grad = reduce_gradient(&grad, &self.shape);

        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> NegBackwards<T> {
    pub(crate) fn new(rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: rhs.get_grad_fn(),
            shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
