use crate::autograd::util::reduce_gradient;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct SubBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>
}


impl<T: FloatDataType> GradientFuncTrait<T> for SubBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let rhs_grad = -grad;

        call_next_backward!(grad, &self.lhs_shape, self.next_functions[0]);
        call_next_backward!(rhs_grad, &self.rhs_shape, self.next_functions[1]);
    }
}


impl<T: FloatDataType> SubBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec()
        }))
    }
}
