use crate::autograd::util::reduce_gradient;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::identity_backwards::IdentityBackwards;
use crate::{FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct AddBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>
}

pub(crate) struct AddScalarBackwards {}

impl<T: FloatDataType> GradientFuncTrait<T> for AddBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = reduce_gradient(grad, &self.lhs_shape);
        let rhs_grad = reduce_gradient(grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> AddBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec()
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl AddScalarBackwards {
    pub(crate) fn new<T: FloatDataType>(lhs: &Tensor<T>, _: T) -> GradientFunction<T> {
        IdentityBackwards::new(lhs)
    }
}
