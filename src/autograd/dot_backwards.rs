use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct DotBackwards<'a, T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) lhs_grad: NdArray<'a, T>,
    pub(super) rhs_grad: NdArray<'a, T>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for DotBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = &self.lhs_grad * grad;
        let rhs_grad = &self.rhs_grad * grad;

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> DotBackwards<'static, T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],
            lhs_grad: rhs.detach(),
            rhs_grad: lhs.detach()
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
