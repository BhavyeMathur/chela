use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct BMMBackwards<'a, T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) lhs_transpose: NdArray<'a, T>,
    pub(super) rhs_transpose: NdArray<'a, T>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for BMMBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad_lhs = grad.bmm(&self.rhs_transpose);
        let grad_rhs = self.lhs_transpose.bmm(grad);

        self.next_functions[0].borrow_mut().backward(&grad_lhs);
        self.next_functions[1].borrow_mut().backward(&grad_rhs);
    }
}

impl<T: FloatDataType> BMMBackwards<'static, T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],

            lhs_transpose: lhs.detach().transpose(1, 2),
            rhs_transpose: rhs.detach().transpose(1, 2),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
