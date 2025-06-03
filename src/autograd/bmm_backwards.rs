use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct BMMBackwards<T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) lhs: Rc<NdArray<'static, T>>,
    pub(super) rhs: Rc<NdArray<'static, T>>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for BMMBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = grad.bmm(self.rhs.as_ref().transpose(1, 2));
        let rhs_grad = self.lhs.as_ref().transpose(1, 2).bmm(grad);

        call_next_backward!(lhs_grad, self.next_functions[0]);
        call_next_backward!(rhs_grad, self.next_functions[1]);
    }
}

impl<T: FloatDataType> BMMBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],
            lhs: lhs.get_ndarray(),
            rhs: rhs.get_ndarray()
        }))
    }
}
