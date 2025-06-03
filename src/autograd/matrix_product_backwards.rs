use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct MatrixProductBackwards<T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) lhs: Rc<NdArray<'static, T>>,
    pub(super) rhs: Rc<NdArray<'static, T>>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for MatrixProductBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        call_next_backward!(grad.matmul(self.rhs.as_ref().T()),
                            self.next_functions[0]);

        call_next_backward!(self.lhs.as_ref().T().matmul(grad),
                            self.next_functions[1]);
    }
}

impl<T: FloatDataType> MatrixProductBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],

            lhs: lhs.get_ndarray(),
            rhs: rhs.get_ndarray(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
