use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct MatrixProductBackwards<'a, T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) lhs_transpose: NdArray<'a, T>,
    pub(super) rhs_transpose: NdArray<'a, T>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for MatrixProductBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        call_next_backward!(grad.matmul(&self.rhs_transpose),
                            self.next_functions[0]);

        call_next_backward!(self.lhs_transpose.matmul(grad),
                            self.next_functions[1]);
    }
}

impl<T: FloatDataType> MatrixProductBackwards<'static, T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],

            lhs_transpose: lhs.detach().T(),
            rhs_transpose: rhs.detach().T(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
