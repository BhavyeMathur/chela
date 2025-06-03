use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct MatrixVecBackwards<T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) matrix: Rc<NdArray<'static, T>>,
    pub(super) vector: Rc<NdArray<'static, T>>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for MatrixVecBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        call_next_backward!(grad.unsqueeze(1) * self.vector.as_ref().unsqueeze(0),
                            self.next_functions[0]);

        call_next_backward!(self.matrix.as_ref().T().matmul(grad),
                            self.next_functions[1]);
    }
}

impl<T: FloatDataType> MatrixVecBackwards<T> {
    pub(crate) fn new(matrix: &Tensor<T>, vector: &Tensor<T>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_functions: [matrix.grad_fn(), vector.grad_fn()],

            matrix: matrix.get_ndarray(),
            vector: vector.get_ndarray(),
        }))
    }
}
