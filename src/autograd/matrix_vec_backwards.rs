use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct MatrixVecBackwards<'a, T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) matrix_transpose: NdArray<'a, T>,
    pub(super) vector: NdArray<'a, T>,
}


impl<T: FloatDataType> GradientFuncTrait<T> for MatrixVecBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let matrix_grad = grad.unsqueeze(1) * &self.vector;
        let vector_grad = self.matrix_transpose.matmul(grad);

        self.next_functions[0].borrow_mut().backward(&matrix_grad);
        self.next_functions[1].borrow_mut().backward(&vector_grad);
    }
}

impl<T: FloatDataType> MatrixVecBackwards<'static, T> {
    pub(crate) fn new(matrix: &Tensor<T>, vector: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [matrix.grad_fn(), vector.grad_fn()],

            matrix_transpose: matrix.detach().T(),
            vector: vector.detach().unsqueeze(0),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
