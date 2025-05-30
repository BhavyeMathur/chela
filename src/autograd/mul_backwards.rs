use crate::autograd::util::reduce_gradient;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct MulBackwards<'a, T: FloatDataType> {
    pub(super) next_functions: [GradientFunction<T>; 2],

    pub(super) lhs_grad: NdArray<'a, T>,
    pub(super) rhs_grad: NdArray<'a, T>,

    pub(super) lhs_shape: Vec<usize>,
    pub(super) rhs_shape: Vec<usize>
}

pub(crate) struct MulScalarBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
    scalar: T,
}


impl<T: FloatDataType> GradientFuncTrait<T> for MulBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = &self.lhs_grad * grad;
        let rhs_grad = &self.rhs_grad * grad;

        let lhs_grad = reduce_gradient(&lhs_grad, &self.lhs_shape);
        let rhs_grad = reduce_gradient(&rhs_grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for MulScalarBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad * self.scalar;
        let grad = reduce_gradient(&grad, &self.shape);

        self.next_function.borrow_mut().backward(&grad);
    }
}


impl<T: FloatDataType> MulBackwards<'static, T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let next_functions = [lhs.get_grad_fn(), rhs.get_grad_fn()];

        let grad_fn = Self {
            next_functions,
            lhs_grad: rhs.detach(),
            rhs_grad: lhs.detach(),

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> MulScalarBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: T) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: lhs.get_grad_fn(),
            shape: lhs.shape().to_vec(),
            scalar: rhs
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
