use crate::autograd::util::reduce_gradient;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{call_next_backward, FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;


pub(crate) struct MulBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs: Rc<NdArray<'static, T>>,
    rhs: Rc<NdArray<'static, T>>,
}

pub(crate) struct MulScalarBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
    scalar: T,
}


impl<T: FloatDataType> GradientFuncTrait<T> for MulBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = self.rhs.as_ref() * grad;
        let rhs_grad = self.lhs.as_ref() * grad;

        call_next_backward!(lhs_grad, self.lhs.shape(), self.next_functions[0]);
        call_next_backward!(rhs_grad, self.rhs.shape(), self.next_functions[1]);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for MulScalarBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad * self.scalar;
        call_next_backward!(grad, &self.shape, self.next_function);
    }
}


impl<T: FloatDataType> MulBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_functions: [lhs.grad_fn(), rhs.grad_fn()],
            lhs: lhs.get_ndarray(),
            rhs: rhs.get_ndarray(),
        }))
    }
}

impl<T: FloatDataType> MulScalarBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: T) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_function: lhs.grad_fn(),
            shape: lhs.shape().to_vec(),
            scalar: rhs
        }))
    }
}
