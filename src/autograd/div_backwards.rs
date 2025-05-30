use crate::gradient_function::GradientFunction;
use crate::mul_backwards::{MulBackwards, MulScalarBackwards};
use crate::{FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct DivBackwards {}
pub(crate) struct DivScalarBackwards {}


impl DivBackwards {
    pub(crate) fn new<T: FloatDataType>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let next_functions = [lhs.grad_fn(), rhs.grad_fn()];

        let lhs = lhs.detach();
        let rhs = rhs.detach();

        let grad_fn = MulBackwards {
            next_functions,

            lhs_grad: NdArray::scalar(T::one()) / &rhs,
            rhs_grad: -&lhs / (&rhs * &rhs),

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl DivScalarBackwards {
    pub(crate) fn new<T: FloatDataType>(lhs: &Tensor<T>, rhs: T) -> GradientFunction<T> {
        MulScalarBackwards::new(lhs, T::one() / rhs)
    }
}
