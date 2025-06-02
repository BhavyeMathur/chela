use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{AxisType, FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct TransposeBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    axis1: isize,
    axis2: isize,
}

impl<T: FloatDataType> GradientFuncTrait<T> for TransposeBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad.transpose(self.axis1, self.axis2);
        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> TransposeBackwards<T> {
    pub(crate) fn new(tensor: &Tensor<T>, axis1: impl AxisType, axis2: impl AxisType) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            next_function: tensor.grad_fn(),
            axis1: axis1.isize(),
            axis2: axis2.isize(),
        }))
    }
}
